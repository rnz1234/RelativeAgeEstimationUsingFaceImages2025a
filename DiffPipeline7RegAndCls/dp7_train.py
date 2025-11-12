import copy
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import seaborn as sn
import pandas as pd
import numpy as np

from condor_pytorch.dataset import levels_from_labelbatch
from condor_pytorch.losses import CondorOrdinalCrossEntropy #condor_negloglikeloss
from condor_pytorch.dataset import logits_to_label
from condor_pytorch.activations import ordinal_softmax
from condor_pytorch.metrics import earth_movers_distance
from condor_pytorch.metrics import ordinal_accuracy
from condor_pytorch.metrics import mean_absolute_error

import dp7_config as cfg

# Modifed from: https://github.com/tensorflow/tensorflow/blob/6dcd6fcea73ad613e78039bd1f696c35e63abb32/tensorflow/python/ops/nn_impl.py#L112-L148
def condor_negloglikeloss_ext(logits, labels, device, reduction='mean'):
    """computes the negative log likelihood loss described in
    condor tbd.
    parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        outputs of the condor layer.
    labels : torch.tensor, shape(num_examples, num_classes-1)
        true labels represented as extended binary vectors
        (via `condor_pytorch.dataset.levels_from_labelbatch`).
    reduction : str or none (default='mean')
        if 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. if none, returns a vector of
        shape (num_examples,)
    returns
    ----------
        loss : torch.tensor
        a torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=none`).
    examples
    ----------
    >>> import torch
    >>> labels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> condor_negloglikeloss(logits, labels)
    tensor(0.4936)
    """
    if not logits.shape == labels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as labels (%s). "
                         % (logits.shape, labels.shape))
    piLab = torch.cat([torch.ones((labels.shape[0],1)).to(device),labels[:,:-1]],dim=1)

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    cond2 = (piLab > zeros)
    relu_logits = torch.where(cond, logits, zeros)
    neg_abs_logits = torch.where(cond, -logits, logits)
    temp = relu_logits - (logits*labels) + torch.log1p(torch.exp(neg_abs_logits))
    val = torch.sum(torch.where(cond2, temp.double(), zeros.double()),dim=1)
    #val = torch.sum(torch.where(cond2, temp.half(), zeros.half()),dim=1)

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss

def validate(model, val_data_loader, iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_size, writer, use_cls_mean_var):
    print('running on validation set...')

    model.eval()

    running_loss = 0.0
    running_mae = 0.0
    running_corrects = 0.0
    running_p1_error = 0.0
    running_p2_and_above_error = 0.0

    for i, batch in enumerate(val_data_loader):
        inputs = batch['image_vec'].to(device)
        classification_labels = batch['label'].to(device).float()
        ages = batch['age_diff'].to(device).float()

        with torch.no_grad():
            classification_logits, age_pred = model(inputs)
            if cfg.IS_ORDINAL:
                class_preds = logits_to_label(classification_logits).float()
            else:
                _, class_preds = torch.max(classification_logits, 1)


        reg_loss = criterion_reg(age_pred, ages)
        loss = reg_loss
        if use_cls_mean_var:
            if cfg.IS_ORDINAL:
                levels = levels_from_labelbatch(classification_labels.int(), num_classes=2*cfg.AGE_DIFF_LEARN_RADIUS_HI+1)
                levels = levels.to(device)
                
                cls_loss = condor_negloglikeloss_ext(classification_logits, levels, device)
                #cls_loss = CondorOrdinalCrossEntropy(classification_logits, levels)
            else:
                cls_loss = criterion_cls(classification_logits, classification_labels.long())
            mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
            loss += cls_loss + mean_loss + var_loss

        running_loss += loss.item() * inputs.size(0)
        running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

        if use_cls_mean_var:
            classification_offset = torch.abs(class_preds - classification_labels.data)
            running_corrects += torch.sum(classification_offset == 0)
            running_p1_error += torch.sum(classification_offset == 1)
            running_p2_and_above_error += torch.sum(classification_offset >= 2)

    epoch_loss = running_loss / dataset_size
    epoch_mae = running_mae / dataset_size

    if use_cls_mean_var:
        epoch_acc = running_corrects.double() / dataset_size
        epoch_p1_error = running_p1_error.double() / dataset_size
        epoch_p2_and_above_error = running_p2_and_above_error.double() / dataset_size

    writer.add_scalar('Loss/val', epoch_loss, iter)
    writer.add_scalar('Mae/val', epoch_mae, iter)

    if use_cls_mean_var:
        writer.add_scalar('Accuracy/val', epoch_acc, iter)
        writer.add_scalar('Accuracy_+-1/val', epoch_p1_error, iter)
        writer.add_scalar('Accuracy_+-2_and_above/val', epoch_p2_and_above_error, iter)

    # writer.add_scalar('alpha', model.alpha.cpu().detach().numpy().squeeze(), iter)

    print('{} Loss: {:.4f} mae: {:.4f}'.format('val', epoch_loss, epoch_mae))

    return epoch_mae


def train_unified_model_iter(
    model,
    criterion_reg,
    criterion_cls,
    mean_var_criterion,
    optimizer,
    scheduler,
    data_loaders,
    dataset_sizes,
    device,
    writer,
    model_path,
    unfreeze_feature_ext_on_rlvnt_epoch,
    unfreeze_feature_ext_epoch,
    NumLabels=1,
    num_epochs=25,
    validate_at_k=100,
    use_cls_mean_var=True
):

    since = time.time()

    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = 100.0
    #epoch_mae = 100.0

    running_loss = 0.0
    running_mae = 0.0
    running_corrects = 0.0
    running_p1_error = 0.0
    running_p2_and_above_error = 0.0

    FINAL_MODEL_FILE = None

    iter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # if epoch == 15:
        # 	model.FreezeBaseCnn(False)

        if unfreeze_feature_ext_on_rlvnt_epoch:
            if epoch == unfreeze_feature_ext_epoch:
                print("UNFREEZING BACKBONE !")
                if cfg.MULTI_GPU:
                    model.module.freeze_base_cnn(False)
                else:
                    model.freeze_base_cnn(False)

        phase = 'train'
        for batch in tqdm(data_loaders[phase]):
            if iter % validate_at_k == 0 and iter != 0:
                norm = validate_at_k * inputs.size(0)

                epoch_loss = running_loss / norm
                epoch_mae = running_mae / norm

                if use_cls_mean_var:
                    epoch_acc = running_corrects.double() / norm
                    epoch_p1_error = running_p1_error.double() / norm
                    epoch_p2_and_above_error = running_p2_and_above_error.double() / norm

                writer.add_scalar('Loss/{}'.format(phase), epoch_loss, iter)
                writer.add_scalar('Mae/{}'.format(phase), epoch_mae, iter)

                if use_cls_mean_var:
                    writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, iter)
                    writer.add_scalar('Accuracy_+-1/{}'.format(phase), epoch_p1_error, iter)
                    writer.add_scalar('Accuracy_+-2_and_above/{}'.format(phase), epoch_p2_and_above_error, iter)

                print('{} Loss: {:.4f} mae: {:.4f}'.format(phase, epoch_loss, epoch_mae))

                val_mae = validate(model, data_loaders['val'], iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_sizes['val'], writer, use_cls_mean_var=use_cls_mean_var)

                # deep copy the model
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_model_wts = copy.deepcopy(model.state_dict())

                    #FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae))
                    
                    if FINAL_MODEL_FILE is not None:
                        print("removing previous weights")
                        os.system("rm -rf {}".format(FINAL_MODEL_FILE))
                    FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae))
                    if cfg.SAVE_ALL_MODEL_METADATA:
                        torch.save({
                            'model_state_dict' : best_model_wts,
                            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        }, FINAL_MODEL_FILE)
                    else:
                        torch.save(best_model_wts, FINAL_MODEL_FILE)
                
                model.train()

                running_loss = 0.0
                running_mae = 0.0
                running_corrects = 0.0
                running_p1_error = 0.0
                running_p2_and_above_error = 0.0

            iter += 1

            inputs = batch['image_vec'].to(device)
            classification_labels = batch['label'].to(device).float()
            ages = batch['age_diff'].to(device).float()

            optimizer.zero_grad()

            if cfg.APPLY_AMP:
                with autocast():
                    classification_logits, age_pred = model(inputs)

                    if cfg.IS_ORDINAL:
                        class_preds = logits_to_label(classification_logits).float()
                    else:
                        _, class_preds = torch.max(classification_logits, 1)


                    reg_loss = criterion_reg(age_pred, ages)
                    loss = reg_loss
                    if use_cls_mean_var:
                        if cfg.IS_ORDINAL:
                            levels = levels_from_labelbatch(classification_labels.int(), num_classes=2*cfg.AGE_DIFF_LEARN_RADIUS_HI+1)
                            levels = levels.to(device)
                            # import pdb
                            # pdb.set_trace()
                            cls_loss = condor_negloglikeloss_ext(classification_logits, levels, device)
                            #cls_loss = CondorOrdinalCrossEntropy(classification_logits, levels)
                        else:
                            cls_loss = criterion_cls(classification_logits, classification_labels.long())
                        mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
                        loss += cls_loss + mean_loss + var_loss

                # loss.backward()
                # optimizer.step()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # import pdb
                # pdb.set_trace()
                
                classification_logits, age_pred = model(inputs)
                if cfg.IS_ORDINAL:
                    class_preds = logits_to_label(classification_logits).float()
                else:
                    _, class_preds = torch.max(classification_logits, 1)

                reg_loss = criterion_reg(age_pred, ages)
                loss = reg_loss
                if use_cls_mean_var:
                    if cfg.IS_ORDINAL:
                        levels = levels_from_labelbatch(classification_labels.int(), num_classes=2*cfg.AGE_DIFF_LEARN_RADIUS_HI+1)
                        levels = levels.to(device)
                        # import pdb
                        # pdb.set_trace()
                        cls_loss = condor_negloglikeloss_ext(classification_logits, levels, device)
                        #cls_loss = CondorOrdinalCrossEntropy(classification_logits, levels)
                    else:
                        cls_loss = criterion_cls(classification_logits, classification_labels.long())
                    mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
                    loss += cls_loss + mean_loss + var_loss

                loss.backward()
                optimizer.step()


            running_loss += loss.item() * inputs.size(0)
            running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

            if use_cls_mean_var:
                classification_offset = torch.abs(class_preds - classification_labels.data)
                running_corrects += torch.sum(classification_offset == 0)
                running_p1_error += torch.sum(classification_offset == 1)
                running_p2_and_above_error += torch.sum(classification_offset >= 2)

            # scheduler.step(epoch_mae)
            if cfg.SCHEDULER_STEP_GRANULARITY != "epoch":
                scheduler.step()

        if cfg.SCHEDULER_STEP_GRANULARITY == "epoch":
            if cfg.SCHEDULER == "ReduceLROnPlateau+GradualWarmupScheduler":
                scheduler.step(epoch=epoch, metrics=best_mae) #epoch_mae)
                cur_lr = scheduler.get_lr()
                if isinstance(cur_lr, list):
                    if len(cur_lr) == 1:
                        cur_lr = cur_lr[0]
                    else:
                        print("lr is not one item - need to debug")
                        import pdb
                        pdb.set_trace()
                print("Current lr: {}".format(cur_lr))
                writer.add_scalar('lr/train', cur_lr, epoch)
            else:
                scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Mae: {:4f}'.format(best_mae))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model