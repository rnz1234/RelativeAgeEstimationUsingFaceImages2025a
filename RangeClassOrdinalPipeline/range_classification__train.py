import time
import copy
import os

import torch 
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

import range_classification__config as cfg


from Common.Analysis.DiffAnalysisMethods import diff_range_pipeline_confusion_matrix_analysis


import torch.nn.functional as F

from condor_pytorch.dataset import levels_from_labelbatch
from condor_pytorch.losses import CondorOrdinalCrossEntropy #condor_negloglikeloss
from condor_pytorch.dataset import logits_to_label
from condor_pytorch.activations import ordinal_softmax
from condor_pytorch.metrics import earth_movers_distance
from condor_pytorch.metrics import ordinal_accuracy
from condor_pytorch.metrics import mean_absolute_error

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

def validate(model, val_data_loader, iter, device, criterion, dataset_size, writer, group):
    print('running on validation ({}) set...'.format(group))

    running_loss = 0.0
    running_corrects = 0.0

    for i, batch in enumerate(val_data_loader):
        inputs = batch['image_vec'].to(device)
        labels = batch['label'].to(device).float()   

        with torch.no_grad():
            classification_logits, age_diff_pred_hard, _ = model(inputs)
            preds = age_diff_pred_hard
            


        levels = levels_from_labelbatch(labels.int(), num_classes=3)
        levels = levels.to(device)
        
        loss = condor_negloglikeloss_ext(classification_logits, levels, device)


        running_loss += loss.item() * inputs.size(0)

        classification_offset = torch.abs(preds - labels.data)
        running_corrects += torch.sum(classification_offset == 0)

    epoch_loss = running_loss / dataset_size

    writer.add_scalar('Loss/{}'.format(group), epoch_loss, iter)

    epoch_acc = running_corrects.double() / dataset_size

    writer.add_scalar('Accuracy/{}'.format(group), epoch_acc, iter)
        
        
    print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(group, epoch_loss, epoch_acc))

    return epoch_acc #, diffs_valid_actual



def train_diff_cls_model_iter(
		model,
        criterion,
        optimizer,
        scheduler,
        data_loaders,
        dataset_sizes,
        device,
        writer,
        model_path,
        num_epochs=25,
        unfreeze_feature_ext_epoch=15,
        unfreeze_feature_ext_on_rlvnt_epoch=False,
		validate_at_k=100,
        validate_at_end_of_epoch=True,
):

    since = time.time()

    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 #100.0

    running_loss = 0.0
    running_corrects = 0.0

    iter = 0

    acc_norm = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if unfreeze_feature_ext_on_rlvnt_epoch:
            if epoch == unfreeze_feature_ext_epoch:
                if cfg.MULTI_GPU:
                    model.module.freeze_base_cnn(False)
                else:
                    model.freeze_base_cnn(False)

        phase = 'train'
        for batch in tqdm(data_loaders[phase]):
            if iter % validate_at_k == 0 and iter != 0:
                #norm = validate_at_k * inputs.size(0)
                norm = acc_norm

                epoch_loss = running_loss / norm
                epoch_acc = running_corrects.double() / norm

                writer.add_scalar('Loss/{}'.format(phase), epoch_loss, iter)
                writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, iter)

                print('{} Loss: {:.4f} Accuracy: {:.4f}'.format('train', epoch_loss, epoch_acc))

                # print(running_mae_hard)
                # print(running_mae_soft)
                # print(norm)

            
                model.eval()
                for group in ['val_apref_ds']: #['val_qtst_rtst', 'val_qtst_rtrn', 'val_apref_ds']:
                    val_accuracy = -1 #validate(model, data_loaders[group], iter, device, criterion, dataset_sizes[group], writer, group)

                val_acc_taken = val_accuracy

                # deep copy the model
                if val_acc_taken > best_acc:
                    best_acc = val_acc_taken
                    best_model_wts = copy.deepcopy(model.state_dict())

                    FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_acc_taken))
                    torch.save(best_model_wts, FINAL_MODEL_FILE)

                model.train()

                running_loss = 0.0
                running_corrects = 0.0

                acc_norm = 0.0

            iter += 1

            # import pdb
            # pdb.set_trace()

            inputs = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
            labels = batch['label'].to(device).float() 

            # import pdb
            # pdb.set_trace()

            acc_norm += inputs.size(0)           

            optimizer.zero_grad()

            #with autocast():
            classification_logits, age_diff_pred_hard, _ = model(inputs)
        
            preds = age_diff_pred_hard
            
            levels = levels_from_labelbatch(labels.int(), num_classes=3)
            levels = levels.to(device)
            
            loss = condor_negloglikeloss_ext(classification_logits, levels, device)
    
            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            running_loss += loss.item() * inputs.size(0)
        
            classification_offset = torch.abs(preds - labels.data)
            running_corrects += torch.sum(classification_offset == 0)
                

            # scheduler.step(epoch_mae)
            if scheduler is not None:
                scheduler.step()

        

        if validate_at_end_of_epoch:
            
            model.eval()
            diff_range_pipeline_confusion_matrix_analysis(model, data_loaders['train'], device, dataset_sizes['train'])
            for group in ['val_apref_ds']: #['val_qtst_rtst', 'val_qtst_rtrn', 'val_apref_ds']:
                #val_accuracy = -1 #
                val_accuracy = validate(model, data_loaders[group], iter, device, criterion, dataset_sizes[group], writer, group)
                diff_range_pipeline_confusion_matrix_analysis(model, data_loaders[group], device, dataset_sizes[group])

            val_acc_taken = val_accuracy

            # deep copy the model
            if val_acc_taken > best_acc:
                best_acc = val_acc_taken
                best_model_wts = copy.deepcopy(model.state_dict())

                FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_acc_taken))
                torch.save(best_model_wts, FINAL_MODEL_FILE)

            model.train()


        print()

    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Mae: {:4f}'.format(best_mae))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model