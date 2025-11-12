import time
import copy
import os

import torch 
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score

import ep1_config as cfg


from Common.Analysis.DiffAnalysisMethods import diff_pipeline_confusion_matrix_analysis


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

def validate(model, val_data_loader, iter, device, criterion, criterion_in_range_est, criterion_age, dataset_size, writer, include_accuracy_metrics, is_ordinal_reg, group):
    print('running on validation ({}) set...'.format(group))


    running_loss = 0.0
    running_mae_hard = 0.0
    running_mae_hard_final = 0.0
    running_mae_soft = 0.0
    running_mae_age = 0.0
    running_loss_diff = 0.0
    running_loss_age = 0.0
    running_loss_in_range = 0.0

    for i, batch in enumerate(val_data_loader):
        inputs = batch['image_vec'].to(device)
        labels = batch['label'].to(device).float()
        age_diff = batch['age_diff'].to(device).float()
        age_ref = batch['age_ref'].to(device)
        ages = batch['age_query'].to(device).float()  

        with torch.no_grad():
            classification_logits, age_diff_pred_hard, age_diff_pred_soft, is_in_range_est, final_age_est = model(input_images=inputs, input_ref_age=age_ref)
            preds = age_diff_pred_hard
            
        levels = levels_from_labelbatch(labels.int(), num_classes=2*cfg.AGE_DIFF_LEARN_RADIUS_HI+1)
        levels = levels.to(device)
        
        loss_diff = condor_negloglikeloss_ext(classification_logits, levels, device)

        is_in_range = torch.logical_and(torch.abs(age_ref-ages) >= cfg.AGE_DIFF_LO, torch.abs(age_ref-ages) <= cfg.AGE_DIFF_HI).to(dtype=torch.float64)
        is_in_range_est_flat = is_in_range_est.view(-1)
        loss_in_range = criterion_in_range_est(is_in_range_est_flat, is_in_range)

        
        loss_age = criterion_age(final_age_est, ages)

        #loss = CondorOrdinalCrossEntropy(classification_logits, levels)

        loss = loss_diff + 0.01*loss_in_range + 0.001*loss_age

        
        #loss = CondorOrdinalCrossEntropy(classification_logits, levels)

        running_loss += loss.item() * inputs.size(0)

        running_loss_diff += loss_diff.item() * inputs.size(0)
        running_loss_age += loss_age.item() * inputs.size(0)
        running_loss_in_range += loss_in_range.item() * inputs.size(0)

        running_mae_hard += mean_absolute_error(classification_logits.double(), levels) * inputs.size(0)
        running_mae_soft += torch.nn.L1Loss()(age_diff_pred_soft, age_diff) * inputs.size(0)
        running_mae_hard_final += torch.nn.L1Loss()(age_diff_pred_hard.double()-cfg.AGE_DIFF_LEARN_RADIUS_HI, age_diff) * inputs.size(0)
            
        running_mae_age += torch.nn.L1Loss()(final_age_est, ages) * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    epoch_mae_hard = running_mae_hard / dataset_size
    epoch_mae_hard_final = running_mae_hard_final / dataset_size
    epoch_mae_soft = running_mae_soft / dataset_size
    epoch_mae_age = running_mae_age / dataset_size

    epoch_loss_diff = running_loss_diff / dataset_size
    epoch_loss_age = running_loss_age / dataset_size
    epoch_loss_in_range = running_loss_in_range / dataset_size

    epoch_auc = create_confusion_matrix(val_data_loader, model, device)

    writer.add_scalar('Loss/{}'.format(group), epoch_loss, iter)
    writer.add_scalar('LossDiff/{}'.format(group), epoch_loss_diff, iter)
    writer.add_scalar('LossAge/{}'.format(group), epoch_loss_age, iter)
    writer.add_scalar('LossInRange/{}'.format(group), epoch_loss_in_range, iter)
    writer.add_scalar('MaeDiff(Hard)/{}'.format(group), epoch_mae_hard, iter)
    writer.add_scalar('MaeDiff(Hard Final)/{}'.format(group), epoch_mae_hard_final, iter)
    writer.add_scalar('MaeDiff(Soft)/{}'.format(group), epoch_mae_soft, iter)
    writer.add_scalar('MaeAge/{}'.format(group), epoch_mae_age, iter)
    writer.add_scalar('RangeAUC/{}'.format(group), epoch_auc, iter)
        
    print('{} Loss: {:.4f} mae diff (hard): {:.4f} mae diff (hard final): {:.4f} mae diff (soft): {:.4f} mae age : {:.4f} auc : {:.4f}'.format('val', epoch_loss, epoch_mae_hard, epoch_mae_hard_final, epoch_mae_soft, epoch_mae_age, epoch_auc))

    return epoch_mae_hard, epoch_mae_hard_final, epoch_mae_soft, epoch_mae_age #, diffs_valid_actual



def train_diff_cls_model_iter(
		model,
        criterion,
        criterion_in_range_est, 
        criterion_age,
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
        validate_at_end_of_epoch=True
):

    since = time.time()

    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = 100.0

    running_loss = 0.0
    running_loss_diff = 0.0
    running_loss_age = 0.0
    running_loss_in_range = 0.0
    running_mae_hard = 0.0
    running_mae_hard_final = 0.0
    running_mae_soft = 0.0
    running_mae_age = 0.0

    iter = 0

    acc_norm = 0.0

    include_accuracy_metrics = False
    is_ordinal_reg = False

    FINAL_MODEL_FILE = None

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

                epoch_loss_diff = running_loss_diff / norm
                epoch_loss_age = running_loss_age / norm
                epoch_loss_in_range = running_loss_in_range / norm
                
                epoch_mae_hard = running_mae_hard / norm
                epoch_mae_hard_final = running_mae_hard_final / norm
                epoch_mae_soft = running_mae_soft / norm
                epoch_mae_age = running_mae_age / norm
                epoch_auc = create_confusion_matrix(data_loaders[phase], model, device)

                writer.add_scalar('Loss/{}'.format(phase), epoch_loss, iter)
                writer.add_scalar('LossDiff/{}'.format(phase), epoch_loss_diff, iter)
                writer.add_scalar('LossAge/{}'.format(phase), epoch_loss_age, iter)
                writer.add_scalar('LossInRange/{}'.format(phase), epoch_loss_in_range, iter)
                writer.add_scalar('MaeDiff(Hard)/{}'.format(phase), epoch_mae_hard, iter)
                writer.add_scalar('MaeDiff(Hard Final)/{}'.format(phase), epoch_mae_hard_final, iter)
                writer.add_scalar('MaeDiff(Soft)/{}'.format(phase), epoch_mae_soft, iter)
                writer.add_scalar('MaeAge/{}'.format(phase), epoch_mae_age, iter)
                writer.add_scalar('RangeAUC/{}'.format(phase), epoch_auc, iter)
                
                print('{} Loss: {:.4f} mae diff (hard): {:.4f} mae diff (hard final): {:.4f} mae diff (soft): {:.4f} mae age : {:.4f} auc : {:.4f}'.format('val', epoch_loss, epoch_mae_hard, epoch_mae_hard_final, epoch_mae_soft, epoch_mae_age, epoch_auc))

                # print(running_mae_hard)
                # print(running_mae_soft)
                # print(norm)

                
                model.eval()
                #print("running on train:")
                #diff_pipeline_confusion_matrix_analysis(model, data_loaders['train'], device, dataset_sizes['train'], cfg, is_ordinal_reg=True, is_pipeline=False)
                for group in ['val_apref_ds']: #['val_qtst_rtst', 'val_qtst_rtrn', 'val_apref_ds']:
                    #for group in ['val_apref_all_ds']:
                    val_mae_diff_hard, val_mae_diff_hard_final, val_mae_diff_soft, val_mae_age = validate(model, data_loaders[group], iter, device, criterion, criterion_in_range_est, criterion_age, dataset_sizes[group], writer, include_accuracy_metrics, is_ordinal_reg, group)

                #val_mae_taken = val_mae_hard_final

                # deep copy the model
                if val_mae_age < best_mae:
                    val_mae_taken = val_mae_age
                    best_mae = val_mae_taken
                    best_model_wts = copy.deepcopy(model.state_dict())

                    if FINAL_MODEL_FILE is not None:
                        print("removing previous weights")
                        os.system("rm -rf {}".format(FINAL_MODEL_FILE))
                    FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae_taken))
                    if cfg.SAVE_ALL_MODEL_METADATA:
                        torch.save({
                            'model_state_dict' : best_model_wts,
                            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                            'scheduler': scheduler.state_dict()
                        }, FINAL_MODEL_FILE)
                    else:
                        torch.save(best_model_wts, FINAL_MODEL_FILE)
                
                model.train()

                running_loss = 0.0
                running_loss_diff = 0.0
                running_loss_age = 0.0
                running_loss_in_range = 0.0
                running_mae_hard = 0.0
                running_mae_hard_final = 0.0 
                running_mae_soft = 0.0
                running_mae_age = 0.0

                acc_norm = 0.0

            iter += 1

            # import pdb
            # pdb.set_trace()

            inputs = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
            labels = batch['label'].to(device).float()
            age_diff = batch['age_diff'].to(device).float()   
            age_ref = batch['age_ref'].to(device)
            ages = batch['age_query'].to(device).float()   

            # import pdb
            # pdb.set_trace()
            

            acc_norm += inputs.size(0)           

            optimizer.zero_grad()

            #with autocast():
            classification_logits, age_diff_pred_hard, age_diff_pred_soft, is_in_range_est, final_age_est = model(input_images=inputs, input_ref_age=age_ref)
    

            preds = age_diff_pred_hard
            levels = levels_from_labelbatch(labels.int(), num_classes=2*cfg.AGE_DIFF_LEARN_RADIUS_HI+1)
            levels = levels.to(device)
            # import pdb
            # pdb.set_trace()

            
            
            loss_diff = condor_negloglikeloss_ext(classification_logits, levels, device)


            is_in_range = torch.logical_and(torch.abs(age_ref-ages) >= cfg.AGE_DIFF_LO, torch.abs(age_ref-ages) <= cfg.AGE_DIFF_HI).to(dtype=torch.float64)
            is_in_range_est_flat = is_in_range_est.view(-1)
            loss_in_range = criterion_in_range_est(is_in_range_est_flat, is_in_range)

            
            loss_age = criterion_age(final_age_est, ages)

            #loss = CondorOrdinalCrossEntropy(classification_logits, levels)

            loss = 0.1*loss_diff + loss_in_range + 0.01*loss_age

            # import pdb
            # pdb.set_trace()
    
            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            running_loss_diff += loss_diff.item() * inputs.size(0)
            running_loss_age += loss_age.item() * inputs.size(0)
            running_loss_in_range += loss_in_range.item() * inputs.size(0)

            running_loss += loss.item() * inputs.size(0)
        
            running_mae_hard += mean_absolute_error(classification_logits.double(), levels) * inputs.size(0)
            running_mae_hard_final += torch.nn.L1Loss()(age_diff_pred_hard.double()-cfg.AGE_DIFF_LEARN_RADIUS_HI, age_diff) * inputs.size(0)
            # import pdb
            # pdb.set_trace()
            running_mae_soft += torch.nn.L1Loss()(age_diff_pred_soft.double(), age_diff) * inputs.size(0)

            running_mae_age += torch.nn.L1Loss()(final_age_est, ages) * inputs.size(0)



            

            
            # import pdb
            # pdb.set_trace()
            if bool(torch.isinf(running_mae_soft)) or bool(torch.isinf(running_mae_hard)):
                # print("got inf !")
                # print("ABORTING")
                # exit()
                import pdb
                pdb.set_trace()

            # scheduler.step(epoch_mae)
            if scheduler is not None :
                scheduler.step()

        

        if validate_at_end_of_epoch:
            
            
            model.eval()
            #diff_pipeline_confusion_matrix_analysis(model, data_loaders['train'], device, dataset_sizes['train'], cfg, is_ordinal_reg=True, is_pipeline=False)
            for group in ['val_apref_ds']: #['val_qtst_rtst', 'val_qtst_rtrn', 'val_apref_ds']:
                #for group in ['val_apref_all_ds']:                 
                val_mae_diff_hard, val_mae_diff_hard_final, val_mae_diff_soft, val_mae_age = validate(model, data_loaders[group], iter, device, criterion, criterion_in_range_est, criterion_age, dataset_sizes[group], writer, include_accuracy_metrics, is_ordinal_reg, group)
                # print('running on validation ({}) set...'.format(group))
                # val_mae_hard, val_mae_hard_final, val_mae_soft = diff_pipeline_confusion_matrix_analysis(model, data_loaders[group], device, dataset_sizes[group], cfg, is_ordinal_reg=True, is_pipeline=False)
                
            #val_mae_taken = val_mae_hard_final
                
            # deep copy the model
            if val_mae_age < best_mae:
                val_mae_taken = val_mae_age
                best_mae = val_mae_taken
                best_model_wts = copy.deepcopy(model.state_dict())

                if FINAL_MODEL_FILE is not None:
                    print("removing previous weights")
                    os.system("rm -rf {}".format(FINAL_MODEL_FILE))
                FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae_taken))
                if cfg.SAVE_ALL_MODEL_METADATA:
                    torch.save({
                        'model_state_dict' : best_model_wts,
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'scheduler': scheduler.state_dict()
                    }, FINAL_MODEL_FILE)
                else:
                    torch.save(best_model_wts, FINAL_MODEL_FILE)
            
            
            model.train()


        #print()

    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Mae: {:4f}'.format(best_mae))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def create_confusion_matrix(loader, model, device):
    model.eval()
    y_pred = [] # save predction
    y_true = [] # save ground truth
    y_prob_of_1 = []

    # iterate over data
    for batch in tqdm(loader, position=0, leave=True):
        inputs = batch['image_vec'].to(device)
        #labels = batch['label'].to(device).float()
        #age_diff = batch['age_diff'].to(device).float()   
        age_ref = batch['age_ref'].to(device)
        ages = batch['age_query'].to(device).float()   
        #classification_labels = batch['label'].to(device).float()
        is_in_range_actual = torch.logical_and(torch.abs(age_ref-ages) >= cfg.AGE_DIFF_LO, torch.abs(age_ref-ages) <= cfg.AGE_DIFF_HI).to(dtype=torch.float64)
            
        age_ref = batch['age_ref'].to(device)

        with torch.no_grad():
            _, _, _, is_in_range_est, _ = model(input_images=inputs, input_ref_age=age_ref)

            pred_err_prob = torch.sigmoid(is_in_range_est)
            pred_err = torch.round(pred_err_prob)

        y_prob_of_1.extend(pred_err_prob.cpu().numpy().reshape(-1))
        pred_err = torch.round(pred_err_prob).cpu().numpy().reshape(-1)

        y_pred.extend(pred_err)  # save prediction

        y_true.extend(is_in_range_actual.cpu().numpy())  # save ground truth

    # fpr, tpr, thresholds = roc_curve(testy, yhat)

    # constant for classes
    classes = ["0", "1"]

    # import pdb
    # pdb.set_trace()

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=range(0,2))
    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes) # cf_matrix/np.sum(cf_matrix) * 10

    plt.figure(figsize=(12, 7))    
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g')#.get_figure()

    # plt.tight_layout()
    # plt.title("Confusion Matrix")
    # plt.xlabel('Predicted Label') #, fontsize=40)
    # plt.ylabel('Actual Label') #, fontsize=40)

    # plt.plot()
    # plt.show()

    # plt.figure(figsize=(12, 7))   
    #fpr, tpr, thresholds = roc_curve(y_true, y_prob_of_1)

    # gmeans = np.sqrt(tpr * (1-fpr))
    # # locate the index of the largest g-mean
    # ix = np.argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # # plot the roc curve for the model
    # plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    # plt.plot(fpr, tpr, marker='.', label='Logistic')
    # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # plt.xlabel('FPR') #, fontsize=40)
    # plt.ylabel('TPR') #, fontsize=40)
    # plt.legend()
    # plt.title("ROC Curve")
    # plt.show()

    auc = roc_auc_score(y_true, y_prob_of_1)
    return auc