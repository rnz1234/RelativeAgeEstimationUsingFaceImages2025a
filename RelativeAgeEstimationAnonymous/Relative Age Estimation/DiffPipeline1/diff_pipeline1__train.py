import time
import copy
import os

import torch 
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

import diff_pipeline1__config as cfg

import torch.nn.functional as F




def validate(model, val_data_loader, iter, device, criterion, dataset_size, writer, include_accuracy_metrics, is_ordinal_reg):
    print('running on validation set...')

    model.eval()

    running_loss = 0.0
    running_mae_hard = 0.0
    running_mae_soft = 0.0

    for i, batch in enumerate(val_data_loader):
        inputs = batch['image_vec'].to(device)
        labels = batch['label'].to(device).float()
        age_diff = batch['age_diff'].to(device).float()       

        with torch.no_grad():
            classification_logits, age_diff_pred_hard, age_diff_pred_soft = model(inputs)
            if is_ordinal_reg:
                preds = age_diff_pred_hard
            else:
                _, preds = torch.max(classification_logits, 1)
         
        
        loss = criterion(classification_logits, labels.long())

        running_loss += loss.item() * inputs.size(0)

        running_mae_hard += torch.nn.L1Loss()(age_diff_pred_hard, age_diff) * inputs.size(0)
        running_mae_soft += torch.nn.L1Loss()(age_diff_pred_soft, age_diff) * inputs.size(0)
        
    epoch_loss = running_loss / dataset_size
    epoch_mae_hard = running_mae_hard / dataset_size
    epoch_mae_soft = running_mae_soft / dataset_size

    writer.add_scalar('Loss/val', epoch_loss, iter)
    writer.add_scalar('Mae(Hard)/val', epoch_mae_hard, iter)
    writer.add_scalar('Mae(Soft)/val', epoch_mae_soft, iter)
        
    print('{} Loss: {:.4f} mae(hard): {:.4f}  mae(soft): {:.4f}'.format('val', epoch_loss, epoch_mae_hard, epoch_mae_soft))

    return epoch_mae_hard, epoch_mae_soft



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
    best_mae = 100.0

    running_loss = 0.0
    running_mae_hard = 0.0
    running_mae_soft = 0.0

    iter = 0

    acc_norm = 0.0

    include_accuracy_metrics = False
    is_ordinal_reg = False

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
                epoch_mae_hard = running_mae_hard / norm
                epoch_mae_soft = running_mae_soft / norm

                writer.add_scalar('Loss/{}'.format(phase), epoch_loss, iter)
                writer.add_scalar('Mae(Hard)/{}'.format(phase), epoch_mae_hard, iter)
                writer.add_scalar('Mae(Soft)/{}'.format(phase), epoch_mae_soft, iter)

                print('{} Loss: {:.4f} mae(hard): {:.4f} mae(soft): {:.4f}'.format(phase, epoch_loss, epoch_mae_hard, epoch_mae_soft))

                # print(running_mae_hard)
                # print(running_mae_soft)
                # print(norm)

                

                val_mae_hard, val_mae_soft = validate(model, data_loaders['val'], iter, device, criterion, dataset_sizes['val'], writer, include_accuracy_metrics, is_ordinal_reg)

                val_mae_taken = val_mae_soft

                # deep copy the model
                if val_mae_taken < best_mae:
                    best_mae = val_mae_taken
                    best_model_wts = copy.deepcopy(model.state_dict())

                    FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae_taken))
                    torch.save(best_model_wts, FINAL_MODEL_FILE)

                model.train()

                running_loss = 0.0
                running_mae_hard = 0.0
                running_mae_soft = 0.0

                acc_norm = 0.0

            iter += 1

            # import pdb
            # pdb.set_trace()

            inputs = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
            labels = batch['label'].to(device).float()
            age_diff = batch['age_diff'].to(device).float()     

            # import pdb
            # pdb.set_trace()

            acc_norm += inputs.size(0)           

            optimizer.zero_grad()

            with autocast():
                classification_logits, age_diff_pred_hard, age_diff_pred_soft = model(inputs)
            
                _, preds = torch.max(classification_logits, 1)
                loss = criterion(classification_logits, labels.long())
            
            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
        
            running_mae_hard += torch.nn.L1Loss()(age_diff_pred_hard, age_diff) * inputs.size(0)
            running_mae_soft += torch.nn.L1Loss()(age_diff_pred_soft.double(), age_diff) * inputs.size(0)
            
            # import pdb
            # pdb.set_trace()
            if bool(torch.isinf(running_mae_soft)) or bool(torch.isinf(running_mae_hard)):
                # print("got inf !")
                # print("ABORTING")
                # exit()
                import pdb
                pdb.set_trace()

            # scheduler.step(epoch_mae)
            scheduler.step()

        

        if validate_at_end_of_epoch:
            val_mae_hard, val_mae_soft = validate(model, data_loaders['val'], iter, device, criterion, dataset_sizes['val'], writer, include_accuracy_metrics, is_ordinal_reg)

            val_mae_taken = val_mae_soft
                
            # deep copy the model
            if val_mae_taken < best_mae:
                best_mae = val_mae_taken
                best_model_wts = copy.deepcopy(model.state_dict())

                FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae_taken))
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