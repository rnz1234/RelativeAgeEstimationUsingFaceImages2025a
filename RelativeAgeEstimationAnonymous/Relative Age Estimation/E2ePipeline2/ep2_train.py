##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline2
#	Date		:	28.10.2023
# 	Description	: 	Models file. Provides relevant pytorch-based models
#					For the task.
##############################################################################

import os
import time
import math
import copy
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast

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

"""
Evaluation method
"""
def evaluate(
	device,
	model,
	dataloader,
	dataset_size,
	criterion_age,
	criterion_age_diff,
	criterion_cls,
	criterion_mean_var,
	num_references,
	writer,
	model_path,
	epoch,
	set_type="val",
	num_classes_diff=71,
	is_ordinal=True,
	regressors_diff_head=False
):
	print(f"running on {set_type} set...")
	model.eval()

	running_loss_age = 0.0
	running_loss_diff = 0.0 #[0.0] * num_references
	running_loss_diff_cls = 0.0
	running_loss = 0.0
	running_mae_age = 0.0
	running_mae_diff = 0.0 #[0.0] * num_references
	running_mae_age_r = 0.0
	running_mae_diff_r = 0.0

	for batch in tqdm(dataloader):
		image_vec = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
		query_age = batch['query_age'].to(device).float()
		query_age_noised = batch['query_age_noised'].to(device).long()
		age_diff = batch['age_diffs_for_reg'].to(device).float() #torch.stack([batch['age_diffs_for_reg'][i].to(device).float() for i in range(num_references)])
		age_refs = batch['age_refs'].to(device).long() #torch.stack([batch['age_refs'][i].to(device).float() for i in range(num_references)])
		age_diff_cls_labels = batch['age_diffs_for_cls'].to(device).int() 
		#age_minus_diff_cls_labels = batch['age_minus_diffs_for_cls'].to(device).int() 
			

		with torch.no_grad():
			# age_pred, age_diff_preds = model(input_images=image_vec, input_ref_ages=age_refs)
			# age_loss = 	criterion_age(age_pred.reshape(age_pred.shape[0]), query_age)
			# age_diff_loss = criterion_age_diff(age_diff_preds, age_diff)
			if regressors_diff_head:
				#age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_minus, age_diff_pred_cls = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, age_diff_pred_cls = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
			else:
				#age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_minus = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
			age_loss = criterion_age(age_pred_f.reshape(age_pred_f.shape[0]), query_age)
			age_loss += criterion_age(age_pred_r.reshape(age_pred_r.shape[0]), query_age)
			age_diff_loss = criterion_age_diff(age_diff_preds_f, age_diff)
			age_diff_loss += criterion_age_diff(age_diff_preds_r, -age_diff)
			if regressors_diff_head:
				age_diff_loss += criterion_age_diff(age_diff_pred_cls, age_diff)
			



			
			if is_ordinal:
				loss_cls = 0.0
				for i in range(age_diff_cls_labels.shape[1]):
					levels = levels_from_labelbatch(age_diff_cls_labels[:, i], num_classes=num_classes_diff).to(device)
					loss_cls += condor_negloglikeloss_ext(classification_logits[:,i,:], levels, device)
				loss_cls = loss_cls / age_diff_cls_labels.shape[1]

				# loss_cls_minus = 0.0
				# for i in range(age_diff_cls_labels.shape[1]):
				# 	levels = levels_from_labelbatch(age_minus_diff_cls_labels[:, i], num_classes=num_classes_diff).to(device)
				# 	loss_cls_minus += condor_negloglikeloss_ext(classification_logits_minus[:,i,:], levels, device)
				# loss_cls_minus = loss_cls_minus / age_minus_diff_cls_labels.shape[1]

				# loss_cls += loss_cls_minus
			else:
				loss_cls = criterion_cls(classification_logits.view(classification_logits.shape[0]*classification_logits.shape[1], classification_logits.shape[2]), age_diff_cls_labels.view(age_diff_cls_labels.shape[0]*age_diff_cls_labels.shape[1]).long())
				#loss_cls += criterion_cls(classification_logits_minus, age_minus_diff_cls_labels)

			if criterion_mean_var is not None:
				loss_mean, loss_var = criterion_mean_var(classification_logits.view(classification_logits.shape[0]*classification_logits.shape[1], classification_logits.shape[2]), age_diff_cls_labels.view(age_diff_cls_labels.shape[0]*age_diff_cls_labels.shape[1]).long())

			total_loss = age_loss + age_diff_loss + loss_cls 
			if criterion_mean_var is not None:
				total_loss += loss_mean + loss_var

			running_loss += total_loss.item() * image_vec.size(0)
			running_loss_age += age_loss.item() * image_vec.size(0)
			running_loss_diff += age_diff_loss.item() * image_vec.size(0)
			running_loss_diff_cls += loss_cls.item() * image_vec.size(0)

			# for i in range(num_references):
			# 	running_loss_diff[i] += age_diff_loss[i].item() * image_vec.size(0)
			running_mae_age += torch.nn.L1Loss()(age_pred_f.reshape(age_pred_f.shape[0]), query_age) * image_vec.size(0)
			running_mae_diff += torch.nn.L1Loss()(age_diff_preds_f, age_diff) * image_vec.size(0)
			running_mae_age_r += torch.nn.L1Loss()(age_pred_r.reshape(age_pred_r.shape[0]), query_age) * image_vec.size(0)
			running_mae_diff_r += torch.nn.L1Loss()(age_diff_preds_r, -age_diff) * image_vec.size(0)
			
			# for i in range(num_references):
			# 	running_mae_diff[i] += torch.nn.L1Loss()(age_diff_preds[:,i], age_diff[i]) * image_vec.size(0)

	epoch_loss = running_loss / dataset_size
	epoch_loss_age = running_loss_age / dataset_size
	epoch_loss_diff = running_loss_diff / dataset_size #[rld / dataset_size for rld in running_loss_diff]
	epoch_age_mae = running_mae_age / dataset_size
	epoch_diff_mae = running_mae_diff / dataset_size #[rmd for rmd in running_mae_diff]
	epoch_age_mae_r = running_mae_age_r / dataset_size
	epoch_diff_mae_r = running_mae_diff_r / dataset_size #[rmd for rmd in running_mae_diff]
	epoch_loss_diff_cls = running_loss_diff_cls / dataset_size
		

	writer.add_scalar(f'TotalLoss/{set_type}', epoch_loss, epoch)
	writer.add_scalar(f'AgeLoss/{set_type}', epoch_loss_age, epoch)
	# for i in range(num_references):
	# 	writer.add_scalar(f'DiffLoss{i}/{set_type}', epoch_loss_diff[i], epoch)
	writer.add_scalar(f'DiffLoss/{set_type}', epoch_loss_diff, epoch)
	writer.add_scalar(f'AgeMaeF/{set_type}', epoch_age_mae, epoch)
	writer.add_scalar(f'AgeMaeR/{set_type}', epoch_age_mae_r, epoch)
	# for i in range(num_references):
	# 	writer.add_scalar(f'DiffMae{i}/{set_type}', epoch_diff_mae[i], epoch)
	writer.add_scalar(f'DiffMaeF/{set_type}', epoch_diff_mae, epoch)
	writer.add_scalar(f'DiffMaeR/{set_type}', epoch_diff_mae_r, epoch)
	writer.add_scalar(f'DiffClsLoss/{set_type}', epoch_loss_diff_cls, epoch)
			

	print(f'Total Loss/{set_type} : {epoch_loss} | Age MAE(F)/{set_type} : {epoch_age_mae} | Diff MAE(F)/{set_type} : {epoch_diff_mae} | Age MAE(R)/{set_type} : {epoch_age_mae_r} | Diff MAE(R)/{set_type} : {epoch_diff_mae_r}')

	return epoch_age_mae, epoch_age_mae_r, epoch_loss


	


"""
Training method
"""
def train(
	device,
	model, 
	multi_gpu,
	dataloaders,
	dataset_sizes,
	criterion_age,
	criterion_age_diff,
	criterion_cls,
	criterion_mean_var,
	optimizer,
    scheduler,
	writer,
	model_path,
	num_references,
	remove_older_checkpoints,
	save_all_model_metadata,
	num_epochs=60,
	unfreeze_feature_ext_epoch=15,
	unfreeze_feature_ext_on_rlvnt_epoch=False,
	num_classes_diff=71,
	is_ordinal=True,
	regressors_diff_head=False
):
	since = time.time()

	scaler = GradScaler()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	running_loss_age = 0.0
	running_loss_diff = 0.0 #[0.0] * num_references
	running_loss_diff_cls = 0.0
	running_loss = 0.0
	running_mae_age = 0.0
	running_mae_diff = 0.0 #[0.0] * num_references
	running_mae_age_r = 0.0
	running_mae_diff_r = 0.0

	model_file_path = None

	for epoch in range(num_epochs):
		print(f"Epoch {epoch}/{num_epochs}")
		print('-' * 10)

		running_loss_age = 0.0
		running_loss_diff = 0.0 #[0.0] * num_references
		running_loss_diff_cls = 0.0
		running_loss = 0.0
		running_mae_age = 0.0
		running_mae_diff = 0.0 #[0.0] * num_references
		running_mae_age_r = 0.0
		running_mae_diff_r = 0.0

		model.train()
		
		# unfreeze when needed 
		if unfreeze_feature_ext_on_rlvnt_epoch:
			if epoch == unfreeze_feature_ext_epoch:
				if multi_gpu:
					model.module.freeze_base_cnn(False)
				else:
					model.freeze_base_cnn(False)
		
		# training phase
		for batch in tqdm(dataloaders['train']):
			#print(f"time loop start:{time.time()}")
			image_vec = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
			query_age = batch['query_age'].to(device).float()
			query_age_noised = batch['query_age_noised'].to(device).long()
			age_diff = batch['age_diffs_for_reg'].to(device).float() #torch.stack([batch['age_diffs_for_reg'][i].to(device).float() for i in range(num_references)])
			age_refs = batch['age_refs'].to(device).long() #torch.stack([batch['age_refs'][i].to(device).float() for i in range(num_references)])
			age_diff_cls_labels = batch['age_diffs_for_cls'].to(device).int() 
			age_minus_diff_cls_labels = batch['age_minus_diffs_for_cls'].to(device).int() 
			

			optimizer.zero_grad()

			with autocast():
				#age_pred, age_diff_preds = model(input_images=image_vec, input_ref_ages=age_refs)
				#age_loss = 	criterion_age(age_pred.reshape(age_pred.shape[0]), query_age)
				#age_diff_loss = criterion_age_diff(age_diff_preds, age_diff)
				if regressors_diff_head:
					#age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_minus, age_diff_pred_cls = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
					age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, age_diff_pred_cls = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
				else:
					#age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_minus = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
					age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
				age_loss = criterion_age(age_pred_f.reshape(age_pred_f.shape[0]), query_age)
				age_loss += criterion_age(age_pred_r.reshape(age_pred_r.shape[0]), query_age)
				age_diff_loss = criterion_age_diff(age_diff_preds_f, age_diff)
				age_diff_loss += criterion_age_diff(age_diff_preds_r, -age_diff)
				if regressors_diff_head:
					age_diff_loss += criterion_age_diff(age_diff_pred_cls, age_diff)
			

				if is_ordinal:
					loss_cls = 0.0
					for i in range(age_diff_cls_labels.shape[1]):
						levels = levels_from_labelbatch(age_diff_cls_labels[:, i], num_classes=num_classes_diff).to(device)
						loss_cls += condor_negloglikeloss_ext(classification_logits[:,i,:], levels, device)
					loss_cls = loss_cls / age_diff_cls_labels.shape[1]

					# loss_cls_minus = 0.0
					# for i in range(age_diff_cls_labels.shape[1]):
					# 	levels = levels_from_labelbatch(age_minus_diff_cls_labels[:, i], num_classes=num_classes_diff).to(device)
					# 	loss_cls_minus += condor_negloglikeloss_ext(classification_logits_minus[:,i,:], levels, device)
					# loss_cls_minus = loss_cls_minus / age_minus_diff_cls_labels.shape[1]

					# loss_cls += loss_cls_minus
				else:
					loss_cls = criterion_cls(classification_logits.view(classification_logits.shape[0]*classification_logits.shape[1], classification_logits.shape[2]), age_diff_cls_labels.view(age_diff_cls_labels.shape[0]*age_diff_cls_labels.shape[1]).long())
					#loss_cls += criterion_cls(classification_logits_minus, age_minus_diff_cls_labels)

				if criterion_mean_var is not None:
					loss_mean, loss_var = criterion_mean_var(classification_logits.view(classification_logits.shape[0]*classification_logits.shape[1], classification_logits.shape[2]), age_diff_cls_labels.view(age_diff_cls_labels.shape[0]*age_diff_cls_labels.shape[1]).long())
					#loss_mean, loss_var = criterion_mean_var(classification_logits, age_diff_cls_labels)
            
			total_loss = age_loss + age_diff_loss + loss_cls 
			if criterion_mean_var is not None:
				total_loss += loss_mean + loss_var


			# total_loss.backward()
			# optimizer.step()

			

			scaler.scale(total_loss).backward()
			scaler.step(optimizer)
			scaler.update()	

			running_loss += total_loss.item() * image_vec.size(0)
			running_loss_age += age_loss.item() * image_vec.size(0)
			running_loss_diff_cls += loss_cls.item() * image_vec.size(0)
			

			running_loss_diff += age_diff_loss.item() * image_vec.size(0)
			# for i in range(num_references):
			# 	running_loss_diff[i] += age_diff_loss[i].item() * image_vec.size(0)

			running_mae_age += torch.nn.L1Loss()(age_pred_f.reshape(age_pred_f.shape[0]), query_age) * image_vec.size(0)
			running_mae_diff += torch.nn.L1Loss()(age_diff_preds_f, age_diff) * image_vec.size(0)
			running_mae_age_r += torch.nn.L1Loss()(age_pred_r.reshape(age_pred_r.shape[0]), query_age) * image_vec.size(0)
			running_mae_diff_r += torch.nn.L1Loss()(age_diff_preds_r, -age_diff) * image_vec.size(0)
			# for i in range(num_references):
			# 	running_mae_diff[i] += torch.nn.L1Loss()(age_diff_preds[:,i], age_diff[i]) * image_vec.size(0)
			# import pdb
			# pdb.set_trace()
			# if bool(torch.isinf(running_mae_age)) or bool(torch.isinf(torch.Tensor([running_loss_diff]))):
			# 	# print("got inf !")
			# 	# print("ABORTING")
			# 	# exit()
			# 	import pdb
			# 	pdb.set_trace()
			
			if math.isnan(running_mae_age):
				print(f"running_mae_age : {running_mae_age}")
				# import pdb
				# pdb.set_trace()

			scheduler.step()
			#print(f"time loop end:{time.time()}")
			
		
		# evaluation - train
		dataset_size = dataset_sizes['train']
		epoch_loss = running_loss / dataset_size
		epoch_loss_age = running_loss_age / dataset_size
		epoch_loss_diff = running_loss_diff / dataset_size #[rld / dataset_size for rld in running_loss_diff]
		epoch_age_mae = running_mae_age / dataset_size
		epoch_diff_mae = running_mae_diff / dataset_size #[rmd for rmd in running_mae_diff]
		epoch_age_mae_r = running_mae_age_r / dataset_size
		epoch_diff_mae_r = running_mae_diff_r / dataset_size #[rmd for rmd in running_mae_diff]
		epoch_loss_diff_cls = running_loss_diff_cls / dataset_size
		

		set_type = 'training'
		writer.add_scalar(f'TotalLoss/{set_type}', epoch_loss, epoch)
		writer.add_scalar(f'AgeLoss/{set_type}', epoch_loss_age, epoch)
		# for i in range(num_references):
		# 	writer.add_scalar(f'DiffLoss{i}/{set_type}', epoch_loss_diff[i], epoch)
		writer.add_scalar(f'DiffLoss/{set_type}', epoch_loss_diff, epoch)
		writer.add_scalar(f'AgeMaeF/{set_type}', epoch_age_mae, epoch)
		writer.add_scalar(f'AgeMaeR/{set_type}', epoch_age_mae_r, epoch)
		# for i in range(num_references):
		# 	writer.add_scalar(f'DiffMae{i}/{set_type}', epoch_diff_mae[i], epoch)
		writer.add_scalar(f'DiffMaeF/{set_type}', epoch_diff_mae, epoch)
		writer.add_scalar(f'DiffMaeR/{set_type}', epoch_diff_mae_r, epoch)
		writer.add_scalar(f'DiffClsLoss/{set_type}', epoch_loss_diff_cls, epoch)
			
		print(f'Total Loss/{set_type} : {epoch_loss} | Age MAE(F)/{set_type} : {epoch_age_mae} | Diff MAE(F)/{set_type} : {epoch_diff_mae} | Age MAE(R)/{set_type} : {epoch_age_mae_r} | Diff MAE(R)/{set_type} : {epoch_diff_mae_r}')


		# evaluation - validation
		val_age_mae, val_age_mae_r, val_loss = evaluate(
			device,
			model,
			dataloaders['val'],
			dataset_sizes['val'],
			criterion_age,
			criterion_age_diff,
			criterion_cls,
			criterion_mean_var,
			num_references,
			writer,
			model_path,
			epoch,
			set_type="val",
			num_classes_diff=num_classes_diff,
			is_ordinal=is_ordinal,
			regressors_diff_head=regressors_diff_head
		)

		# take checkpoint in case best so far
		if val_age_mae < best_mae:
			best_mae = val_age_mae 
			best_model_wts = copy.deepcopy(model.state_dict())
			if remove_older_checkpoints:
				if model_file_path is not None:
					print("removing previous weights")
					os.system("rm -rf {}".format(model_file_path))
			model_file_path = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(epoch, val_age_mae))
			if save_all_model_metadata:
				torch.save({
					'model_state_dict' : best_model_wts,
					'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
					'scheduler': scheduler.state_dict()
				}, model_file_path)
			else:
				torch.save(best_model_wts, model_file_path)
		
            
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model