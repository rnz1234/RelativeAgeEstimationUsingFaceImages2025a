import shutil

# importing the sys module
import sys        
 
# appending the directory of mod.py
# in the sys.path list
sys.path.append('../')   


# import pdb
# pdb.set_trace()

# from global_config_diff_infer import cfg
# MIMIC = False

# if MIMIC:
# 	import config_mimic as cfg
# else:
# 	#import config as cfg
# 	import diff_pipeline1__config as cfg

from tqdm import tqdm

import json

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import seaborn
import matplotlib.pyplot as plt
from condor_pytorch.metrics import mean_absolute_error
from condor_pytorch.dataset import levels_from_labelbatch

from scipy.stats import pearsonr

# To be run with the AgePredictBasicPipeline pipeline. 
# If run with bypass_diff=True - evaluates the age predict with diff fix
# If run with bypass_diff=False - evaluates the age predict without diff fix
def evaluate_pipeline(pipeline, val_data_loader, device, dataset_size, bypass_diff, compare_post_round):
	print('running on validation set...')

	c_in_radius = 0 
	c_out_of_radius = 0

	running_mae = 0.0
	running_mae_out_of_radius = 0.0
	dataset_eff_size = dataset_size
	cur_dataset_size = 0
	running_mae_tot = 0.0

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()
		indep_image = batch['indep_image'].to(device)

		with torch.no_grad():
			if bypass_diff:
				valid, age_pred = pipeline.predict(inputs, indep_image, bypass_diff=bypass_diff, ref_age=ages, compare_post_round=compare_post_round)
			# import pdb
			# pdb.set_trace
			else:
				#valid, age_pred = pipeline.predict(inputs, indep_image, bypass_diff=bypass_diff, ref_age=ages, compare_post_round=compare_post_round)
				valid, age_pred = pipeline.predict_stochastic(inputs, indep_image, ref_age=ages, compare_post_round=compare_post_round)

		if valid == -1:
			print("invalid query, not taking in account")# - aborting")
			#exit()
		elif valid == -2:
			c_out_of_radius += 1
			#print("query not relevant, ignoring")
			dataset_eff_size -= 1
			running_mae_out_of_radius += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			running_mae_tot += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			cur_dataset_size += 1
		else:
			# import pdb
			# pdb.set_trace()
			if inputs.size(0) != 1:
				print("error - batch is not with size=1")
				exit()
			c_in_radius += 1
			running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			running_mae_tot += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			cur_dataset_size +=1
		#print(i)
		if i % 100 == 50:
			cur_mae = running_mae / c_in_radius
			cur_mae_out = running_mae_out_of_radius / c_out_of_radius
			cur_mae_tot = running_mae_tot / cur_dataset_size
			print("- in radius: ", c_in_radius)
			print("- in radius mae sum: ", running_mae.cpu())
			print("- out of radius: ", c_out_of_radius)
			print("- out of radius mae sum: ", running_mae_out_of_radius.cpu())
			print("- total: ", cur_dataset_size)
			print("- total mae sum : ", running_mae_tot.cpu())
			print("progress : {:.2f}%".format(((i + 1) / dataset_size) * 100))
			print("current MAE (in radius): {:.4f}".format(cur_mae))
			print("current MAE (out of radius): {:.4f}".format(cur_mae_out))
			print("current MAE (total): {:.4f}".format(cur_mae_tot))

	total_mae_in = running_mae / dataset_eff_size
	total_mae_out = running_mae_out_of_radius / c_out_of_radius
	total_mae_tot = running_mae_tot / dataset_size

	print('Total MAE (in): {:.4f}'.format(total_mae_in))
	print('Total MAE (out): {:.4f}'.format(total_mae_out))
	print('Total MAE (total): {:.4f}'.format(total_mae_tot))

	return total_mae_tot



def evaluate_pipeline_clean(pipeline, val_data_loader, device, dataset_size, bypass_diff, compare_post_round):
	print('running on validation set...')

	c_in_radius = 0 
	c_out_of_radius = 0

	running_mae = 0.0
	running_mae_out_of_radius = 0.0
	dataset_eff_size = dataset_size
	cur_dataset_size = 0
	running_mae_tot = 0.0

	pred_diff2actual_diffs = dict()

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()
		indep_image = batch['indep_image'].to(device)

		with torch.no_grad():
			if bypass_diff:
				age_pred = pipeline.predict(inputs, indep_image, bypass_diff=bypass_diff) #, ref_age=ages, compare_post_round=compare_post_round)
			# import pdb
			# pdb.set_trace
			else:
				#age_ref, diff = pipeline.predict(inputs, indep_image)
				
				# if float(diff.to('cpu')) not in pred_diff2actual_diffs.keys():
				# 	pred_diff2actual_diffs[float(diff.to('cpu'))] = []
				
				# pred_diff2actual_diffs[float(diff.to('cpu'))].append(float((ages-age_ref).to('cpu')))
				# import pdb
				# pdb.set_trace()
				age_pred = pipeline.predict(inputs, indep_image)
				#valid, age_pred = pipeline.predict(inputs, indep_image, bypass_diff=bypass_diff, ref_age=ages, compare_post_round=compare_post_round)
				#valid, age_pred = pipeline.predict_stochastic(inputs, indep_image, ref_age=ages, compare_post_round=compare_post_round)

		
		if inputs.size(0) != 1:
			print("error - batch is not with size=1")
			exit()

		running_mae_tot += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
		#running_mae_tot += torch.nn.L1Loss()(diff, ages-age_ref) * inputs.size(0)
		cur_dataset_size +=1

		#print(i)
		if i % 100 == 50:
			# if i % 1000 == 50:
			# 	import pdb
			# 	pdb.set_trace()

			cur_mae_tot = running_mae_tot / cur_dataset_size
			print("- total: ", cur_dataset_size)
			print("- total mae sum : ", running_mae_tot.cpu())
			print("progress : {:.2f}%".format(((i + 1) / dataset_size) * 100))
			print("current MAE (total): {:.4f}".format(cur_mae_tot))

	total_mae_tot = running_mae_tot / dataset_size

	print('Total MAE (total): {:.4f}'.format(total_mae_tot))

	# import pdb
	# pdb.set_trace()

	return total_mae_tot

def evaluate_pipeline_clean_enhanced(pipeline, 
										val_data_loader, 
										device, 
										dataset_size, 
										bypass_diff, 
										compare_post_round, 
										range_lo, 
										range_hi, 
										im2age_map_test=None, 
										check_im2age_map_test=False, 
										create_im2age_map_test=False,
										check_against_pred_db=False,
										apply_diff_model_only_in_range=False):
	print('running on validation set...')

	c_in_radius = 0 
	c_out_of_radius = 0

	running_mae_in_radius = 0.0
	running_mae_out_of_radius = 0.0
	dataset_eff_size = dataset_size
	cur_dataset_size = 0
	running_mae_tot = 0.0
	running_mae_diff_in_radius = 0.0
	running_mae_diff_out_of_radius = 0.0

	pred_diff2actual_diffs = dict()
	pred_dev = []
	im2age_map = dict()	
	

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()
		indep_image = batch['indep_image'].to(device)
		idxs = batch['idx'].to(device).float()

		# import pdb
		# pdb.set_trace()	

		
		
		

		with torch.no_grad():
			if bypass_diff:
				age_pred, in_range = pipeline.predict(query=inputs, 
														idx=idxs[0],
														query_indep=indep_image, 
														real_age=ages, 
														range_in=range_lo,
														range_out=range_hi, 
														bypass_diff=True,
														check_ranges=True,
														pred_db_val=im2age_map_test[str(i)], 
														check_against_pred_db=check_against_pred_db) #, ref_age=ages, compare_post_round=compare_post_round)
				if check_im2age_map_test:
					if im2age_map_test[str(i)] != float(age_pred.cpu()): #float(age_pred.to('cpu')):
						#print("unequal: ", i, im2age_map_test[str(i)], age_pred)
						pred_dev.append(im2age_map_test[str(i)]-float(age_pred.cpu()))
				if create_im2age_map_test:
					if len(idxs) != 1:
						print("error - more than 1 index")
					im2age_map[int(idxs[0].cpu())] = float(age_pred.cpu())

			# import pdb
			# pdb.set_trace
			else:
				#age_ref, diff = pipeline.predict(inputs, indep_image)
				
				# if float(diff.to('cpu')) not in pred_diff2actual_diffs.keys():
				# 	pred_diff2actual_diffs[float(diff.to('cpu'))] = []
				
				# pred_diff2actual_diffs[float(diff.to('cpu'))].append(float((ages-age_ref).to('cpu')))
				# import pdb
				# pdb.set_trace()
				age_pred, in_range, diff_pred, diff_real = pipeline.predict(query=inputs, 
														idx=idxs[0],
														query_indep=indep_image, 
														real_age=ages, 
														range_in=range_lo,
														range_out=range_hi, 
														bypass_diff=False,
														check_ranges=True,
														pred_db_val=im2age_map_test[str(i)], 
														check_against_pred_db=check_against_pred_db, 
														apply_diff_model_only_in_range=apply_diff_model_only_in_range)

				#valid, age_pred = pipeline.predict(inputs, indep_image, bypass_diff=bypass_diff, ref_age=ages, compare_post_round=compare_post_round)
				#valid, age_pred = pipeline.predict_stochastic(inputs, indep_image, ref_age=ages, compare_post_round=compare_post_round)

		
		if inputs.size(0) != 1:
			print("error - batch is not with size=1")
			exit()
		
		if in_range:
			c_in_radius += 1
			running_mae_in_radius += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			if not bypass_diff:
				running_mae_diff_in_radius += torch.nn.L1Loss()(diff_pred, diff_real) * inputs.size(0)
		else:
			c_out_of_radius += 1
			running_mae_out_of_radius += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			if not bypass_diff:
				running_mae_diff_out_of_radius += torch.nn.L1Loss()(diff_pred, diff_real) * inputs.size(0)
		
		running_mae_tot += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
		#running_mae_tot += torch.nn.L1Loss()(diff, ages-age_ref) * inputs.size(0)
		cur_dataset_size +=1

		#print(i)
		if i % 100 == 50:
			# if i % 1000 == 50:
			# 	import pdb
			# 	pdb.set_trace()

			cur_mae_out_of_radius = running_mae_out_of_radius / c_out_of_radius
			cur_mae_in_radius = running_mae_in_radius / c_in_radius
			cur_mae_tot = running_mae_tot / cur_dataset_size
			if not bypass_diff:
				cur_mae_diff_out_of_radius = running_mae_diff_out_of_radius / c_out_of_radius
				cur_mae_diff_in_radius = running_mae_diff_in_radius / c_in_radius
			print("- IN RANGE: ", c_in_radius)
			print("- IN RANGE mae sum : ", running_mae_in_radius.cpu())
			print("- OUT OF RANGE: ", c_out_of_radius)
			print("- OUT OF RANGE mae sum : ", running_mae_out_of_radius.cpu())
			print("- total: ", cur_dataset_size)
			print("- total mae sum : ", running_mae_tot.cpu())
			print("progress : {:.2f}%".format(((i + 1) / dataset_size) * 100))
			print("current MAE (in range): {:.4f}".format(cur_mae_in_radius))
			print("current MAE (out of range): {:.4f}".format(cur_mae_out_of_radius))
			print("current MAE (total): {:.4f}".format(cur_mae_tot))
			if not bypass_diff:
				print("current MAE diff (in range): {:.4f}".format(cur_mae_diff_in_radius))
				print("current MAE diff (out of range): {:.4f}".format(cur_mae_diff_out_of_radius))

	total_mae_inr = running_mae_in_radius / c_in_radius
	total_mae_outr = running_mae_out_of_radius / c_out_of_radius
	if not bypass_diff:
		total_mae_diff_inr = running_mae_diff_in_radius / c_in_radius
		total_mae_diff_outr = running_mae_diff_out_of_radius / c_out_of_radius
	total_mae_tot = running_mae_tot / dataset_size

	print('In Range MAE (total): {:.4f}'.format(total_mae_inr))
	print('Out Of Range MAE (total): {:.4f}'.format(total_mae_outr))
	if not bypass_diff:
		print('In Range diff MAE (total): {:.4f}'.format(total_mae_diff_inr))
		print('Out Of Range diff MAE (total): {:.4f}'.format(total_mae_diff_outr))
	print('Total MAE (total): {:.4f}'.format(total_mae_tot))
	pipeline.print_moved_refs()

	if create_im2age_map_test:
		im2age_map_js = json.dumps(im2age_map)

		with open("im2age_map_test_from_actual_pipe.json", 'w') as fmap:
			fmap.write(im2age_map_js)

	if check_im2age_map_test:
		total_pred_dev_me = np.mean(np.array(pred_dev))
		total_pred_dev_mae = np.mean(np.abs(np.array(pred_dev)))
		print("pred dev me: {}".format(total_pred_dev_me))
		print("pred dev mae: {}".format(total_pred_dev_mae))
		print("you can review pred_dev")
		import pdb
		pdb.set_trace()
		


	# import pdb
	# pdb.set_trace()

	return total_mae_tot


def evaluate_pipeline_clean_compare_enhanced(pipeline, 
										val_data_loader, 
										device, 
										dataset_size, 
										bypass_diff, 
										compare_post_round, 
										range_lo, 
										range_hi, 
										im2age_map_test=None, 
										check_im2age_map_test=False, 
										create_im2age_map_test=False,
										check_against_pred_db=False):
	print('running on validation set...')

	c_in_radius = 0 
	c_out_of_radius = 0

	running_mae_in_radius = 0.0
	running_mae_out_of_radius = 0.0 #torch.Tensor([0.0]).to(device)
	dataset_eff_size = dataset_size
	cur_dataset_size = 0
	running_mae_tot = 0.0
	running_mae_diff_in_radius = 0.0
	running_mae_diff_out_of_radius = 0.0 #torch.Tensor([0.0]).to(device)

	cur_mae_out_of_radius = 0.0
	cur_mae_diff_out_of_radius = 0.0

	pred_diff2actual_diffs = dict()
	pred_dev = []
	im2age_map = dict()	
	

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()
		indep_image = batch['indep_image'].to(device)
		idxs = batch['idx'].to(device).float()

		# import pdb
		# pdb.set_trace()	

		
		
		

		with torch.no_grad():
			if bypass_diff:
				age_pred, in_range = pipeline.predict(query=inputs, 
														idx=idxs[0],
														query_indep=indep_image, 
														real_age=ages, 
														range_in=range_lo,
														range_out=range_hi, 
														bypass_diff=True,
														check_ranges=True,
														pred_db_val=im2age_map_test[str(i)], 
														check_against_pred_db=check_against_pred_db) #, ref_age=ages, compare_post_round=compare_post_round)
				if check_im2age_map_test:
					if im2age_map_test[str(i)] != float(age_pred.cpu()): #float(age_pred.to('cpu')):
						#print("unequal: ", i, im2age_map_test[str(i)], age_pred)
						pred_dev.append(im2age_map_test[str(i)]-float(age_pred.cpu()))
				if create_im2age_map_test:
					if len(idxs) != 1:
						print("error - more than 1 index")
					im2age_map[int(idxs[0].cpu())] = float(age_pred.cpu())

			# import pdb
			# pdb.set_trace
			else:
				#age_ref, diff = pipeline.predict(inputs, indep_image)
				
				# if float(diff.to('cpu')) not in pred_diff2actual_diffs.keys():
				# 	pred_diff2actual_diffs[float(diff.to('cpu'))] = []
				
				# pred_diff2actual_diffs[float(diff.to('cpu'))].append(float((ages-age_ref).to('cpu')))
				# import pdb
				# pdb.set_trace()
				age_pred, in_range, diff_pred, diff_real = pipeline.predict(query=inputs, 
														idx=idxs[0],
														query_indep=indep_image, 
														real_age=ages, 
														range_in=range_lo,
														range_out=range_hi, 
														bypass_diff=False,
														check_ranges=True,
														pred_db_val=im2age_map_test[str(i)], 
														check_against_pred_db=check_against_pred_db)

				#print(age_pred, ages, in_range, diff_pred, diff_real)

				#valid, age_pred = pipeline.predict(inputs, indep_image, bypass_diff=bypass_diff, ref_age=ages, compare_post_round=compare_post_round)
				#valid, age_pred = pipeline.predict_stochastic(inputs, indep_image, ref_age=ages, compare_post_round=compare_post_round)

		
		if inputs.size(0) != 1:
			print("error - batch is not with size=1")
			exit()
		
		if in_range:
			c_in_radius += 1
			running_mae_in_radius += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			if not bypass_diff:
				running_mae_diff_in_radius += torch.nn.L1Loss()(diff_pred, diff_real) * inputs.size(0)
		else:
			c_out_of_radius += 1
			running_mae_out_of_radius += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			if not bypass_diff:
				running_mae_diff_out_of_radius += torch.nn.L1Loss()(diff_pred, diff_real) * inputs.size(0)
		
		running_mae_tot += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
		#running_mae_tot += torch.nn.L1Loss()(diff, ages-age_ref) * inputs.size(0)
		cur_dataset_size +=1

		#print(i)
		if i % 100 == 50:
			# if i % 1000 == 50:
			# import pdb
			# pdb.set_trace()
			if c_out_of_radius != 0:
				cur_mae_out_of_radius = running_mae_out_of_radius / c_out_of_radius

			if c_in_radius != 0:
				cur_mae_in_radius = running_mae_in_radius / c_in_radius
			
			cur_mae_tot = running_mae_tot / cur_dataset_size
			if not bypass_diff:
				if c_out_of_radius != 0:
					cur_mae_diff_out_of_radius = running_mae_diff_out_of_radius / c_out_of_radius
				if c_in_radius != 0:
					cur_mae_diff_in_radius = running_mae_diff_in_radius / c_in_radius
			print("- IN RANGE: ", c_in_radius)
			print("- IN RANGE mae sum : ", running_mae_in_radius.cpu())
			print("- OUT OF RANGE: ", c_out_of_radius)
			print("- OUT OF RANGE mae sum : ", running_mae_out_of_radius.cpu())
			print("- total: ", cur_dataset_size)
			print("- total mae sum : ", running_mae_tot.cpu())
			print("progress : {:.2f}%".format(((i + 1) / dataset_size) * 100))
			print("current MAE (in range): {:.4f}".format(cur_mae_in_radius))
			print("current MAE (out of range): {:.4f}".format(cur_mae_out_of_radius))
			print("current MAE (total): {:.4f}".format(cur_mae_tot))
			if not bypass_diff:
				print("current MAE diff (in range): {:.4f}".format(cur_mae_diff_in_radius))
				print("current MAE diff (out of range): {:.4f}".format(cur_mae_diff_out_of_radius))

	if c_in_radius != 0:
		total_mae_inr = running_mae_in_radius / c_in_radius
	if c_out_of_radius != 0:
		total_mae_outr = running_mae_out_of_radius / c_out_of_radius
	if not bypass_diff:
		if c_in_radius != 0:
			total_mae_diff_inr = running_mae_diff_in_radius / c_in_radius
		if c_out_of_radius != 0:
			total_mae_diff_outr = running_mae_diff_out_of_radius / c_out_of_radius
	total_mae_tot = running_mae_tot / dataset_size

	print('In Range MAE (total): {:.4f}'.format(total_mae_inr))
	print('Out Of Range MAE (total): {:.4f}'.format(total_mae_outr))
	if not bypass_diff:
		print('In Range diff MAE (total): {:.4f}'.format(total_mae_diff_inr))
		print('Out Of Range diff MAE (total): {:.4f}'.format(total_mae_diff_outr))
	print('Total MAE (total): {:.4f}'.format(total_mae_tot))

	if create_im2age_map_test:
		im2age_map_js = json.dumps(im2age_map)

		with open("im2age_map_test_from_actual_pipe.json", 'w') as fmap:
			fmap.write(im2age_map_js)

	if check_im2age_map_test:
		total_pred_dev_me = np.mean(np.array(pred_dev))
		total_pred_dev_mae = np.mean(np.abs(np.array(pred_dev)))
		print("pred dev me: {}".format(total_pred_dev_me))
		print("pred dev mae: {}".format(total_pred_dev_mae))
		print("you can review pred_dev")
		import pdb
		pdb.set_trace()
		


	# import pdb
	# pdb.set_trace()

	return total_mae_tot




def evaluate_pipeline_diff(pipeline, val_data_loader, device, dataset_size, compare_post_round, add_graph):
	print('running on validation set...')

	c_in_radius = 0 
	c_out_of_radius = 0

	running_mae = 0.0
	running_mae_out_of_radius = 0.0
	dataset_eff_size = dataset_size
	cur_dataset_size = 0
	running_mae_tot = 0.0

	real_diff_arr = []
	pred_diff_arr = []

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()
		indep_image = batch['indep_image'].to(device)

		with torch.no_grad():
			valid, diff_pred, diff_truth = pipeline.predict_diff(inputs, indep_image, ref_age=ages, compare_post_round=compare_post_round)
			#valid, age_pred = pipeline.predict_stochastic(inputs, indep_image, ref_age=ages, compare_post_round=compare_post_round)

		if valid == -1:
			print("invalid query, not taking in account")# - aborting")
			#exit()
		elif valid == -2:
			c_out_of_radius += 1
			#print("query not relevant, ignoring")
			#dataset_eff_size -= 1
			running_mae_out_of_radius += torch.nn.L1Loss()(diff_pred, diff_truth) * inputs.size(0)
			running_mae_tot += torch.nn.L1Loss()(diff_pred, diff_truth) * inputs.size(0)
			cur_dataset_size += 1
			if add_graph:
				real_diff_arr.append(float(diff_truth.cpu()))
				pred_diff_arr.append(float(diff_pred.cpu()))
		else:
			# import pdb
			# pdb.set_trace()
			if inputs.size(0) != 1:
				print("error - batch is not with size=1")
				exit()
			c_in_radius += 1
			running_mae += torch.nn.L1Loss()(diff_pred, diff_truth) * inputs.size(0)
			running_mae_tot += torch.nn.L1Loss()(diff_pred, diff_truth) * inputs.size(0)
			cur_dataset_size +=1
			if add_graph:
				real_diff_arr.append(float(diff_truth.cpu()))
				pred_diff_arr.append(float(diff_pred.cpu()))
		#print(i)
		if i % 100 == 50:
			cur_mae = running_mae / c_in_radius
			cur_mae_out = running_mae_out_of_radius / c_out_of_radius
			cur_mae_tot = running_mae_tot / cur_dataset_size
			print("- in radius: ", c_in_radius)
			print("- in radius mae sum: ", running_mae.cpu())
			print("- out of radius: ", c_out_of_radius)
			print("- out of radius mae sum: ", running_mae_out_of_radius.cpu())
			print("- total: ", cur_dataset_size)
			print("- total mae sum : ", running_mae_tot.cpu())
			print("progress : {:.2f}%".format(((i + 1) / dataset_size) * 100))
			print("current MAE (in radius): {:.4f}".format(cur_mae))
			print("current MAE (out of radius): {:.4f}".format(cur_mae_out))
			print("current MAE (total): {:.4f}".format(cur_mae_tot))

	total_mae_in = running_mae / c_in_radius #dataset_eff_size
	total_mae_out = running_mae_out_of_radius / c_out_of_radius
	total_mae_tot = running_mae_tot / dataset_size

	print('Total MAE (in): {:.4f}'.format(total_mae_in))
	print('Total MAE (out): {:.4f}'.format(total_mae_out))
	print('Total MAE (total): {:.4f}'.format(total_mae_tot))
	print("Pearson Corr (read diff <-> pred diff): " + str(pearsonr(real_diff_arr, pred_diff_arr)))


	plt.plot(real_diff_arr, pred_diff_arr, '.')

	
	plt.show()

	return total_mae_tot




# To be used with DiffPredictCheckPipeline. This feeds noised age : age + dev, where dev ~ N(0,STDDEV)
# and outputs:
# - MAE for age estimation done using the pipeline 
# - MAdev : the actual mean absolute dev of the noise signal
# - Theoretical MAdev : the theoretical mean absolute dev of the noise signal
def check_pipeline(pipeline, val_data_loader, device, dataset_size, cfg):
	print('running on validation set...')

	running_mae = 0.0
	running_mae_out_of_radius = 0.0
	cur_dataset_size = 0
	running_mae_tot = 0.0
	running_madev = 0.0
	ignored = 0

	

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()

		valid = -1
		while valid == -1:
			dev = np.random.normal(loc=0, scale=cfg.STDDEV)

			with torch.no_grad():
				valid, age_pred = pipeline.predict(inputs, dev=dev, ref_age=ages)
			
			if valid == -1:
				print("sample age not found, trying again...")

		if valid == -1:
			print("invalid query, not taking in account") # - aborting")
			ignored += 1
			#exit()
		else:
			# import pdb
			# pdb.set_trace()
			if inputs.size(0) != 1:
				print("error - batch is not with size=1")
				exit()
			running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)
			running_madev += abs(dev)
			cur_dataset_size +=1

		if i % 100 == 50:
			cur_mae = running_mae / cur_dataset_size
			cur_madev = running_madev / cur_dataset_size
			print("progress : {:.2f}%".format(((i + 1) / dataset_size) * 100))
			print("current MAE : {:.4f}".format(cur_mae))
			print("current MAdev : {:.4f}".format(cur_madev))
		

	total_mae_tot = running_mae / cur_dataset_size
	total_madev = running_madev / cur_dataset_size
	total_ignored = (ignored / cur_dataset_size) * 100

	print('Total MAE : {:.4f}'.format(total_mae_tot))
	print('Total MAdev : {:.4f}'.format(total_madev))
	print('Theoretical MAdev : {:.4f}'.format(STDDEV * np.sqrt(2 / np.pi)))
	print('Total ignored : {:.4f}%'.format(total_ignored))

	return total_mae_tot



00


def diff_pipeline_confusion_matrix_analysis(pipeline, val_data_loader, device, dataset_size, cfg, is_ordinal_reg=False, is_pipeline=True, age_ref_input=False):
	#print('running on dataset...')

	running_mae_hard = 0.0
	running_mae_hard_condor = 0.0
	running_mae_soft = 0.0

	pred_full = np.array([])
	labels_cls = np.array([])

	limit = -1
	with torch.no_grad():
		for i, batch in enumerate(tqdm(val_data_loader)): #tqdm(val_data_loader)
			inputs = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
			labels = batch['label'].to(device).float()
			age_diff = batch['age_diff'].to(device).float()  
			age_ref = batch['age_ref'].to(device).long()  
			#print(i)
			#
			with_hard_pred = False
			if is_pipeline:
				if age_ref_input:
					model_run_res = pipeline.predict_diff_only(inputs=inputs, age_ref=age_ref)
				else:
					model_run_res = pipeline.predict_diff_only(inputs)
				if len(model_run_res) == 2:
					classification_logits, age_diff_pred_soft = model_run_res
				else:
					with_hard_pred = True
					classification_logits, age_diff_pred_hard, age_diff_pred_soft = model_run_res
			else:
				with_hard_pred = True
				# model
				classification_logits, age_diff_pred_hard, age_diff_pred_soft = pipeline(inputs)

			if with_hard_pred:
				if is_ordinal_reg: #cfg.IS_ORDINAL_REG:
					pred = age_diff_pred_hard.detach().to('cpu').numpy() - cfg.AGE_DIFF_LEARN_RADIUS_HI
				else:
					pred = age_diff_pred_hard.detach().to('cpu').numpy()
			else:
				pred = np.round(age_diff_pred_soft.detach().to('cpu').numpy())

			labels_cls = np.concatenate((labels_cls, age_diff.to('cpu').numpy()), axis=0)
			
			pred_full = np.concatenate((pred_full, pred), axis=0)
			
			if is_ordinal_reg: #cfg.IS_ORDINAL_REG:
				if with_hard_pred:
					levels = levels_from_labelbatch(labels.int(), num_classes=2*cfg.AGE_DIFF_LEARN_RADIUS_HI+1)
					levels = levels.to(device)
					running_mae_hard_condor += mean_absolute_error(classification_logits.double(), levels) * inputs.size(0)
					running_mae_hard += torch.nn.L1Loss()(age_diff_pred_hard.double() - cfg.AGE_DIFF_LEARN_RADIUS_HI, age_diff) * inputs.size(0)
			else:
				if with_hard_pred:
					running_mae_hard += torch.nn.L1Loss()(age_diff_pred_hard, age_diff) * inputs.size(0)
			
			running_mae_soft += torch.nn.L1Loss()(age_diff_pred_soft, age_diff) * inputs.size(0)


			if limit != -1:
				if i == limit-1:
					break
		#torch.cuda.empty_cache()

	# import pdb
	# pdb.set_trace()

	if with_hard_pred:
		total_mae_hard = running_mae_hard / dataset_size
	if is_ordinal_reg: #cfg.IS_ORDINAL_REG:
		if with_hard_pred:
			total_mae_hard_condor = running_mae_hard_condor / dataset_size
	total_mae_soft = running_mae_soft / dataset_size

	print('Total MAE (soft)): {:.4f}'.format(total_mae_soft))
	if with_hard_pred:
		print('Total MAE (hard): {:.4f}'.format(total_mae_hard))
	if is_ordinal_reg: #cfg.IS_ORDINAL_REG:
		if with_hard_pred:
			print('Total MAE (hard condor): {:.4f}'.format(total_mae_hard_condor))

	# import pdb
	# pdb.set_trace()



	fig, ax = plt.subplots(figsize=(20, 20)) #(figsize=(20, 20)) # plt.subplots(figsize=(100, 100))

	class_names = [str(i) for i in range(-cfg.AGE_DIFF_LEARN_RADIUS_HI, cfg.AGE_DIFF_LEARN_RADIUS_HI + 1)]
	cnf_matrix = metrics.confusion_matrix(labels_cls, pred_full, labels=range(-cfg.AGE_DIFF_LEARN_RADIUS_HI, cfg.AGE_DIFF_LEARN_RADIUS_HI + 1))
	
	seaborn.heatmap(pd.DataFrame(data=cnf_matrix, index=class_names, columns=class_names), annot=True, cmap="YlGnBu", fmt='g') #, annot_kws={"size": 40})

	ax.xaxis.set_label_position('top')
	plt.tight_layout()
	plt.xlabel('Predicted Label') #, fontsize=40)
	plt.ylabel('Actual Label') #, fontsize=40)

	#plt.rcParams.update({'font.size': 40})
	ax.tick_params(axis ='both', which ='major', 
               #labelsize = 40, #, pad = 12, 
               colors ='r')

	plt.plot()
	plt.show()

	# total_mae_hard == hard_final from *train.py file
	# total_mae_hard_condor == hard from *train.py file
	if with_hard_pred:
		return total_mae_hard_condor, total_mae_hard, total_mae_soft
	else:
		return total_mae_soft


def diff_range_pipeline_confusion_matrix_analysis(model, val_data_loader, device, dataset_size):
	#print('running on validation set...')

	pred_full = np.array([])
	labels_cls = np.array([])
	prob_full = prob_full = np.empty((0,3)) #np.array([])

	limit = -1
	for i, batch in enumerate(tqdm(val_data_loader)):
		inputs = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
		labels = batch['label'].to(device).float()
		age_diff = batch['age_diff'].to(device).float()  

		#with torch.no_grad():
		classification_logits, age_diff_pred_hard, pred_probs = model(inputs)
		
		pred = age_diff_pred_hard.detach().to('cpu').numpy()
		prob = pred_probs.detach().to('cpu').numpy()

		labels_cls = np.concatenate((labels_cls, labels.to('cpu').numpy()), axis=0)

		pred_full = np.concatenate((pred_full, pred), axis=0)

		# import pdb
		# pdb.set_trace()
		prob_full = np.concatenate((prob_full, prob) ,axis=0 )



		if limit != -1:
			if i == limit-1:
				break



	# import pdb
	# pdb.set_trace()

	fig, ax = plt.subplots()


	class_names = [0,1,2] #range(-cfg.AGE_RADIUS, cfg.AGE_RADIUS + 1)]
	cnf_matrix = metrics.confusion_matrix(labels_cls, pred_full, labels=[0,1,2])
	seaborn.heatmap(pd.DataFrame(data=cnf_matrix, index=class_names, columns=class_names), annot=True, cmap="YlGnBu", fmt='g')

	ax.xaxis.set_label_position('top')
	plt.tight_layout()
	plt.xlabel('Predicted Label')
	plt.ylabel('Actual Label')

	print(metrics.classification_report(labels_cls, pred_full, digits=3))
	#labels_cls       = labels_cls.view(-1,1).long().to('cpu').numpy()
	#reshaped_labels_cls = labels_cls.reshape((labels_cls.shape[0],))
	# import pdb
	# pdb.set_trace()
	print("AUC (OVO): ", roc_auc_score(labels_cls, prob_full, multi_class='ovo', labels=[0., 1., 2.]))
	print("AUC (OVR): ", roc_auc_score(labels_cls, prob_full, multi_class='ovr', labels=[0., 1., 2.]))

	plt.plot()
	plt.show()


# TODO : complete
def mimic_diff_pipeline_confusion_matrix_analysis(pipeline, val_data_loader, device, dataset_size):
    pass


# To be run with the AgePredictMimicPipeline pipeline
def evaluate_mimic_pipeline_simple(pipeline, val_data_loader, device, dataset_size, compare_post_round):
	running_mae_soft = 0.0
	running_mae_hard = 0.0
	running_mae_hard_ord = 0.0
	running_mae_noisy_age = 0.0
	cur_dataset_size = 0

	pred_soft_arr = []
	pred_hard_arr = []
	pred_hard_ord_arr = []
	orig_arr = []
	noisy_age_arr = []

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()
		indep_image = batch['indep_image'].to(device)

		with torch.no_grad():
			valid, age_pred_soft, age_pred_hard, age_hard_ord_pred, noisy_age = pipeline.predict_stochastic(inputs, indep_image, compare_post_round=compare_post_round, orig_age=float(ages.cpu()))

		if valid == -1:
			print("invalid query, not taking in account")# - aborting")
			exit()
		else:
			# import pdb
			# pdb.set_trace()
			if inputs.size(0) != 1:
				print("error - batch is not with size=1")
				exit()

			# import pdb
			# pdb.set_trace()

			running_mae_soft += torch.nn.L1Loss()(age_pred_soft, ages) * inputs.size(0)
			running_mae_hard += torch.nn.L1Loss()(age_pred_hard, ages) * inputs.size(0)
			#running_mae_hard_ord += torch.nn.L1Loss()(age_hard_ord_pred, ages) * inputs.size(0)
			running_mae_noisy_age += torch.nn.L1Loss()(torch.Tensor([noisy_age]).to(device), ages) * inputs.size(0)
			cur_dataset_size +=1


			pred_soft_arr.append(np.float(age_pred_soft.cpu()))
			pred_hard_arr.append(np.float(age_pred_hard.cpu()))
			#pred_hard_ord_arr.append(np.float(age_hard_ord_pred.cpu()))
			orig_arr.append(np.float(ages.cpu()))
			noisy_age_arr.append(noisy_age)

		#print(i)
		if i % 100 == 50:
			cur_mae_soft = running_mae_soft / cur_dataset_size
			cur_mae_hard = running_mae_hard / cur_dataset_size
			cur_mae_hard_ord = running_mae_hard_ord / cur_dataset_size
			cur_mae_noisy = running_mae_noisy_age / cur_dataset_size
			print("current items observed: " + str(cur_dataset_size))
			print("- soft mae sum: ", running_mae_soft.cpu())
			print("- hard mae sum: ", running_mae_hard.cpu())
			#print("- hard ord mae sum: ", running_mae_hard_ord.cpu())
			print("- noise mae sum: ", running_mae_noisy_age.cpu())
			print("progress : {:.2f}%".format(((i + 1) / dataset_size) * 100))
			print("current MAE (soft): {:.4f}".format(cur_mae_soft))
			print("current MAE (hard): {:.4f}".format(cur_mae_hard))
			#print("current MAE (hard ord): {:.4f}".format(cur_mae_hard_ord))
			print("current MAE (noisy): {:.4f}".format(cur_mae_noisy))

	total_mae_soft = running_mae_soft / dataset_size
	total_mae_hard = running_mae_hard / dataset_size
	#total_mae_hard_ord = running_mae_hard_ord / dataset_size
	total_mae_noise = running_mae_noisy_age / dataset_size
	

	print('Total MAE (soft): {:.4f}'.format(total_mae_soft))
	print('Total MAE (hard): {:.4f}'.format(total_mae_hard))
	print('Total MAE (noise): {:.4f}'.format(total_mae_noise))
	#print('Total MAE (hard ord): {:.4f}'.format(total_mae_hard_ord))

	import pdb
	pdb.set_trace()


def diff_dataset_stats(diff_dataloader, dataset_name, device):
	diff_vec = np.array([])
	for batch in tqdm(diff_dataloader):
		diff_vec = np.concatenate((diff_vec, batch['age_diff'].cpu().numpy()))

	print("----------------------------------------------")
	print("-- Diff Dataset ({})- Stats".format(dataset_name))
	print("--")
	print("- ME (mean error)      : {}".format(np.mean(diff_vec)))
	print("- STDev (standard dev) : {}".format(np.std(diff_vec)))

	plt.hist(np.array(diff_vec), bins=20)
	plt.show() 