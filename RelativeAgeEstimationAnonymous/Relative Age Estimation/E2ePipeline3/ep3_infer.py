##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline2
#	Date		:	1.11.2023
# 	Description	: 	Main run file
##############################################################################

import shutil

# importing the sys module
import sys        
 
# appending the directory of mod.py
# in the sys.path list
sys.path.append('../')   

import json
import pickle
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import utils
from datetime import datetime
from Common.Datasets.Morph2.data_parser import DataParser
from tqdm import tqdm

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Common.Analysis.GeneralMethods import get_statistics
from Common.Datasets.CACD.CacdDataParser import CacdDataParser
from Common.Datasets.Morph2.dataset_utils import *

import ep3_config as cfg
from ep3_dataset import QueryAndMultiAgeRefsDataset
from ep3_model import DiffBasedAgeDetectionModel
from ep3_train import train

from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

profiler_output_dir = "./profiler_logs"

# Create output directory if it doesn't exist
os.makedirs(profiler_output_dir, exist_ok=True)


#####################################################
#           Preparations
#####################################################

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
random.seed(cfg.RANDOM_SEED)

if cfg.USE_GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(device)

torch.cuda.empty_cache()

#####################################################
#           Data Loading
#####################################################

# Load data
print(f"Dataset: {cfg.DATASET_SELECT}")
print("Reading dataset...")

# Load data
if cfg.DATASET_SELECT == "Morph2":
	data_parser = DataParser(cfg.MORPH2_DATASET_PATH, small_data=cfg.SMALL_DATA)
	data_parser.initialize_data()
	x_train, y_train, x_test, y_test, chosen_idxs_trn, chosen_idxs_tst = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test, data_parser.chosen_idxs_trn, data_parser.chosen_idxs_tst
elif cfg.DATASET_SELECT == "CACD":
	data_parser = CacdDataParser(cfg.CACD_DATASET_PATH)
	data_parser.initialize_data()
	x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test

if cfg.RANDOM_SPLIT:
    all_images = np.concatenate((x_train, x_test), axis=0)
    all_labels = np.concatenate((y_train, y_test), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=cfg.TEST_SIZE_FOR_RS, random_state=cfg.RANDOM_SEED)

#####################################################
#           Metadata Loading
#####################################################

# Emebeddings
face2emb_arr_trn_r = np.load(f'{cfg.DATASET_SELECT}_face2emb_arr_trn_recog.npy', allow_pickle=True)
face2emb_arr_vld_r = np.load(f'{cfg.DATASET_SELECT}_face2emb_arr_vld_recog.npy', allow_pickle=True)

if cfg.SMALL_DATA:
	if cfg.APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL or cfg.APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL:
		print("Unsupported modes from small data. Please cancel these modes and rerun. Aborting")
		exit()

	if cfg.DATASET_SELECT == "Morph2":
		face2emb_arr_trn_r = face2emb_arr_trn_r[chosen_idxs_trn]
		face2emb_arr_vld_r = face2emb_arr_vld_r[chosen_idxs_tst]

# Base model inference results loading
with open(cfg.INPUT_ESTIMATION_FILE_NAME_TEST, 'r') as im2age_map_test_f:
	im2age_map_test = json.load(im2age_map_test_f)
                  
with open(cfg.INPUT_ESTIMATION_FILE_NAME_TRAIN, 'r') as im2age_map_train_and_dist_f:
	im2age_map_train_and_dist = json.load(im2age_map_train_and_dist_f)
                  

"""
Creatig now these for next stages:

train actual
test actual 
dist
embeddings train actual
embeddings test actual
map test
base_model_err_dist_on_non_trained_set
"""


if cfg.APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL:
	print("applying dist and isol train sets split")
	# load dist and isol test indexes
	with open(f'{cfg.INDEXES_SAVE_DIR}/{cfg.DATASET_SELECT}_dist_indexes.pkl', 'rb') as f_dist_indexes:
		dist_indexes = pickle.load(f_dist_indexes)
	with open(f'{cfg.INDEXES_SAVE_DIR}/{cfg.DATASET_SELECT}_isolated_train_indexes.pkl', 'rb') as f_isolated_train_indexes:
		isolated_train_indexes = pickle.load(f_isolated_train_indexes)


	print(f"Original train set size: {len(data_parser.y_train)}")

	x_train_dist, y_train_dist, im2age_map_train_dist, x_train_isol, y_train_isol, im2age_map_train_isol = gen_dist_and_isol_test_sets(x_src_dataset=data_parser.x_train, 
																										y_src_dataset=data_parser.y_train, 
																										im2age_map_src_dataset_orig=im2age_map_train_and_dist, 
																										dist_indexes=dist_indexes, 
																										isolated_src_dataset_indexed=isolated_train_indexes)

	print(f"Actual train set size: {len(y_train_isol)}")

	# Outcome to next stages
	face2emb_arr_trn_r_actual = face2emb_arr_trn_r[isolated_train_indexes]
	face2emb_arr_vld_r_actual = face2emb_arr_vld_r
	
	base_model_err_dist_on_non_trained_set = get_statistics(dataset_metadata=y_train_dist,
										dataset_indexes=[i for i in range(len(y_train_dist))], 
										im2age_map_batst=im2age_map_train_dist)
	
	#print(f"""MAE (dist): {np.mean(np.abs(base_model_err_dist_on_non_trained_set["data"]))}""")
	
	# train_set_stats = get_statistics(dataset_metadata=y_train_isol,
	# 									dataset_indexes=[i for i in range(len(y_train_isol))], 
	# 									im2age_map_batst=im2age_map_train_isol)
	
	# print(f"""MAE (train): {np.mean(np.abs(train_set_stats["data"]))}""")
	
	# test_set_stats = get_statistics(dataset_metadata=y_test,
	# 									dataset_indexes=[i for i in range(len(y_test))], 
	# 									im2age_map_batst=im2age_map_test)
	
	# print(f"""MAE (test): {np.mean(np.abs(test_set_stats["data"]))}""")
	

	x_test_actual = x_test
	y_test_actual = y_test
	x_train_actual = x_train_isol
	y_train_actual = y_train_isol
	im2age_map_test_actual = im2age_map_test
elif cfg.APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL:
	print("applying dist and isol test sets split")
	# load dist and isol test indexes
	with open(f'{cfg.DATASET_SELECT}_dist_indexes.pkl', 'rb') as f_dist_indexes:
			dist_indexes = pickle.load(f_dist_indexes)
	with open(f'{cfg.DATASET_SELECT}_isolated_test_indexed.pkl', 'rb') as f_isolated_test_indexed:
			isolated_test_indexed = pickle.load(f_isolated_test_indexed)

	x_test_dist, y_test_dist, im2age_map_dist, x_test_isol, y_test_isol, im2age_map_isol = gen_dist_and_isol_test_sets(x_test, y_test, im2age_map_test, dist_indexes, isolated_test_indexed)


	face2emb_arr_vld_r_actual = face2emb_arr_vld_r[isolated_test_indexed]

	test_err_distribution = get_statistics(dataset_metadata=y_test_dist,
										dataset_indexes=[i for i in range(len(y_test_dist))],#chosen_idxs_tst, 
										im2age_map_batst=im2age_map_dist)
	x_test_actual = x_test_isol
	y_test_actual = y_test_isol
	im2age_map_test_actual = im2age_map_isol
else:
	print("NOT applying dist and isol test sets split")
	face2emb_arr_vld_r_actual = face2emb_arr_vld_r
    
	test_err_distribution = get_statistics(dataset_metadata=y_test,
										dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
										im2age_map_batst=im2age_map_test)
	x_test_actual = x_test
	y_test_actual = y_test
	im2age_map_test_actual = im2age_map_test



#####################################################
#           Dataset Creation
#####################################################


# Test - Transforms
transf_tst = transforms.Compose([
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])


# Train set
if cfg.APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL:
	# The original train set is composed from the actual train set and dist set.
	# The dist is isolated from train set completely, hence it is kind of "test set".
	# So we use same settings used with test set definition (e.g. we don't use the 
	# embeddings of the dist set as references). I.e. we run on dist set as if it 
	# was "test set"

	train_ds = QueryAndMultiAgeRefsDataset(
		min_age=cfg.MIN_AGE,
		max_age=cfg.MAX_AGE,
		age_interval=cfg.AGE_INTERVAL,
		transform=transf_tst,
		num_references=cfg.NUM_REFERENCES,
		embeddings_knn=cfg.EMBEDDINGS_KNN,
		base_data_set_images=x_train,                
		base_data_set_metadata=y_train,   
		base_data_set_embeddings=face2emb_arr_trn_r,            
		ref_data_set_images=x_train_actual,                
		ref_data_set_metadata=y_train_actual,              
		ref_data_set_embeddings=face2emb_arr_trn_r_actual,
		dataset_size_factor=cfg.DATASET_SIZE_FACTOR,
		base_set_is_ref_set=False,
		disable_same_ref_being_query=False,
		knn_reduced_pool_size=cfg.KNN_REDUCED_POOL_SIZE,
		sample_knn_reduced_pool=True,
		base_model_distribution=None,
		im2age_map=im2age_map_train_and_dist,
		mode_select="apply_map"
		)
	
	print("Train+Dist (q vld, r trn) set size: " + str(len(train_ds)))
	

# Test set
test_ds = QueryAndMultiAgeRefsDataset(
	min_age=cfg.MIN_AGE,
	max_age=cfg.MAX_AGE,
	age_interval=cfg.AGE_INTERVAL,
	transform=transf_tst,
	num_references=cfg.NUM_REFERENCES,
	embeddings_knn=cfg.EMBEDDINGS_KNN,
	base_data_set_images=x_test,                
	base_data_set_metadata=y_test,   
	base_data_set_embeddings=face2emb_arr_vld_r,            
	ref_data_set_images=x_train,                
	ref_data_set_metadata=y_train,              
	ref_data_set_embeddings=face2emb_arr_trn_r,
	dataset_size_factor=cfg.DATASET_SIZE_FACTOR,
	base_set_is_ref_set=False,
	disable_same_ref_being_query=False,
	knn_reduced_pool_size=cfg.KNN_REDUCED_POOL_SIZE,
	sample_knn_reduced_pool=True,
    base_model_distribution=None,
	im2age_map=im2age_map_test,
	mode_select="apply_map"
    )

print("Testing (q vld, r trn) set size: " + str(len(test_ds)))

if cfg.APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL:
	test_isol_ds = QueryAndMultiAgeRefsDataset(
		min_age=cfg.MIN_AGE,
		max_age=cfg.MAX_AGE,
		age_interval=cfg.AGE_INTERVAL,
		transform=transf_tst,
		num_references=cfg.NUM_REFERENCES,
		embeddings_knn=cfg.EMBEDDINGS_KNN,
		base_data_set_images=x_test_actual,                
		base_data_set_metadata=y_test_actual,   
		base_data_set_embeddings=face2emb_arr_vld_r_actual,            
		ref_data_set_images=x_train,                
		ref_data_set_metadata=y_train,              
		ref_data_set_embeddings=face2emb_arr_trn_r,
		dataset_size_factor=cfg.DATASET_SIZE_FACTOR,
		base_set_is_ref_set=False,
		disable_same_ref_being_query=False,
		knn_reduced_pool_size=cfg.KNN_REDUCED_POOL_SIZE,
		sample_knn_reduced_pool=True,
		base_model_distribution=None,
		im2age_map=im2age_map_test_actual,
		mode_select="apply_map"
		)

	print("Testing isolated (q vld, r trn) set size: " + str(len(test_isol_ds)))


# full test 
image_datasets = {
    'val_full' : test_ds
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['val_full']}

data_loaders = {
    'val_full': DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=False),
}

# other sets
if cfg.APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL:
	image_datasets['val_isol'] = test_isol_ds
	dataset_sizes['val_isol'] = len(image_datasets['val_isol'])
	data_loaders['val_isol'] = DataLoader(test_isol_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=False)

if cfg.APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL:
	image_datasets['train'] = train_ds
	dataset_sizes['train'] = len(image_datasets['train'])
	data_loaders['train'] = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=False)


if cfg.DIST_APPROX_METHOD == "kde_based_saturated":
	min_age_diff = cfg.ERROR_SAT_RANGE_MIN
	max_age_diff = cfg.ERROR_SAT_RANGE_MAX
else:
	min_age_diff = cfg.MIN_AGE - cfg.MAX_AGE 
	max_age_diff = cfg.MAX_AGE - cfg.MIN_AGE 
	
num_classes_diff = max_age_diff - min_age_diff + 1
print(f"num of diff classes: {num_classes_diff}")


#####################################################
#           Model
#####################################################

model = DiffBasedAgeDetectionModel(
    device=device,
	min_age=cfg.MIN_AGE,
	max_age=cfg.MAX_AGE,
	age_interval=cfg.AGE_INTERVAL,
	num_references=cfg.NUM_REFERENCES,
	pretrained_model_path=cfg.PRETRAINED_MODEL_PATH,
	pretrained_model_file_name=cfg.PRETRAINED_MODEL_FILE_NAME,
	load_pretrained=cfg.LOAD_PRETRAINED_RECOG,
	dropout_p=cfg.DROPOUT_P,
    num_of_fc_layers=cfg.NUM_OF_FC_LAYERS,
    is_ordinal=cfg.IS_ORDINAL,
    min_age_diff=min_age_diff,
	max_age_diff=max_age_diff,
	num_classes_diff=num_classes_diff,
    regressors_diff_head=cfg.REGRESSORS_DIFF_HEAD,
	fc_head_base_layer_size=cfg.FC_HEAD_BASE_LAYER_SIZE,
	use_vit=cfg.USE_VIT,
	use_convnext=cfg.USE_CONVNEXT,
	use_efficientnet=cfg.USE_EFFICIENTNET,
	use_resnet51q=cfg.USE_RESNET51Q
)

model.to(device)

# if cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH:
#     model.freeze_base_cnn(True)

if cfg.USE_GPU and cfg.MULTI_GPU:
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs (" + str(torch.cuda.device_count()) + ")")
        model = torch.nn.DataParallel(model)


# test_err_distribution = get_statistics(dataset_metadata=y_test,
#                                        dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
#                                        im2age_map_batst=im2age_map_test)

# mae_age = np.mean(np.abs(test_err_distribution["data"]))
# print(f"MAE : {mae_age}")

model.eval()

print("loading weights...")
#loaded = torch.load("/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_24_12_2023_16_20_38/weights_54_2.5240.pt")
#loaded = torch.load("/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_08_01_2024_22_18_11/weights_59_2.5057.pt")
#loaded = torch.load("/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_17_01_2024_01_18_28/weights_19_2.4980.pt")
#loaded = torch.load("/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_19_01_2024_23_38_06/weights_19_2.4949.pt")
#loaded = torch.load("/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_20_01_2024_22_06_08/weights_1_2.4824.pt")
loaded = torch.load(cfg.INFERENCE_MODEL_WEIGHTS_PATH)
model.load_state_dict(loaded['model_state_dict'], strict=True)#, map_location=torch.device('cuda:0')))#, strict=False)
			

running_mae_age = 0.0


if cfg.RUN_PROFILER:
	print("running inference...")
	print("######################################################################################")
	print("### Full test set - Profiling Session")

	with profile(
		activities=[
			ProfilerActivity.CPU,
			ProfilerActivity.CUDA
		],
		schedule=schedule(wait=1, warmup=1, active=10, repeat=1),  # customizable
		on_trace_ready=tensorboard_trace_handler(profiler_output_dir),
		record_shapes=True,
		with_stack=True,
		profile_memory=True,
	) as prof:

		for batch in tqdm(data_loaders['val_full']):
			image_vec = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
			query_age = batch['query_age'].to(device).float()
			query_age_noised = batch['query_age_noised'].to(device).long()
			age_diff = batch['age_diffs_for_reg'].to(device).float() #torch.stack([batch['age_diffs_for_reg'][i].to(device).float() for i in range(num_references)])
			age_refs = batch['age_refs'].to(device).long() #torch.stack([batch['age_refs'][i].to(device).float() for i in range(num_references)])
			idxs = batch['actual_query_idx'].to(device)

			with torch.no_grad():
				# age_pred, age_diff_preds = model(input_images=image_vec, input_ref_ages=age_refs)
				# age_loss = 	criterion_age(age_pred.reshape(age_pred.shape[0]), query_age)
				# age_diff_loss = criterion_age_diff(age_diff_preds, age_diff)
				if cfg.USE_GENDER:
					age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus, gender_head_cls_pre_sigmoid = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
				else:
					age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
				
				if cfg.INFERENCE_BASED_ON_F:
					running_mae_age += torch.nn.L1Loss()(age_pred_f.reshape(age_pred_f.shape[0]), query_age) * image_vec.size(0)
				else:
					running_mae_age += torch.nn.L1Loss()(age_pred_r.reshape(age_pred_r.shape[0]), query_age) * image_vec.size(0)

			prof.step()  # Step profiler after each batch

SKIP_TEST = False
if not SKIP_TEST:
	im2age_map_test_next = dict()
	print("running inference...")
	print("######################################################################################")
	print("### Full test set - formal evaluation (generate json file)")

	for batch in tqdm(data_loaders['val_full']):
		image_vec = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
		query_age = batch['query_age'].to(device).float()
		query_age_noised = batch['query_age_noised'].to(device).long()
		age_diff = batch['age_diffs_for_reg'].to(device).float() #torch.stack([batch['age_diffs_for_reg'][i].to(device).float() for i in range(num_references)])
		age_refs = batch['age_refs'].to(device).long() #torch.stack([batch['age_refs'][i].to(device).float() for i in range(num_references)])
		idxs = batch['actual_query_idx'].to(device)

		with torch.no_grad():
			# age_pred, age_diff_preds = model(input_images=image_vec, input_ref_ages=age_refs)
			# age_loss = 	criterion_age(age_pred.reshape(age_pred.shape[0]), query_age)
			# age_diff_loss = criterion_age_diff(age_diff_preds, age_diff)
			if cfg.USE_GENDER:
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus, gender_head_cls_pre_sigmoid = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
			else:
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
			
			if cfg.INFERENCE_BASED_ON_F:
				running_mae_age += torch.nn.L1Loss()(age_pred_f.reshape(age_pred_f.shape[0]), query_age) * image_vec.size(0)
				print(f"age_pred : {age_pred_f.view(-1)}, age actual :{query_age}")
			else:
				running_mae_age += torch.nn.L1Loss()(age_pred_r.reshape(age_pred_r.shape[0]), query_age) * image_vec.size(0)
				print(f"age_pred : {age_pred_r.view(-1)}, age actual :{query_age}")

		
		for j in range(len(idxs)):
			im2age_map_test_next[int(idxs[j].cpu())] = float(age_pred_r[j].cpu())


	im2age_map_test_js = json.dumps(im2age_map_test_next)

	with open(cfg.INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH, 'w') as fmap:
		fmap.write(im2age_map_test_js)
		
	mae_age = running_mae_age / dataset_sizes['val_full']
	print(f"MAE (full original test set): {mae_age}")


running_mae_age = 0.0
if cfg.APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL:
	im2age_map_train_next = dict()
	print("#############################################################################")
	print("### Train set including Dist (generate json file for next potential training)")
	for batch in tqdm(data_loaders['train']):
		image_vec = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
		query_age = batch['query_age'].to(device).float()
		query_age_noised = batch['query_age_noised'].to(device).long()
		age_diff = batch['age_diffs_for_reg'].to(device).float() #torch.stack([batch['age_diffs_for_reg'][i].to(device).float() for i in range(num_references)])
		age_refs = batch['age_refs'].to(device).long() #torch.stack([batch['age_refs'][i].to(device).float() for i in range(num_references)])
		idxs = batch['actual_query_idx'].to(device)

		with torch.no_grad():
			# age_pred, age_diff_preds = model(input_images=image_vec, input_ref_ages=age_refs)
			# age_loss = 	criterion_age(age_pred.reshape(age_pred.shape[0]), query_age)
			# age_diff_loss = criterion_age_diff(age_diff_preds, age_diff)
			if cfg.USE_GENDER:
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus, gender_head_cls_pre_sigmoid = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
			else:
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
			
			if cfg.INFERENCE_BASED_ON_F:
				running_mae_age += torch.nn.L1Loss()(age_pred_f.reshape(age_pred_f.shape[0]), query_age) * image_vec.size(0)
				print(f"age_pred : {age_pred_f.view(-1)}, age actual :{query_age}")
			else:
				running_mae_age += torch.nn.L1Loss()(age_pred_r.reshape(age_pred_r.shape[0]), query_age) * image_vec.size(0)
				print(f"age_pred : {age_pred_r.view(-1)}, age actual :{query_age}")

		
		for j in range(len(idxs)):
			im2age_map_train_next[int(idxs[j].cpu())] = float(age_pred_r[j].cpu())

	im2age_map_train_js = json.dumps(im2age_map_train_next)

	with open(cfg.INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH, 'w') as fmap:
		fmap.write(im2age_map_train_js)
		
	mae_age = running_mae_age / dataset_sizes['train']
	print(f"MAE (train+dist set): {mae_age}")




if cfg.APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL:
	running_mae_age_isol = 0.0

	print("####################################################################")
	print("### Isolated test set (evaluation)")
	for batch in tqdm(data_loaders['val_isol']):
		image_vec = batch['image_vec'].to(device) #batch['image_vec'][:,:2,:,:,:].to(device)
		query_age = batch['query_age'].to(device).float()
		query_age_noised = batch['query_age_noised'].to(device).long()
		age_diff = batch['age_diffs_for_reg'].to(device).float() #torch.stack([batch['age_diffs_for_reg'][i].to(device).float() for i in range(num_references)])
		age_refs = batch['age_refs'].to(device).long() #torch.stack([batch['age_refs'][i].to(device).float() for i in range(num_references)])
		idxs = batch['actual_query_idx'].to(device)

		with torch.no_grad():
			# age_pred, age_diff_preds = model(input_images=image_vec, input_ref_ages=age_refs)
			# age_loss = 	criterion_age(age_pred.reshape(age_pred.shape[0]), query_age)
			# age_diff_loss = criterion_age_diff(age_diff_preds, age_diff)
			if cfg.USE_GENDER:
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus, gender_head_cls_pre_sigmoid = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
			else:
				age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, classification_logits_main_diff_minus = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs)
			
			if cfg.INFERENCE_BASED_ON_F:
				running_mae_age_isol += torch.nn.L1Loss()(age_pred_f.reshape(age_pred_f.shape[0]), query_age) * image_vec.size(0)
				print(f"age_pred : {age_pred_f.view(-1)}, age actual :{query_age}")
			else:
				running_mae_age_isol += torch.nn.L1Loss()(age_pred_r.reshape(age_pred_r.shape[0]), query_age) * image_vec.size(0)
				print(f"age_pred : {age_pred_r.view(-1)}, age actual :{query_age}")

	mae_age = running_mae_age_isol / dataset_sizes['val_isol']
	print(f"MAE (isolated test set): {mae_age}")

