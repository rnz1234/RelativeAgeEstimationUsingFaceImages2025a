##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline3
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
from Common.Datasets.Morph2.dataset_utils import gen_dist_and_isol_test_sets_no_im2age_map
from Common.Losses.MeanVarianceLoss import MeanVarianceLoss
from Common.Losses.YprojOrdinalMeanVarianceLoss import OrdinalMeanVarianceLoss
from tqdm import tqdm

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Common.Analysis.GeneralMethods import get_statistics, get_statistics_range
from Common.Datasets.Morph2.dataset_utils import *
from Common.Datasets.CACD.CacdDataParser import CacdDataParser



import ep3_config as cfg
from ep3_dataset import QueryAndMultiAgeRefsDataset
from ep3_model import DiffBasedAgeDetectionModel, PerNoisyRangeAgeModel
from ep3_train import train



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
# data_parser = DataParser('../Common/Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5', small_data=cfg.SMALL_DATA)
# data_parser.initialize_data()
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
	print("Random split mode is currently not supported. Aborting")
	exit()

	all_images = np.concatenate((x_train, x_test), axis=0)
	all_labels = np.concatenate((y_train, y_test), axis=0)

	x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=cfg.TEST_SIZE_FOR_RS, random_state=cfg.RANDOM_SEED)

#####################################################
#           Metadata Loading
#####################################################

# Emebeddings loading
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

	# Outcome to next stages
	face2emb_arr_trn_r_actual = face2emb_arr_trn_r
	face2emb_arr_vld_r_actual = face2emb_arr_vld_r[isolated_test_indexed]

	base_model_err_dist_on_non_trained_set = get_statistics(dataset_metadata=y_test_dist,
										dataset_indexes=[i for i in range(len(y_test_dist))],#chosen_idxs_tst, 
										im2age_map_batst=im2age_map_dist)
	
	x_test_actual = x_test_isol
	y_test_actual = y_test_isol
	x_train_actual = x_train
	y_train_actual = y_train
	im2age_map_test_actual = im2age_map_isol

else:
	print("NOT applying dist and isol test sets split")

	# Outcome to next stages
	face2emb_arr_trn_r_actual = face2emb_arr_trn_r
	face2emb_arr_vld_r_actual = face2emb_arr_vld_r
    
	# dist on full test set
	base_model_err_dist_on_non_trained_set = get_statistics(dataset_metadata=y_test,
										dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
										im2age_map_batst=im2age_map_test)
	x_test_actual = x_test
	y_test_actual = y_test
	x_train_actual = x_train
	y_train_actual = y_train
	im2age_map_test_actual = im2age_map_test

# base_model_err_dist_on_non_trained_set_low = get_statistics_range(dataset_metadata=y_test,
#                                        dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
#                                        im2age_map_batst=im2age_map_test,
#                                         age_range_min=0, 
#                                         age_range_max=49)

# base_model_err_dist_on_non_trained_set_high = get_statistics_range(dataset_metadata=y_test,
#                                        dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
#                                        im2age_map_batst=im2age_map_test,
#                                         age_range_min=50, 
#                                         age_range_max=90)


# base_model_err_dist_on_non_trained_set = {
#     "low" : base_model_err_dist_on_non_trained_set_low,
#     "high" : base_model_err_dist_on_non_trained_set_high,
#     "mid_value" : 50
# }

if cfg.DIST_APPROX_METHOD == "kde_based_saturated":
	min_age_diff = cfg.ERROR_SAT_RANGE_MIN
	max_age_diff = cfg.ERROR_SAT_RANGE_MAX
else:
	min_age_diff = cfg.MIN_AGE - cfg.MAX_AGE 
	max_age_diff = cfg.MAX_AGE - cfg.MIN_AGE 

num_classes_diff = max_age_diff - min_age_diff + 1
print(f"num of diff classes: {num_classes_diff}")

#####################################################
#           Dataset Creation
#####################################################

# Train - Transforms
transf_trn = transforms.Compose([
        transforms.RandomResizedCrop(224, (0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1
        )], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            interpolation=Image.BICUBIC
        )], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])

# Test - Transforms
transf_tst = transforms.Compose([
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])


if cfg.REF_SUBSET:
	print("using subset of reference set")
	ref_set_factor = 0.01
	ref_set_size = int(ref_set_factor*x_train_actual.shape[0])
	chosen_idxs_for_ref_set = np.random.choice(x_train_actual.shape[0], ref_set_size, replace=False)
	ref_data_set_images = x_train_actual[chosen_idxs_for_ref_set]
	ref_data_set_metadata = y_train_actual[chosen_idxs_for_ref_set]
	ref_data_set_embeddings = face2emb_arr_trn_r_actual[chosen_idxs_for_ref_set]
else:
	print("using all of reference set")
	ref_data_set_images = x_train_actual
	ref_data_set_metadata = y_train_actual
	ref_data_set_embeddings = face2emb_arr_trn_r_actual


# Training set
train_ds = QueryAndMultiAgeRefsDataset(
	min_age=cfg.MIN_AGE,
	max_age=cfg.MAX_AGE,
	age_interval=cfg.AGE_INTERVAL,
	transform=transf_trn,
	num_references=cfg.NUM_REFERENCES,
	embeddings_knn=cfg.EMBEDDINGS_KNN,
	base_data_set_images=x_train_actual,                
	base_data_set_metadata=y_train_actual,   
	base_data_set_embeddings=face2emb_arr_trn_r_actual,            
	ref_data_set_images=x_train_actual,                
	ref_data_set_metadata=y_train_actual,              
	ref_data_set_embeddings=face2emb_arr_trn_r_actual,
	dataset_size_factor=cfg.DATASET_SIZE_FACTOR,
	base_set_is_ref_set=True,
	disable_same_ref_being_query=cfg.DISABLE_SAME_REF_BEING_QUERY,
	knn_reduced_pool_size=cfg.KNN_REDUCED_POOL_SIZE,
	sample_knn_reduced_pool=cfg.SAMPLE_KNN_REDUCED_POOL_TRAIN, #True,
    base_model_distribution=base_model_err_dist_on_non_trained_set,
	im2age_map=None,
	mode_select="apply_distribution",
    distribution_approximation_method=cfg.DIST_APPROX_METHOD,
	use_knn=cfg.USE_KNN
	)

print("Training (q trn, r trn) set size: " + str(len(train_ds)))



# Test set
test_ds = QueryAndMultiAgeRefsDataset(
	min_age=cfg.MIN_AGE,
	max_age=cfg.MAX_AGE,
	age_interval=cfg.AGE_INTERVAL,
	transform=transf_tst,
	num_references=cfg.NUM_REFERENCES,
	embeddings_knn=cfg.EMBEDDINGS_KNN,
	base_data_set_images=x_test_actual,                
	base_data_set_metadata=y_test_actual,   
	base_data_set_embeddings=face2emb_arr_vld_r_actual,            
	ref_data_set_images=ref_data_set_images,                
	ref_data_set_metadata=ref_data_set_metadata,              
	ref_data_set_embeddings=ref_data_set_embeddings,
	dataset_size_factor=cfg.DATASET_SIZE_FACTOR,
	base_set_is_ref_set=False,
	disable_same_ref_being_query=False,
	knn_reduced_pool_size=cfg.KNN_REDUCED_POOL_SIZE,
	sample_knn_reduced_pool=cfg.SAMPLE_KNN_REDUCED_POOL_TEST,
    base_model_distribution=None,
	im2age_map=im2age_map_test_actual,
	mode_select="apply_map",
	use_knn=cfg.USE_KNN
    )

print("Testing (q vld, r trn) set size: " + str(len(test_ds)))


image_datasets = {
    'train': train_ds,
    'val' : test_ds
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

data_loaders = {
    'train': DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
    'val': DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER_VAL, pin_memory=True, shuffle=False, drop_last=True),
}




#####################################################
#           Model
#####################################################

if cfg.MODEL_PER_RANGE:
	model_range1 = DiffBasedAgeDetectionModel(
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
    
	model_range2 = DiffBasedAgeDetectionModel(
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
       
	model = PerNoisyRangeAgeModel(
        model_range1=model_range1,
        model_range2=model_range2, 
        device=device,
        threshold=cfg.RANGE_THRESHOLD
	)

else:
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

if cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH:
    model.freeze_base_cnn(True)

if cfg.USE_GPU and cfg.MULTI_GPU:
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs (" + str(torch.cuda.device_count()) + ")")
        model = torch.nn.DataParallel(model)


criterion_age = nn.MSELoss().to(device)
criterion_age_diff = nn.MSELoss().to(device)


if cfg.APPLY_WEIGHT_DECAY:
    optimizer = RangerLars(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY_VAL)
else:
	optimizer = RangerLars(model.parameters(), lr=cfg.LEARNING_RATE)
print("optimizer: using RangerLars")



cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=cfg.NUM_ITERS
)
scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier=1,
    total_epoch=10000,
    after_scheduler=cosine_scheduler
)
print("scheduler: using CosineAnnealingLR+GradualWarmupScheduler")

if cfg.FULL_MODEL_PRETRAINED_WEIGHTS:
	print("loading weights...")
	#loaded = torch.load("/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_24_12_2023_16_20_38/weights_54_2.5240.pt")
	loaded = torch.load(cfg.FULL_MODEL_PRETRAINED_WEIGHTS_PATH)
	model.load_state_dict(loaded['model_state_dict'], strict=False)#, map_location=torch.device('cuda:0')))#, strict=False)
	#optimizer.load_state_dict(loaded['optimizer_state_dict'])
	#scheduler.load_state_dict(loaded['scheduler'])


criterion_cls = torch.nn.CrossEntropyLoss().to(device)
# if cfg.IS_MEAN_VAR_LOSS:
# 	criterion_mean_var = MeanVarianceLoss(0, num_classes_diff, device, lambda_mean=0.2, lambda_variance=0.05).to(device)
# else:
#     criterion_mean_var = None

criterion_cls_main_reg = torch.nn.CrossEntropyLoss().to(device)
criterion_mean_var_main_reg = MeanVarianceLoss(0, num_classes_diff, device, lambda_mean=0.2, lambda_variance=0.05).to(device)
criterion_ordinal_mean_var_side = None #OrdinalMeanVarianceLoss(LamdaMean=0.2, LamdaVar=0.05, device=device)
criterion_gender_est = torch.nn.BCEWithLogitsLoss()

#####################################################
#           Experiment Management
#####################################################

cur_time = datetime.now()
cur_time_str = cur_time.strftime("time_%d_%m_%Y_%H_%M_%S")
experiment_name = cur_time_str

writer = SummaryWriter(f'logs/{cfg.DATASET_SELECT}Diff/unified/iter/' + experiment_name) 

model_path = f'weights/{cfg.DATASET_SELECT}Diff/unified/iter/' + experiment_name
if not os.path.exists(model_path):
    os.makedirs(model_path)


shutil.copyfile("ep3_config.py", model_path + "/ep3_config.py")
shutil.copyfile("ep3_dataset.py", model_path + "/ep3_dataset.py")
shutil.copyfile("ep3_model.py", model_path + "/ep3_model.py")
shutil.copyfile("ep3_train.py", model_path + "/ep3_train.py")
shutil.copyfile("ep3_main_training.py", model_path + "/ep3_main_training.py")
shutil.copyfile("ep3_infer.py", model_path + "/ep3_infer.py")

#####################################################
#           Train
#####################################################

best_diff_model = train(
	device=device,
	model=model, 
	multi_gpu=cfg.MULTI_GPU,
	dataloaders=data_loaders,
	dataset_sizes=dataset_sizes,
	criterion_age=criterion_age,
	criterion_age_diff=criterion_age_diff,
    criterion_cls=criterion_cls,
    criterion_mean_var=None,#criterion_mean_var,
	criterion_cls_main_reg=criterion_cls_main_reg,
	criterion_mean_var_main_reg=criterion_mean_var_main_reg,
    criterion_gender_est=criterion_gender_est,
    criterion_ordinal_mean_var_side=criterion_ordinal_mean_var_side,
	optimizer=optimizer,
    scheduler=scheduler,
	writer=writer,
	model_path=model_path,    
    remove_older_checkpoints=cfg.REMOVE_OLDER_CHECKPOINTS,
    save_all_model_metadata=cfg.SAVE_ALL_MODEL_METADATA,
	num_references=cfg.NUM_REFERENCES,
	num_epochs=cfg.NUM_EPOCHS,
	unfreeze_feature_ext_epoch=cfg.UNFREEZE_FEATURE_EXT_EPOCH,
	unfreeze_feature_ext_on_rlvnt_epoch=cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH,
    num_classes_diff=num_classes_diff,
	is_ordinal=cfg.IS_ORDINAL,
    regressors_diff_head=cfg.REGRESSORS_DIFF_HEAD,
    gender_factor=cfg.GENDER_FACTOR
)

print('saving best model')

final_model_file = os.path.join(model_path, "weights.pt")
torch.save(best_diff_model.state_dict(), final_model_file)

# for batch in data_loaders['train']:
#     model(input_images=batch["image_vec"], input_ref_ages=batch["age_refs"])
#     print("###### New Batch")
#     for field_name in ['general_query_idx', 'actual_query_idx', 'ref_idxs', 'age_diffs_for_reg', 'age_diffs_for_cls', 'age_refs']:
#         print(f"- {field_name} : {batch[field_name]}")
        

