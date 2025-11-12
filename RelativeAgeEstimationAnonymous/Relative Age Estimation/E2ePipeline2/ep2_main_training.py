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
from Common.Losses.MeanVarianceLoss import MeanVarianceLoss
from tqdm import tqdm

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Common.Analysis.GeneralMethods import get_statistics, get_statistics_range

import ep2_config as cfg
from ep2_dataset import QueryAndMultiAgeRefsDataset
from ep2_model import DiffBasedAgeDetectionModel
from ep2_train import train



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
data_parser = DataParser('../Common/Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5', small_data=cfg.SMALL_DATA)
data_parser.initialize_data()


x_train, y_train, x_test, y_test, chosen_idxs_trn, chosen_idxs_tst = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test, data_parser.chosen_idxs_trn, data_parser.chosen_idxs_tst
if cfg.RANDOM_SPLIT:
    all_images = np.concatenate((x_train, x_test), axis=0)
    all_labels = np.concatenate((y_train, y_test), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=cfg.TEST_SIZE_FOR_RS, random_state=cfg.RANDOM_SEED)

#####################################################
#           Metadata Loading
#####################################################

# Emebeddings
face2emb_arr_trn_r = np.load('face2emb_arr_trn_recog.npy', allow_pickle=True)
face2emb_arr_vld_r = np.load('face2emb_arr_vld_recog.npy', allow_pickle=True)

if cfg.SMALL_DATA:
    face2emb_arr_trn_r = face2emb_arr_trn_r[chosen_idxs_trn]
    face2emb_arr_vld_r = face2emb_arr_vld_r[chosen_idxs_tst]
    

with open(cfg.INPUT_ESTIMATION_FILE_NAME, 'r') as im2age_map_test_f:
	im2age_map_test = json.load(im2age_map_test_f)
                  


test_err_distribution = get_statistics(dataset_metadata=y_test,
                                       dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
                                       im2age_map_batst=im2age_map_test)

# test_err_distribution_low = get_statistics_range(dataset_metadata=y_test,
#                                        dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
#                                        im2age_map_batst=im2age_map_test,
#                                         age_range_min=0, 
#                                         age_range_max=49)

# test_err_distribution_high = get_statistics_range(dataset_metadata=y_test,
#                                        dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
#                                        im2age_map_batst=im2age_map_test,
#                                         age_range_min=50, 
#                                         age_range_max=90)


# test_err_distribution = {
#     "low" : test_err_distribution_low,
#     "high" : test_err_distribution_high,
#     "mid_value" : 50
# }

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

# Training set
train_ds = QueryAndMultiAgeRefsDataset(
	min_age=cfg.MIN_AGE,
	max_age=cfg.MAX_AGE,
	age_interval=cfg.AGE_INTERVAL,
	transform=transf_trn,
	num_references=cfg.NUM_REFERENCES,
	embeddings_knn=cfg.EMBEDDINGS_KNN,
	base_data_set_images=x_train,                
	base_data_set_metadata=y_train,   
	base_data_set_embeddings=face2emb_arr_trn_r,            
	ref_data_set_images=x_train,                
	ref_data_set_metadata=y_train,              
	ref_data_set_embeddings=face2emb_arr_trn_r,
	dataset_size_factor=cfg.DATASET_SIZE_FACTOR,
	base_set_is_ref_set=True,
	disable_same_ref_being_query=cfg.DISABLE_SAME_REF_BEING_QUERY,
	knn_reduced_pool_size=cfg.KNN_REDUCED_POOL_SIZE,
	sample_knn_reduced_pool=False, #True,
    base_model_distribution=test_err_distribution,
	im2age_map=None,
	mode_select="apply_distribution"
	)

print("Training (q trn, r trn) set size: " + str(len(train_ds)))

# Training set
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


image_datasets = {
    'train': train_ds,
    'val' : test_ds
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

data_loaders = {
    'train': DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
    'val': DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
}




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
	dropout_p=cfg.DROPOUT_P,
    num_of_fc_layers=cfg.NUM_OF_FC_LAYERS,
    is_ordinal=cfg.IS_ORDINAL,
    min_age_diff=min_age_diff,
	max_age_diff=max_age_diff,
	num_classes_diff=num_classes_diff,
    regressors_diff_head=cfg.REGRESSORS_DIFF_HEAD
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

criterion_cls = torch.nn.CrossEntropyLoss().to(device)
if cfg.IS_MEAN_VAR_LOSS:
	criterion_mean_var = MeanVarianceLoss(0, num_classes_diff, device, lambda_mean=0.2, lambda_variance=0.05).to(device)
else:
    criterion_mean_var = None

#####################################################
#           Experiment Management
#####################################################

cur_time = datetime.now()
cur_time_str = cur_time.strftime("time_%d_%m_%Y_%H_%M_%S")
experiment_name = cur_time_str

writer = SummaryWriter('logs/Morph2Diff/unified/iter/' + experiment_name) 

model_path = 'weights/Morph2Diff/unified/iter/' + experiment_name
if not os.path.exists(model_path):
    os.makedirs(model_path)


shutil.copyfile("ep2_config.py", model_path + "/ep2_config.py")
shutil.copyfile("ep2_dataset.py", model_path + "/ep2_dataset.py")
shutil.copyfile("ep2_model.py", model_path + "/ep2_model.py")
shutil.copyfile("ep2_train.py", model_path + "/ep2_train.py")
shutil.copyfile("ep2_main_training.py", model_path + "/ep2_main_training.py")

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
    criterion_mean_var=None,
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
    regressors_diff_head=cfg.REGRESSORS_DIFF_HEAD
)

print('saving best model')

final_model_file = os.path.join(model_path, "weights.pt")
torch.save(best_diff_model.state_dict(), final_model_file)

# for batch in data_loaders['train']:
#     model(input_images=batch["image_vec"], input_ref_ages=batch["age_refs"])
#     print("###### New Batch")
#     for field_name in ['general_query_idx', 'actual_query_idx', 'ref_idxs', 'age_diffs_for_reg', 'age_diffs_for_cls', 'age_refs']:
#         print(f"- {field_name} : {batch[field_name]}")
        

