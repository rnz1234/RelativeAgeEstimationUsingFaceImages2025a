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
from tqdm import tqdm

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Common.Analysis.GeneralMethods import get_statistics

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
    

with open("im2age_map_test.json", 'r') as im2age_map_test_f:
	im2age_map_test = json.load(im2age_map_test_f)
                  


test_err_distribution = get_statistics(dataset_metadata=y_test,
                                       dataset_indexes=[i for i in range(len(y_test))],#chosen_idxs_tst, 
                                       im2age_map_batst=im2age_map_test)




#####################################################
#           Dataset Creation
#####################################################


# Test - Transforms
transf_tst = transforms.Compose([
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])


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


image_datasets = {
    'val' : test_ds
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

data_loaders = {
    'val': DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=False),
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
	dropout_p=cfg.DROPOUT_P
)

model.to(device)

if cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH:
    model.freeze_base_cnn(True)

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
loaded = torch.load("/data/rans/time_25_11_2023_23_47_13/weights_26_2.5486.pt")
model.load_state_dict(loaded['model_state_dict'], strict=True)#, map_location=torch.device('cuda:0')))#, strict=False)
			

running_mae_age = 0.0

im2age_map_next = dict()
print("running inference...")
for batch in tqdm(data_loaders['val']):
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
		age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r = model(input_images=image_vec, query_noisy_age=query_age_noised, input_ref_ages=age_refs) 
		
		running_mae_age += torch.nn.L1Loss()(age_pred_f.reshape(age_pred_f.shape[0]), query_age) * image_vec.size(0)
		#print(f"age_pred : {age_pred_f.view(-1)}, age actual :{query_age}")

	
	for j in range(len(idxs)):
		im2age_map_next[int(idxs[j].cpu())] = float(age_pred_f[j].cpu())

im2age_map_js = json.dumps(im2age_map_next)

with open(f'im2age_map_test_next.json', 'w') as fmap:
	fmap.write(im2age_map_js)
      
mae_age = running_mae_age / dataset_sizes['val']
print(f"MAE : {mae_age}")
