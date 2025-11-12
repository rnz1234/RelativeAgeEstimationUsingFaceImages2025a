##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline3
#	Date		:	1.11.2023
# 	Description	: 	generate split of the test set to dist and isol
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
from Common.Losses.MeanVarianceLoss import MeanVarianceLoss
from tqdm import tqdm

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Common.Analysis.GeneralMethods import get_statistics, get_statistics_range
from Common.Datasets.Morph2.dataset_utils import *
from Common.Datasets.CACD.CacdDataParser import CacdDataParser

import ep3_config as cfg
from ep3_dataset import QueryAndMultiAgeRefsDataset
from ep3_model import DiffBasedAgeDetectionModel
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

print(f"Dataset: {cfg.DATASET_SELECT}")

# Load data
if cfg.DATASET_SELECT == "Morph2":
	data_parser = DataParser(cfg.MORPH2_DATASET_PATH, small_data=False)
elif cfg.DATASET_SELECT == "CACD":
	data_parser = CacdDataParser(cfg.CACD_DATASET_PATH)
data_parser.initialize_data()

if cfg.DATASET_SELECT == "Morph2":
    x_train, y_train, x_test, y_test, _, _ = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test, data_parser.chosen_idxs_trn, data_parser.chosen_idxs_tst
elif cfg.DATASET_SELECT == "CACD":
    x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test

# if cfg.RANDOM_SPLIT:
#     all_images = np.concatenate((x_train, x_test), axis=0)
#     all_labels = np.concatenate((y_train, y_test), axis=0)

#     x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=cfg.TEST_SIZE_FOR_RS, random_state=cfg.RANDOM_SEED)


dist_indexes, isolated_test_indexed = split_set_to_id_isolated_idxs_sets(y_test, set_1_factor=cfg.DIST_SET_SIZE_FACTOR)

with open(f'{cfg.DATASET_SELECT}_dist_indexes.pkl', 'wb') as f_dist_indexes:
	pickle.dump(dist_indexes, f_dist_indexes)
      
print("saved dist_indexes")
     
with open(f'{cfg.DATASET_SELECT}_isolated_test_indexed.pkl', 'wb') as f_isolated_test_indexed:
	pickle.dump(isolated_test_indexed, f_isolated_test_indexed)
      
print("saved isolated_test_indexed")