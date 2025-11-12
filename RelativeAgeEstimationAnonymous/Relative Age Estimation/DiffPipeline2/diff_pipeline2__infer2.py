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

from datetime import datetime


import diff_pipeline2__config as cfg

from diff_pipeline2__dataset import DiffPipeline2Dataset
from Common.Datasets.Morph2.data_parser import DataParser
from Common.Datasets.Morph2.Morph2ClassifierDataset2 import Morph2ClassifierDataset2
from diff_pipeline2__train import train_diff_cls_model_iter

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler


from diff_pipeline2__model import DiffPipeline2Model

from Common.Pipelines.AgePredictSingleRefPipeline import AgePredictSingleRefPipeline
#from Common.Analysis.DiffAnalysisMethods import diff_pipeline_confusion_matrix_analysis
from Common.Analysis.DiffAnalysisMethods import evaluate_pipeline_clean


from tqdm import tqdm


from Common.Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel
from Common.Models.transformer import *
from Common.Models.unified_transformer_model import AgeTransformer

from sklearn import utils


if __name__ == "__main__":

	torch.manual_seed(cfg.RANDOM_SEED)
	np.random.seed(cfg.RANDOM_SEED)
	random.seed(cfg.RANDOM_SEED)

	if cfg.USE_GPU:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device("cpu")
		
	print(device)

	torch.cuda.empty_cache()





	# Load data
	data_parser = DataParser('../Common/Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5', small_data=cfg.SMALL_DATA)
	data_parser.initialize_data()

	x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test,
	if cfg.RANDOM_SPLIT:
		all_images = np.concatenate((x_train, x_test), axis=0)
		all_labels = np.concatenate((y_train, y_test), axis=0)

		x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=cfg.TEST_SIZE_FOR_RS, random_state=cfg.RANDOM_SEED)

    

	with open("im2age_map_test.json", 'r') as im2age_map_test_f:
		im2age_map_test = json.load(im2age_map_test_f)

	idx_test = range(len(y_test))

	x_test_shuffled, y_test_shuffled, idx_test_shuffled = utils.shuffle(x_test, y_test, idx_test)

	transf = transforms.Compose([
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
				resample=Image.BICUBIC
			)], p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.RandomErasing(p=0.5)
		])

	# test_ds = DiffPipeline2Dataset(
	# 	dataset_type="test",
	# 	batrn_set_images=x_train,
	# 	batrn_set_metadata=y_train,
	# 	batst_set_images=x_test_shuffled,
	# 	batst_set_metadata=y_test_shuffled,
	# 	batst_set_idx=idx_test_shuffled,
	# 	train_test_ratio=cfg.DIFF_TRAIN_SIZE_RATIO,
	# 	im2age_map_batst=im2age_map_test,
	# 	min_age=cfg.MIN_AGE,
	# 	age_interval=cfg.AGE_INTERVAL,
	# 	max_age=cfg.MAX_AGE,
	# 	transform=transf,
	# 	copies=1, 
	# 	age_radius=cfg.AGE_RADIUS
	# )

	test_ds = Morph2ClassifierDataset2( #Morph2ClassifierDataset
		x_test_shuffled[int(cfg.DIFF_TRAIN_SIZE_RATIO*len(x_test_shuffled)):],
		y_test_shuffled[int(cfg.DIFF_TRAIN_SIZE_RATIO*len(x_test_shuffled)):],
		cfg.MIN_AGE,
		cfg.AGE_INTERVAL,
		transform=transforms.Compose([
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
				resample=Image.BICUBIC
			)], p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.RandomErasing(p=0.5)
		]),
		copies=cfg.NUM_OF_COPIES_AGE_PREDICT
	)

	image_datasets = {
		'train': [],
		'val': test_ds
	}

	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	data_loaders = {
		'train': None, #DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
		'val': DataLoader(test_ds, batch_size=1, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=True)
	}

	
	num_classes = int((cfg.MAX_AGE - cfg.MIN_AGE) / cfg.AGE_INTERVAL + 1)

	
	

	print("testing set size: " + str(len(test_ds)))

	pipeline = AgePredictSingleRefPipeline(
						device,
						num_classes, 
						cfg.AGE_INTERVAL, 
						cfg.MIN_AGE, 
						cfg.MAX_AGE, 
						cfg.AGE_RADIUS,
						cfg.MID_FEATURE_SIZE_AGE_PREDICT, 
						x_train,
						y_train,
						0, #cfg.DEEP, 
						0, #cfg.NUM_OF_REFERENCES, 
						0, #cfg.DIFF_MODEL_CONFIG_TYPE, 
						0, #cfg.ADDED_EMBED_LAYER_SIZE,
						0, #cfg.DIFF_EMBED_LAYER_SIZE,
						transf,
						cfg.NUM_OF_COPIES_AGE_PREDICT,
						UnifiedClassificaionAndRegressionAgeModel,
						TransformerModel,
						AgeTransformer,
						DiffPipeline2Model, #AgeDiffModelNonOpt, #AgeDiffModel,
						True, #cfg.IS_ORDINAL_REG,
						no_age_transformer_init=False)


	#diff_pipeline_confusion_matrix_analysis(pipeline, data_loaders['val'], device, dataset_sizes['val'], is_ordinal_reg=True)
	
	if cfg.CHECK_DIFF_BASED:
		mae_on_test = evaluate_pipeline_clean(pipeline, data_loaders['val'], device, dataset_sizes['val'], bypass_diff=False, compare_post_round=False)
	else:
		mae_on_test = evaluate_pipeline_clean(pipeline, data_loaders['val'], device, dataset_sizes['val'], bypass_diff=True, compare_post_round=False)
		
