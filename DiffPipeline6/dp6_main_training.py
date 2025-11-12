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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split

from datetime import datetime

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Common.Schedulers.GradualWarmupScheduler2 import GradualWarmupScheduler2
from Common.Schedulers.ReduceLROnPlateauEnhanced import ReduceLROnPlateauEnhanced


import dp6_config as cfg

from dp6_dataset import AgeDiffSameUniformDiffDataset, AgeDiffMixedUniformDiffDataset, AgeDiffMimicDiffDataset, get_error_constrained_dataset
from Common.Datasets.Morph2.data_parser import DataParser
from dp6_train import train_diff_cls_model_iter


from dp6_model import AgeDiffModel, DiffModelConfigType

from tqdm import tqdm

from sklearn import utils

if __name__ == "__main__":

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

	if cfg.APREF_VAL_SET_AGE_INFERENCE_SOURCE == "serial":
		with open("im2age_map_test_from_ser_e2e_infer.json", 'r') as im2age_map_test_f:
			im2age_map_test = json.load(im2age_map_test_f)
	elif cfg.APREF_VAL_SET_AGE_INFERENCE_SOURCE == "parallel":
		with open("im2age_map_test.json", 'r') as im2age_map_test_f:
			im2age_map_test = json.load(im2age_map_test_f)
	else:
		print("incorrect APREF_VAL_SET_AGE_INFERENCE_SOURCE configuration")
		print("ABORTING...")
		exit()
			
	# Load data
	data_parser = DataParser('../Common/Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5', small_data=cfg.SMALL_DATA)
	data_parser.initialize_data()


	x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test,
	if cfg.RANDOM_SPLIT:
		all_images = np.concatenate((x_train, x_test), axis=0)
		all_labels = np.concatenate((y_train, y_test), axis=0)

		x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=cfg.TEST_SIZE_FOR_RS, random_state=cfg.RANDOM_SEED)


	#face2emb_arr_trn_r = np.load('face2emb_arr_trn.npy', allow_pickle=True)
	face2emb_arr_trn_r = np.load('face2emb_arr_trn_recog.npy', allow_pickle=True)

	#     import pdb
	#     pdb.set_trace()
	#face2emb_arr_trn_r = face2emb_arr_trn_r.item()

	#face2emb_arr_vld_r = np.load('face2emb_arr_vld.npy', allow_pickle=True)
	face2emb_arr_vld_r = np.load('face2emb_arr_vld_recog.npy', allow_pickle=True)
	#face2emb_arr_vld_r = face2emb_arr_vld_r.item()

	# shuffle basic aligned test
	# x_test_shuffled
	# y_test_shuffled


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
				interpolation=Image.BICUBIC
			)], p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.RandomErasing(p=0.5)
		])

	# transf = transforms.Compose([
	#         transforms.Resize(224), # just for testing
	# #          transforms.RandomResizedCrop(224, (0.9, 1.0)),
	# #          transforms.RandomHorizontalFlip(),
	#         transforms.RandomApply([transforms.ColorJitter(
	#             brightness=0.1,
	#             contrast=0.1,
	#             saturation=0.1,
	#             hue=0.1
	#         )], p=0.5),
	# #         transforms.RandomApply([transforms.RandomAffine(
	# #             degrees=10,
	# #             translate=(0.1, 0.1),
	# #             scale=(0.9, 1.1),
	# #             shear=5,
	# #             resample=Image.BICUBIC
	# #         )], p=0.5),
	#         transforms.ToTensor(),
	#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	# #        transforms.RandomErasing(p=0.5)
	#     ])


	train_ds = AgeDiffSameUniformDiffDataset(
		data_set_images=x_train,
		data_set_metadata=y_train,
		min_age=cfg.MIN_AGE,
		age_interval=cfg.AGE_INTERVAL,
		max_age=cfg.MAX_AGE,
		transform=transf,
		copies=1, 
		age_diff_learn_radius_lo=cfg.AGE_DIFF_LEARN_RADIUS_LO,
		age_diff_learn_radius_hi=cfg.AGE_DIFF_LEARN_RADIUS_HI,
		embs=face2emb_arr_trn_r,
		num_references=cfg.NUM_OF_REFS,
		use_embs=cfg.USE_EMBEDDINGS,
		embs_knn=cfg.EMBEDDINGS_KNN,
		embs_far_knn=cfg.EMBEDDINGS_FAR_KNN,
		embeddings_based_ratio=cfg.EMBEDDINGS_BASED_RATIO,
		embs_normalize=cfg.NORMALIZE_EMBEDDINGS,
		dataset_type="train"
	)

	print("Training (q trn, r trn) set size: " + str(len(train_ds)))

	test_qtst_rtst_ds = AgeDiffSameUniformDiffDataset(
		data_set_images=x_test,
		data_set_metadata=y_test,
		min_age=cfg.MIN_AGE,
		age_interval=cfg.AGE_INTERVAL,
		max_age=cfg.MAX_AGE,
		transform=transf,
		copies=1, 
		age_diff_learn_radius_lo=cfg.AGE_DIFF_LEARN_RADIUS_LO,
		age_diff_learn_radius_hi=cfg.AGE_DIFF_LEARN_RADIUS_HI,
		embs=face2emb_arr_vld_r,
		num_references=cfg.NUM_OF_REFS,
		use_embs=cfg.USE_EMBEDDINGS,
		embs_knn=cfg.EMBEDDINGS_KNN,
		embeddings_based_ratio=cfg.EMBEDDINGS_BASED_RATIO,
		embs_normalize=cfg.NORMALIZE_EMBEDDINGS,
		dataset_type="valid"
	)

	print("Testing (q tst, r tst) set size: " + str(len(test_qtst_rtst_ds)))

	test_qtst_rtrn_ds = AgeDiffMixedUniformDiffDataset(
		batrn_set_images=x_train,
		batrn_set_metadata=y_train,
		batst_set_images=x_test,
		batst_set_metadata=y_test,
		min_age=cfg.MIN_AGE,
		age_interval=cfg.AGE_INTERVAL,
		max_age=cfg.MAX_AGE,
		transform=transf,
		copies=1, 
		age_diff_learn_radius_lo=cfg.AGE_DIFF_LEARN_RADIUS_LO,
		age_diff_learn_radius_hi=cfg.AGE_DIFF_LEARN_RADIUS_HI,
		embs_trn=face2emb_arr_trn_r,
		embs_vld=face2emb_arr_vld_r,
		num_references=cfg.NUM_OF_REFS,
		use_embs=cfg.USE_EMBEDDINGS,
		embs_knn=cfg.EMBEDDINGS_KNN,
		embs_normalize=cfg.NORMALIZE_EMBEDDINGS,
	)

	print("Testing (q tst, r trn) set size: " + str(len(test_qtst_rtrn_ds)))

	x_test_filtered, y_test_filtered, batst_set_filtered_indexes = get_error_constrained_dataset(orig_dataset_images=x_test, 
																									orig_dataset_metadata=y_test,
																									age_diff_learn_radius_lo=cfg.APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO,
																									age_diff_learn_radius_hi=cfg.APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI,#cfg.AGE_DIFF_LEARN_RADIUS_HI,
																									im2age_map_batst=im2age_map_test)

	# apref = Age Predict Reference (dataset is based on (q,r) pairs where r's age is what AgePredict model returns on q)
	test_apref_ds = AgeDiffMimicDiffDataset(
		batrn_set_images=x_train,
		batrn_set_metadata=y_train,
		batst_set_images=x_test_filtered,
		batst_set_metadata=y_test_filtered,
		batst_set_indexes=batst_set_filtered_indexes,
		im2age_map_batst=im2age_map_test,
		min_age=cfg.MIN_AGE,
		age_interval=cfg.AGE_INTERVAL,
		max_age=cfg.MAX_AGE,
		transform=transf,
		copies=1, 
		age_radius=cfg.APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI, # this should be the radius of actual learning in order to give correct labels. Not the absolute AGE_RADIUS
		embs_trn=face2emb_arr_trn_r,
		embs_vld=face2emb_arr_vld_r,
		num_references=cfg.NUM_OF_REFS,
		use_embs=cfg.USE_EMBEDDINGS,
		embs_knn=cfg.EMBEDDINGS_KNN,
		embs_normalize=cfg.NORMALIZE_EMBEDDINGS,
	)

	print("Testing (q tst where AgePredict(q)  {age_diff_learn_radius_lo} <= error <= {age_diff_learn_radius_hi}, r trn where age(r)=AgePredict(q)) set size: ".format(age_diff_learn_radius_lo=cfg.AGE_DIFF_LEARN_RADIUS_LO, age_diff_learn_radius_hi=cfg.AGE_DIFF_LEARN_RADIUS_HI) + str(len(test_apref_ds)))

	x_test_all, y_test_all, batst_set_all_indexes = get_error_constrained_dataset(orig_dataset_images=x_test, 
																									orig_dataset_metadata=y_test,
																									age_diff_learn_radius_lo=0,
																									age_diff_learn_radius_hi=35,#cfg.AGE_RADIUS,
																									im2age_map_batst=im2age_map_test)


	# apref = Age Predict Reference (dataset is based on (q,r) pairs where r's age is what AgePredict model returns on q)
	# test_apref_all_ds = DiffPipeline4MimicDiffDataset(
	# 	batrn_set_images=x_train,
	# 	batrn_set_metadata=y_train,
	#     batst_set_images=x_test_all,
	#     batst_set_metadata=y_test_all,
	#     batst_set_indexes=batst_set_all_indexes,
	#     im2age_map_batst=im2age_map_test,
	# 	min_age=cfg.MIN_AGE,
	# 	age_interval=cfg.AGE_INTERVAL,
	#     max_age=cfg.MAX_AGE,
	# 	transform=transf,
	#     copies=1, 
	#     age_radius=cfg.AGE_RADIUS,
	#     embs_trn=face2emb_arr_trn_r,
	#     embs_vld=face2emb_arr_vld_r#,
	#     #num_references=cfg.NUM_OF_REFS
	# )

	# print("Testing (q tst where AgePredict(q)  {age_diff_learn_radius_lo} <= error <= {age_diff_learn_radius_hi}, r trn where age(r)=AgePredict(q)) set size: ".format(age_diff_learn_radius_lo=cfg.AGE_DIFF_LEARN_RADIUS_LO, age_diff_learn_radius_hi=35) + str(len(test_apref_all_ds))) #cfg.AGE_RADIUS





	image_datasets = {
		'train': train_ds,
		'val_qtst_rtst': test_qtst_rtst_ds,
		'val_qtst_rtrn': test_qtst_rtrn_ds,
		'val_apref_ds': test_apref_ds,
		#'val_apref_all_ds': test_apref_all_ds
	}

	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val_qtst_rtst', 'val_qtst_rtrn', 'val_apref_ds']}#, 'val_apref_all_ds']}

	data_loaders = {
		'train': DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
		'val_qtst_rtst': DataLoader(test_qtst_rtst_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER_VLD, pin_memory=True, shuffle=False, drop_last=True),
		'val_qtst_rtrn': DataLoader(test_qtst_rtrn_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER_VLD, pin_memory=True, shuffle=False, drop_last=True),
		'val_apref_ds': DataLoader(test_apref_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER_VLD, pin_memory=True, shuffle=False, drop_last=True),
		#'val_apref_all_ds' : DataLoader(test_apref_all_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=True)
	}

	#####################################################
	#           Model
	#####################################################

	model = AgeDiffModel(age_interval=cfg.AGE_INTERVAL, 
								min_age=cfg.MIN_AGE, 
								max_age=cfg.MAX_AGE, 
								age_diff_learn_radius_hi=cfg.AGE_DIFF_LEARN_RADIUS_HI, 
								device=device, 
								deep=True, 
								num_references=cfg.NUM_OF_REFS,#cfg.NUM_OF_REFERENCES, 
								config_type=DiffModelConfigType.AddedEmbeddingAndMlpHeadWithDiffHead, 
								added_embed_layer_size=cfg.ADDED_EMBED_LAYER_SIZE, 
								diff_embed_layer_size=cfg.ADDED_EMBED_LAYER_SIZE,
								fc_2nd_layer_size=cfg.FC_2ND_LAYER_SIZE,
								agg_type=cfg.AGG_TYPE,
								is_ordinal_reg=True)


	if cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH or cfg.INITIAL_BACKBONE_FREEZING:
		print("backbone initially frozen")
		model.freeze_base_cnn(True)

	if cfg.USE_GPU and cfg.MULTI_GPU:
		if torch.cuda.device_count() > 1:
			print("Using multiple GPUs (" + str(torch.cuda.device_count()) + ")")
			model = torch.nn.DataParallel(model)

	if cfg.OPTIMIZER == "RangerLars":
		optimizer = RangerLars(model.parameters(), lr=cfg.LEARNING_RATE)
		print("optimizer: using RangerLars")
	else:
		optimizer = Adam(model.parameters(), lr=cfg.LEARNING_RATE)
		print("optimizer: using Adam")

	pretrained_optimizer = False
	if cfg.LOAD_PRETRAINED_FULL:
		if cfg.LOAD_FROM_ALL_MODEL_METADATA:
			checkpoint = torch.load(cfg.FULL_PRETRAINED_WEIGHTS_PATH, map_location=lambda storage, loc: storage)#, map_location=device)
			model.load_state_dict(checkpoint['model_state_dict'])
			if not cfg.DONT_LOAD_OPTIMIZER:
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				print("loaded pretrained model - all")
				pretrained_optimizer = True
		else:
			model.load_state_dict(torch.load(cfg.FULL_PRETRAINED_WEIGHTS_PATH))#, strict=False)
			print("loaded pretrained model - only weights")
	model.to(device)

	criterion = torch.nn.CrossEntropyLoss().to(device)

	

	if cfg.SCHEDULER == "CosineAnnealingLR+GradualWarmupScheduler":
		cosine_scheduler = CosineAnnealingLR(
			optimizer,
			T_max=cfg.NUM_ITERS
		)
		scheduler = GradualWarmupScheduler(
			optimizer,
			multiplier=1,
			total_epoch=cfg.WARMUP_EPOCHS,#10000,
			after_scheduler=cosine_scheduler
		)
		print("scheduler: using CosineAnnealingLR+GradualWarmupScheduler")
	elif cfg.SCHEDULER == "ReduceLROnPlateau+GradualWarmupScheduler":
		reduce_lr_on_plateau_scheduler = ReduceLROnPlateauEnhanced(optimizer, 'min')
		# import pdb
		# pdb.set_trace()
		if cfg.DISABLE_WARMUP:
			scheduler = reduce_lr_on_plateau_scheduler
			print("scheduler: using ReduceLROnPlateau")
		else:
			print(f"initial lr is = {optimizer.param_groups[0]['lr']}")
			scheduler = GradualWarmupScheduler2(
				optimizer,
				multiplier=1,
				total_epoch=cfg.WARMUP_EPOCHS,
				after_scheduler=reduce_lr_on_plateau_scheduler,
				pretrained=pretrained_optimizer,
			)
			print("scheduler: using ReduceLROnPlateau+GradualWarmup2Scheduler")
	elif cfg.SCHEDULER == "MultiStepLR":
		#import torch.optim.lr_scheduler.StepLR
		#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
		from torch.optim.lr_scheduler import MultiStepLR
		scheduler = MultiStepLR(optimizer, milestones=[cfg.UNFREEZE_FEATURE_EXT_EPOCH], gamma=0.1)
		print("scheduler: using MultiStepLR")
	else:
		scheduler = None


	# #           Logging
	#####################################################


	### Train ###

	cur_time = datetime.now()
	cur_time_str = cur_time.strftime("time_%d_%m_%Y_%H_%M_%S")
	experiment_name = cur_time_str

	writer = SummaryWriter('logs/Morph2Diff/unified/iter/' + experiment_name) 

	model_path = 'weights/Morph2Diff/unified/iter/' + experiment_name
	if not os.path.exists(model_path):
		os.makedirs(model_path)


	shutil.copyfile("dp6_config.py", model_path + "/dp6_config.py")
	shutil.copyfile("dp6_dataset.py", model_path + "/dp6_dataset.py")
	shutil.copyfile("dp6_model.py", model_path + "/dp6_model.py")
	shutil.copyfile("dp6_train.py", model_path + "/dp6_train.py")
	shutil.copyfile("dp6_main_training.ipynb", model_path + "/dp6_main_training.ipynb")
	shutil.copyfile("dp6_main_diff_inference.ipynb", model_path + "/dp6_main_diff_inference.ipynb")
	shutil.copyfile("dp6_main_e2e_inference.ipynb", model_path + "/dp6_main_e2e_inference.ipynb")


	#####################################################
	#           Training
	#####################################################

	if cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH:
		print(f"going to unfreeze backbone on epoch {cfg.UNFREEZE_FEATURE_EXT_EPOCH}")
	else:
		print(f"not going to unfreeze backbone")

	best_diff_model = train_diff_cls_model_iter(
		model,
		criterion,
		optimizer,
		scheduler,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		model_path,
		num_epochs=cfg.NUM_EPOCHS,
		unfreeze_feature_ext_epoch=cfg.UNFREEZE_FEATURE_EXT_EPOCH,
		unfreeze_feature_ext_on_rlvnt_epoch=cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH,
		validate_at_k=cfg.VALIDATION_PERIOD_ITERS,
		validate_at_end_of_epoch=cfg.VALIDATE_AT_END_OF_EPOCH
	)


	print('saving best model')

	FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
	torch.save(best_diff_model.state_dict(), FINAL_MODEL_FILE)