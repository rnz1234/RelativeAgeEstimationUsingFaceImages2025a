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

from Common.Optimizers.RangerLars import RangerLars
from Common.Schedulers.GradualWarmupScheduler import GradualWarmupScheduler

import diff_pipeline1__config as cfg

from diff_pipeline1__dataset import DiffPipeline1Dataset
from Common.Datasets.Morph2.data_parser import DataParser
from diff_pipeline1__train import train_diff_cls_model_iter


from diff_pipeline1__model import DiffPipeline1Model

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

    with open("im2age_map_test.json", 'r') as im2age_map_test_f:
        im2age_map_test = json.load(im2age_map_test_f)

    # Load data
    data_parser = DataParser('../Common/Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5', small_data=cfg.SMALL_DATA)
    data_parser.initialize_data()


    x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test,
    if cfg.RANDOM_SPLIT:
        all_images = np.concatenate((x_train, x_test), axis=0)
        all_labels = np.concatenate((y_train, y_test), axis=0)

        x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=cfg.TEST_SIZE_FOR_RS, random_state=cfg.RANDOM_SEED)

    # import pdb
    # pdb.set_trace()

    # shuffle basic aligned test
    # x_test_shuffled
    # y_test_shuffled

    idx_test = range(len(y_test))

    x_test_shuffled, y_test_shuffled, idx_test_shuffled = utils.shuffle(x_test, y_test, idx_test)
    


    train_ds = DiffPipeline1Dataset(
        dataset_type="train",
        batrn_set_images=x_train,
        batrn_set_metadata=y_train,
        batst_set_images=x_test_shuffled,
        batst_set_metadata=y_test_shuffled,
        batst_set_idx=idx_test_shuffled,
        train_test_ratio=cfg.DIFF_TRAIN_SIZE_RATIO,
        im2age_map_batst=im2age_map_test,
        min_age=cfg.MIN_AGE,
        age_interval=cfg.AGE_INTERVAL,
        max_age=cfg.MAX_AGE,
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
        copies=1, 
        age_radius=cfg.AGE_RADIUS,
        dataset_size_factor=cfg.DATASET_SIZE_FACTOR
    )

    print("training set size: " + str(len(train_ds)))

    test_ds = DiffPipeline1Dataset(
        dataset_type="test",
		batrn_set_images=x_train,
		batrn_set_metadata=y_train,
        batst_set_images=x_test_shuffled,
        batst_set_metadata=y_test_shuffled,
        train_test_ratio=cfg.DIFF_TRAIN_SIZE_RATIO,
        im2age_map_batst=im2age_map_test,
		min_age=cfg.MIN_AGE,
		age_interval=cfg.AGE_INTERVAL,
        max_age=cfg.MAX_AGE,
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
        copies=1, 
        age_radius=cfg.AGE_RADIUS
	)

    print("testing set size: " + str(len(test_ds)))


    image_datasets = {
		'train': train_ds,
		'val': test_ds
	}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
   
    data_loaders = {
		'train': DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
		'val': DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=True)
	}


    #####################################################
    #           Model
    #####################################################


    # create model and parameters
    model = DiffPipeline1Model(age_interval=cfg.AGE_INTERVAL, 
                            min_age=cfg.MIN_AGE, 
                            max_age=cfg.MAX_AGE, 
                            age_radius=cfg.AGE_RADIUS, 
                            device=device)

    if cfg.UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH:
        model.freeze_base_cnn(True)

    if cfg.USE_GPU and cfg.MULTI_GPU:
        if torch.cuda.device_count() > 1:
            print("Using multiple GPUs (" + str(torch.cuda.device_count()) + ")")
            model = torch.nn.DataParallel(model)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    if cfg.OPTIMIZER == "RangerLars":
        optimizer = RangerLars(model.parameters(), lr=cfg.LEARNING_RATE)
    else:
        optimizer = Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    if cfg.SCHEDULER == "CosineAnnealingLR+GradualWarmupScheduler":
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
    else:
        scheduler = None


    #####################################################
    #           Logging
    #####################################################
    

     ### Train ###
    
    cur_time = datetime.now()
    cur_time_str = cur_time.strftime("time_%d_%m_%Y_%H_%M_%S")
    experiment_name = cur_time_str
    
    writer = SummaryWriter('logs/Morph2Diff/unified/iter/' + experiment_name) 

    model_path = 'weights/Morph2Diff/unified/iter/' + experiment_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    shutil.copyfile("diff_pipeline1__config.py", model_path + "/config.py")
    

    #####################################################
    #           Training
    #####################################################

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