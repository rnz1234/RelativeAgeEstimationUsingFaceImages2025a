##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline2
#	Date		:	28.10.2023
# 	Description	: 	Config file. Provided relevant configuration for the 
#					project.
##############################################################################

# Basic running/env settings
USE_GPU = True 
MULTI_GPU = True 
DISABLE_MULTIPROCESSING = False #False

# Advanced running/env settings
NUM_OF_WORKERS_DATALOADER = 16
if DISABLE_MULTIPROCESSING:
    NUM_OF_WORKERS_DATALOADER = 0

# randomization control
RANDOM_SEED = 42

# general dataset control
RANDOM_SPLIT = False
TEST_SIZE_FOR_RS = 0.2
DIFF_TRAIN_SIZE_RATIO = 0.75

# debug
SMALL_DATA = False #False

# domain related
MIN_AGE = 15
MAX_AGE = 80
AGE_INTERVAL = 1

# advanced dataset control
DATASET_SIZE_FACTOR = 1
NUM_REFERENCES = 10
EMBEDDINGS_KNN = 15
DISABLE_SAME_REF_BEING_QUERY = True
KNN_REDUCED_POOL_SIZE = 1024
SAMPLE_KNN_REDUCED_POOL = True
INPUT_ESTIMATION_FILE_NAME = "im2age_map_test.json" # im2age_map_test.json

# ML Infra control
BATCH_SIZE = 16
PRETRAINED_MODEL_PATH = '../Common/Weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64'
PRETRAINED_MODEL_FILE_NAME = 'weights.pt'
#PRETRAINED_MODEL_PATH = '../Common/Weights/Morph2/transformer/mae2.56'
DROPOUT_P = 0.9
UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH = True
UNFREEZE_FEATURE_EXT_EPOCH = 0 #15
LEARNING_RATE = 0.0003
NUM_ITERS = int(1.5e5)
NUM_EPOCHS = 200
NUM_OF_FC_LAYERS = 3
IS_ORDINAL = True #True
IS_MEAN_VAR_LOSS = False#True

# Model arch
REGRESSORS_DIFF_HEAD = False

# Checkpoints
SAVE_ALL_MODEL_METADATA = True
REMOVE_OLDER_CHECKPOINTS = True