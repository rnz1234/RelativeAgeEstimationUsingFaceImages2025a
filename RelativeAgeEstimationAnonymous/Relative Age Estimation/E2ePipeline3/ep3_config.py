##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline3
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
NUM_OF_WORKERS_DATALOADER_VAL = 16
if DISABLE_MULTIPROCESSING:
    NUM_OF_WORKERS_DATALOADER = 0
    NUM_OF_WORKERS_DATALOADER_VAL = 0

# randomization control
RANDOM_SEED = 42

# general dataset control
RANDOM_SPLIT = False
TEST_SIZE_FOR_RS = 0.2
DIFF_TRAIN_SIZE_RATIO = 0.75
DATASET_SELECT = "Morph2" # "CACD" / "Morph2"
MORPH2_DATASET_PATH = '../Common/Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5'
CACD_DATASET_PATH = '/home/eng/workspace/AgeEstimationMultiProject/Common/Datasets/CACD/se_data/se_dataset_cacd_resized.hdf5' # se_dataset_cacd

# debug
SMALL_DATA = False #False

# domain related
if DATASET_SELECT == "Morph2":
    MIN_AGE = 15
    MAX_AGE = 80
elif DATASET_SELECT == "CACD":
    MIN_AGE = 14
    MAX_AGE = 62
else:
    print("unsupported dataset")
    exit()

AGE_INTERVAL = 1

# pre-training dataset control
DIST_SET_SIZE_FACTOR = 0.1
APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL = False
APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL = True
INDEXES_SAVE_DIR = "train_set_split"

# advanced dataset control
DATASET_SIZE_FACTOR = 1
NUM_REFERENCES = 10
EMBEDDINGS_KNN = 30
if DATASET_SELECT == "Morph2":
    REF_SUBSET = False
    DISABLE_SAME_REF_BEING_QUERY = True
elif DATASET_SELECT == "CACD":
    REF_SUBSET = False
    DISABLE_SAME_REF_BEING_QUERY = True
else:
    print("unsupported dataset")
    exit()

USE_KNN = True
KNN_REDUCED_POOL_SIZE = 1024
SAMPLE_KNN_REDUCED_POOL_TRAIN = False
SAMPLE_KNN_REDUCED_POOL_TEST = False

if DATASET_SELECT == "Morph2":
    ######### MORPH 2 
    # good eval (78%,2%,20%) - kde_based_saturated - for cls ADR
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_cls_adr_fixed/using_F/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_cls_adr_fixed.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_cls_adr_fixed/using_F/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_cls_adr_fixed.json"

    ######### MORPH 2 - Efficient Net
    # good eval (78%,2%,20%) - kde_based_saturated - for reg+cls ADR
    # baseline
    INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed_efficientnetv2m/using_F/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_fixed_efficientnetv2m.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed_efficientnetv2m/using_F/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_fixed_efficientnetv2m.json"

    ######### MORPH 2 - THIS THE ONE REPORTED AS BEST IN THE PAPER
    # good eval (78%,2%,20%) - kde_based_saturated - for reg+cls ADR
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_fixed.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_fixed.json"
    # iter 1
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_train_Morph2_isol_10refs_iter1_good_eval_using_R_kde_sat_fixed.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_test_Morph2_isol_10refs_iter1_good_eval_using_R_kde_sat_fixed.json"
    
    # good eval (78%,2%,20%) - kde_based_saturated - for reg+cls ADR (bug)
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat.json"
    
    # good eval (78%,2%,20%) - kde_based_saturated - for cls ADR (bug)
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    # iter 0
    

    # good eval (78%,2%,20%) - kde_based_saturated - for reg+cls ADR
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat.json"
    

    # good eval (78%,2%,20%) - kde_based
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563.json"
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R.json"
    
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TEST = "im2age_map_test_Morph2.json" # im2age_map_test_Morph2_next2.json
    # 2refs - iter 0 output
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_2refs_iter0.json"

    # 10refs (new)
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_10refs_iter0.json"
    # iter 1
    # INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_10refs_iter1.json"
    # iter 2
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_10refs_iter2_lr0.0001.json"


    # 10refs (old)
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_run1.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_run2.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_run3.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_run4.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_run4_other_run.json"
elif DATASET_SELECT == "CACD":
    ######### CACD
    # good eval (78%,2%,20%)
    # baseline
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_CACD_good_eval_time_19_09_2024_01_21_19_weights_79500_5.2991.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_CACD_good_eval_time_19_09_2024_01_21_19_weights_79500_5.2991.json"
    # iter 0
    #INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_CACD/new/using_R/im2age_map_train_CACD_isol_10refs_iter0_good_eval_using_R.json"
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_CACD/new/using_R/im2age_map_test_CACD_isol_10refs_iter0_good_eval_using_R.json"

    # good eval (78%,2%,20%) - Using Efficient Net
    # baseline
    INPUT_ESTIMATION_FILE_NAME_TRAIN = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_train_CACD_good_eval_time_19_09_2024_01_21_19_weights_79500_5.2991.json"
    INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval/im2age_map_test_CACD_good_eval_time_19_09_2024_01_21_19_weights_79500_5.2991.json"
    # iter 0


    #INPUT_ESTIMATION_FILE_NAME_TEST = "im2age_map_test_CACD.json"

    # 10 refs (new) 
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_CACD/im2age_map_test_CACD_isol_go2_run1.json"

    # 10 refs (older)
    #INPUT_ESTIMATION_FILE_NAME_TEST = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_CACD/im2age_map_test_CACD_isol_run1.json"
else:
    print("unsupported dataset")
    exit()

DIST_APPROX_METHOD = "kde_based_saturated" #"uniform_based" #"kde_based" # "gaussian_based" "kde_based_saturated"
ERROR_SAT_RANGE_MIN = -20
ERROR_SAT_RANGE_MAX = 20

MODEL_PER_RANGE = False #True
RANGE_THRESHOLD = 45

USE_DIFF_CLS_AND_REG = False
LEARN_MINUS_DIFF_CLS = True

# ML Infra control
USE_VIT = False
USE_CONVNEXT = False
USE_EFFICIENTNET = True
USE_RESNET51Q = False
if DATASET_SELECT == "Morph2":
    if USE_VIT:
        BATCH_SIZE = 16
    elif USE_CONVNEXT:
        BATCH_SIZE = 16
    elif USE_EFFICIENTNET:
        BATCH_SIZE = 16
    elif USE_RESNET51Q:
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 32
    #PRETRAINED_MODEL_PATH = '../Common/Weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64'
    PRETRAINED_MODEL_PATH = '../Common/Weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64_sat_bb_time_02_01_2024_19_57_05'
    #PRETRAINED_MODEL_PATH = '../Common/Weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64_sat_bb_time_05_01_2024_17_52_42'
    PRETRAINED_MODEL_FILE_NAME = 'weights.pt'
    LOAD_PRETRAINED_RECOG = False
elif DATASET_SELECT == "CACD":
    if USE_VIT:
        BATCH_SIZE = 16
    elif USE_CONVNEXT:
        BATCH_SIZE = 16
    elif USE_EFFICIENTNET:
        BATCH_SIZE = 16
    elif USE_RESNET51Q:
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 32
    PRETRAINED_MODEL_PATH = ''
    PRETRAINED_MODEL_FILE_NAME = 'weights.pt'
    LOAD_PRETRAINED_RECOG = False
else:
    print("unsupported dataset")
    exit()

#PRETRAINED_MODEL_PATH = '../Common/Weights/Morph2/transformer/mae2.56'
DROPOUT_P = 0.7 #0.9
UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH = True
UNFREEZE_FEATURE_EXT_EPOCH = 0 #15
LEARNING_RATE = 0.0003 #0.0001 # 0.0003
NUM_ITERS = int(1.5e5)
NUM_EPOCHS = 500
NUM_OF_FC_LAYERS = 3
if USE_VIT:
    FC_HEAD_BASE_LAYER_SIZE = 1024
elif USE_CONVNEXT:
    FC_HEAD_BASE_LAYER_SIZE = 1024
elif USE_EFFICIENTNET:
    FC_HEAD_BASE_LAYER_SIZE = 1024
elif USE_RESNET51Q:
    FC_HEAD_BASE_LAYER_SIZE = 2048
else:
    FC_HEAD_BASE_LAYER_SIZE = 2048
IS_ORDINAL = True #True
#IS_MEAN_VAR_LOSS = False#True
APPLY_WEIGHT_DECAY = False
WEIGHT_DECAY_VAL = 0.001
USE_GENDER = False
GENDER_FACTOR = 0
if DATASET_SELECT == "Morph2":
    IS_GENDER_IS_DATASET = True
elif DATASET_SELECT == "CACD":
    IS_GENDER_IS_DATASET = False
else:
    print("unsupported dataset")
    exit()


# Full override for pretraining the model
FULL_MODEL_PRETRAINED_WEIGHTS = False
FULL_MODEL_PRETRAINED_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_08_01_2024_22_18_11/weights_59_2.5057.pt"

# Model arch
REGRESSORS_DIFF_HEAD = False

# Checkpoints
SAVE_ALL_MODEL_METADATA = True
REMOVE_OLDER_CHECKPOINTS = True

# Inference control
if DATASET_SELECT == "Morph2":
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_02_02_2024_18_18_08/weights_43_2.5056.pt"
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_03_02_2024_21_21_48/weights_36_2.5013.pt"
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_05_02_2024_21_55_19/weights_31_2.5005.pt"
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_06_02_2024_16_38_29/weights_0_2.4971.pt"
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_05_03_2024_23_57_04/weights_188_2.4909.pt"
    
    # I.I. - refs=2
    # iter 0 
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_06_06_2024_09_49_03/weights_154_2.4924.pt"
    
    # I.I. - refs=10
    # iter 0 
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_31_05_2024_11_18_39/weights_163_2.4913.pt"
    # further iterations (lr=0.0003)
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_21_06_2024_18_41_42/weights_0_2.4874.pt"
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_26_06_2024_00_08_10/weights_1_2.4851.pt"
    # iter 1 (lr=0.0001) - done after iter 0
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_26_06_2024_23_57_02/weights_1_2.4835.pt"

    # I.I - good eval (78%,2%,20%)
    # iter 0 (Diff model result after refining Absolute model)
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_12_09_2024_12_58_15/weights_18_2.5001.pt"
    # iter 1
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_15_09_2024_20_30_35/weights_34_2.4790.pt"

    # I.I - good eval (78%,2%,20%) - kde_based_saturated - reg+cls adr
    # iter 0 (Diff model result after refining Absolute model)
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_24_01_2025_11_34_56/weights_43_2.4968.pt"
    # iter 1 
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_28_01_2025_14_47_10/weights_47_2.4739.pt"
    
    # I.I - good eval (78%,2%,20%) - kde_based_saturated - cls adr
    # iter 0 (Diff model result after refining Absolute model)
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_26_01_2025_00_38_59/weights_63_2.4932.pt"

    # I.I - good eval (78%,2%,20%) - kde_based_saturated - reg+cls adr - with EfficientNet v2 m
    # iter 0 (Diff model result after refining Absolute model)
    INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/Morph2Diff/unified/iter/time_28_06_2025_19_11_38/weights_26_2.4519.pt"
    

elif DATASET_SELECT == "CACD":
    # I.I - ref=10 (go 1)
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/CACDDiff/unified/iter/time_02_05_2024_19_15_35/weights_28_5.6322.pt"
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/CACDDiff/unified/iter/time_13_05_2024_17_08_37/weights_22_5.3275.pt"

    # I.I - refs=10 (go 2)
    # iter 0
    #INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/CACDDiff/unified/iter/time_29_06_2024_23_18_57/weights_46_5.6355.pt"

    # I.I - good eval (78%,2%,20%)
    # iter 0 (Diff model result after refining Absolute model)
    INFERENCE_MODEL_WEIGHTS_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/weights/CACDDiff/unified/iter/time_21_09_2024_12_07_43/weights_4_5.2594.pt"
else:
    print("unsupported dataset")
    exit()
    
INFERENCE_BASED_ON_F = True
if DATASET_SELECT == "Morph2":
    #INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_run4.json"
    #INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_2refs_iter0.json"
    #INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_10refs_iter2.json"
    #INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_10refs_iter3.json"

    # kde_based
    # if INFERENCE_BASED_ON_F:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new/using_F/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_F.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new/using_F/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_F.json"
    # else:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R.json"

    ####################################################################
    #                       kde_based_saturated
    # kde_based_saturated - reg+cls adr - iter 0
    # if INFERENCE_BASED_ON_F:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_F/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_fixed.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_F/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_fixed.json"
    # else:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_fixed.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_fixed.json"

    # # kde_based_saturated - reg+cls adr - iter 1
    # if INFERENCE_BASED_ON_F:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_F/im2age_map_test_Morph2_isol_10refs_iter1_good_eval_using_F_kde_sat_fixed.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_F/im2age_map_train_Morph2_isol_10refs_iter1_good_eval_using_F_kde_sat_fixed.json"
    # else:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_test_Morph2_isol_10refs_iter1_good_eval_using_R_kde_sat_fixed.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed/using_R/im2age_map_train_Morph2_isol_10refs_iter1_good_eval_using_R_kde_sat_fixed.json"

    ####################################################################

    # kde_based_saturated - reg+cls adr - efficient net - iter 0
    if INFERENCE_BASED_ON_F:
        INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed_efficientnetv2m/using_F/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_fixed_efficientnetv2m.json"
        INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed_efficientnetv2m/using_F/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_fixed_efficientnetv2m.json"
    else:
        INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed_efficientnetv2m/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_fixed_efficientnetv2m.json"
        INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_fixed_efficientnetv2m/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_fixed_efficientnetv2m.json"




    # kde_based_saturated - cls adr
    # if INFERENCE_BASED_ON_F:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_cls_adr_fixed/using_F/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_cls_adr_fixed.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_cls_adr_fixed/using_F/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_F_kde_sat_cls_adr_fixed.json"
    # else:
    #     INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_cls_adr_fixed/using_R/im2age_map_test_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_cls_adr_fixed.json"
    #     INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/new_kde_sat_cls_adr_fixed/using_R/im2age_map_train_Morph2_isol_10refs_iter0_good_eval_using_R_kde_sat_cls_adr_fixed.json"


    #INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/im2age_map_test_Morph2_isol_10refs_iter1_good_eval.json"
    #INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_Morph2/im2age_map_train_Morph2_isol_10refs_iter1_good_eval.json"
elif DATASET_SELECT == "CACD":
    #INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_CACD/im2age_map_test_CACD_isol_run2.json"
    #INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/isol_iterative_improvement_results_CACD/im2age_map_test_CACD_isol_go2_run1.json"
    if INFERENCE_BASED_ON_F:
        INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_CACD/new/using_F/im2age_map_test_CACD_isol_10refs_iter0_good_eval_using_F.json"
        INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_CACD/new/using_F/im2age_map_train_CACD_isol_10refs_iter0_good_eval_using_F.json"
    else:
        INFERENCE_MODEL_RESULTS_TEST_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_CACD/new/using_R/im2age_map_test_CACD_isol_10refs_iter0_good_eval_using_R.json"
        INFERENCE_MODEL_RESULTS_TRAIN_BASE_SAVE_PATH = "/home/eng/workspace/AgeEstimationMultiProject/E2ePipeline3/good_eval_isol_iterative_improvement_results_CACD/new/using_R/im2age_map_train_CACD_isol_10refs_iter0_good_eval_using_R.json"
else:
    print("unsupported dataset")
    exit()


RUN_PROFILER = True