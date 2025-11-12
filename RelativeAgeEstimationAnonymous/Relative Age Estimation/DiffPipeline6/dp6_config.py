
ENABLE_DEBUG_BREAKPOINTS = False #True

USE_GPU = True #True
MULTI_GPU = True #True
RANDOM_SEED = 42
RANDOM_SPLIT = False
TEST_SIZE_FOR_RS = 0.2


DIFF_TRAIN_SIZE_RATIO = 0.75

TASK_TYPE = "range4to10" # "range4to10" / "range0to3" / "range0to10" / "rangeall"
SETTING_TYPE = "real" # "real"

MIN_AGE = 15
MAX_AGE = 80
AGE_INTERVAL = 1
AGE_RADIUS = 35                                 # radius is the topmost range the diff can be in

if TASK_TYPE == "range0to3":
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI = 3      # the uppermost of the range for the apref validation group 
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO = 0      # the lowermost of the range for the apref validation group
    AGE_DIFF_LEARN_RADIUS_HI = 3 #35 #10            # this is the uppermost value of the range the model learns
    AGE_DIFF_LEARN_RADIUS_LO = 0                    # DO NOT CHANGE !
elif TASK_TYPE == "range4to10":
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI = 10      # the uppermost of the range for the apref validation group 
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO = 4      # the lowermost of the range for the apref validation group
    AGE_DIFF_LEARN_RADIUS_HI = 10 #35 #10            # this is the uppermost value of the range the model learns
    AGE_DIFF_LEARN_RADIUS_LO = 4                    # DO NOT CHANGE !
elif TASK_TYPE == "range0to10":
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI = 10      # the uppermost of the range for the apref validation group 
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO = 0      # the lowermost of the range for the apref validation group
    AGE_DIFF_LEARN_RADIUS_HI = 10 #35 #10            # this is the uppermost value of the range the model learns
    AGE_DIFF_LEARN_RADIUS_LO = 0                    # DO NOT CHANGE !
elif TASK_TYPE == "rangeall":
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI = 35      # the uppermost of the range for the apref validation group 
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO = 0      # the lowermost of the range for the apref validation group
    AGE_DIFF_LEARN_RADIUS_HI = 35 #35 #10            # this is the uppermost value of the range the model learns
    AGE_DIFF_LEARN_RADIUS_LO = 0                    # DO NOT CHANGE !
else:
    exit()

BATCH_SIZE = 32
NUM_EPOCHS = 200 #60
NUM_ITERS = int(1.5e5)
LEARNING_RATE = 0.0003 #0.000003 #0.0003 #03

INITIAL_BACKBONE_FREEZING = True
UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH = True #False

DISABLE_WARMUP = False#False
WARMUP_EPOCHS = 4

LOAD_PRETRAINED_FULL = False
#FULL_PRETRAINED_WEIGHTS_PATH = "D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_24_04_2023_20_35_32/weights_271166_2.7249.pt"
FULL_PRETRAINED_WEIGHTS_PATH = 'D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_11_03_2023_15_25_17/weights_152100_4.5447.pt' # good_4_10: trained on 4-10 and achieved better MAE than original age predict on this range
#FULL_PRETRAINED_WEIGHTS_PATH = 'D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_14_05_2023_00_56_34/weights_29043_3.0017.pt'
#FULL_PRETRAINED_WEIGHTS_PATH = 'D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_07_05_2023_14_54_39/weights_131385_2.6990.pt' # trained from good_4_10 for 100 epochs with initial lr=0.0003 + warmpup and reduce_lr_on_platuae, with unfreeze of backbone until epoch=9
#FULL_PRETRAINED_WEIGHTS_PATH = 'D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_16_05_2023_09_20_47/weights_100959_2.7652.pt' # trained from good_4_10 for 100 epochs with initial lr=0.0003 + warmpup and reduce_lr_on_platuae, with unfreeze of backbone until epoch=16
#FULL_PRETRAINED_WEIGHTS_PATH = 'D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_22_05_2023_17_39_17/weights_139683_2.6962.pt' # trained from good_4_10 for 100 epochs with initial lr=0.0003 + warmpup and reduce_lr_on_platuae, with unfreeze of backbone until epoch=9. Ran for ~120 epochs. 
#FULL_PRETRAINED_WEIGHTS_PATH = 'D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_26_05_2023_12_01_02/weights_41490_2.6739.pt' # time_22_05_2023_17_39_17/weights_139683_2.6962.pt run for some more epochs
#FULL_PRETRAINED_WEIGHTS_PATH = 'D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_29_05_2023_09_03_02/weights_41490_2.6677.pt' # this one started from time_22_05_2023_17_39_17/weights_139683_2.6962.pt 

IMMEDIATE_UNFREEZE_ON_PRETRAINED = False #False
#FULL_PRETRAINED_WEIGHTS_PATH = "D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_28_04_2023_01_55_02/weights.pt"
#FULL_PRETRAINED_WEIGHTS_PATH = "D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_01_05_2023_12_03_20/weights_38738_2.6543.pt"
#FULL_PRETRAINED_WEIGHTS_PATH = "D:/projects/age_estimation/DiffPipeline6/weights/Morph2Diff/unified/iter/time_02_05_2023_15_49_30/weights_83010_2.6473.pt"

APPLY_AMP = True

SAVE_ALL_MODEL_METADATA = True        # if True saves all (weights, optimizer, ...). Else saves only weights
LOAD_FROM_ALL_MODEL_METADATA = False#False  # if True loads all (weights, optimizer, ...). Else loads only weights  
DONT_LOAD_OPTIMIZER = True#False

if SETTING_TYPE == "test":
    UNFREEZE_FEATURE_EXT_EPOCH = 0
    SMALL_DATA = True
    VALIDATION_PERIOD_ITERS = 5
elif SETTING_TYPE == "real":
    if LOAD_PRETRAINED_FULL:
        if IMMEDIATE_UNFREEZE_ON_PRETRAINED:
            UNFREEZE_FEATURE_EXT_EPOCH = 0
        else:
            UNFREEZE_FEATURE_EXT_EPOCH = 9 #16 #16
    else:
        UNFREEZE_FEATURE_EXT_EPOCH = 9 #16
    SMALL_DATA = False
    VALIDATION_PERIOD_ITERS = 3900 #600
else:
    exit()

VALIDATE_BETWEEN_EPOCHS = False
VALIDATE_AT_END_OF_EPOCH = True



NUM_OF_COPIES_AGE_PREDICT = 10
MID_FEATURE_SIZE_AGE_PREDICT = 1024

NUM_OF_WORKERS_DATALOADER = 16
if ENABLE_DEBUG_BREAKPOINTS:
    NUM_OF_WORKERS_DATALOADER = 0

NUM_OF_WORKERS_DATALOADER_VLD = 16
if ENABLE_DEBUG_BREAKPOINTS:
    NUM_OF_WORKERS_DATALOADER_VLD = 0


OPTIMIZER = "RangerLars" #"RangerLars" # "RangerLars" / "Adam"
SCHEDULER = "CosineAnnealingLR+GradualWarmupScheduler" #"CosineAnnealingLR+GradualWarmupScheduler" # "CosineAnnealingLR+GradualWarmupScheduler" # "NoScheduler" # "MultiStepLR"

SCHEDULER_STEP_GRANULARITY = "epoch" # "epoch" / "minibatch"
RESTART_SCHEDULER_AND_OPTIMIZER_ON_SMALL_LR = False #True

CHECK_DIFF_BASED = True #True

NUM_OF_REFS = 5

from dp6_model import DiffModelConfigType

DIFF_MODEL_CONFIG_TYPE = DiffModelConfigType.AddedEmbeddingAndMlpHeadWithDiffHead #AddedEmbeddingAndMlpHead

ADDED_EMBED_LAYER_SIZE = 256
FC_2ND_LAYER_SIZE = 128
AGG_TYPE = "maxpool" # "fc" / "maxpool" / "avgpool"

USE_EMBEDDINGS = False #True
EMBEDDINGS_KNN = 10
EMBEDDINGS_BASED_RATIO = 1.0 # 1.0 for avoiding non-embeddings in train set. In valid sets if USE_EMBEDDINGS == True --> just embedding based (q,r) pairs are used
EMBEDDINGS_FAR_KNN = False # This means KNN are not really KNN for train set, but with usually with some distance (i.e. D + 0, ... D + EMBEDDINGS_KNN nearest neighbours - see code)
NORMALIZE_EMBEDDINGS = True

# This flag means we simualte that we can tell when the original AgePredict model error is in the 
# range APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO and APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI when 
# evaluating
SIMULATE_ERROR_RANGE_IS_KNOWN = True

ENFORCE_CPU_ON_INFERENCE = False

# select according to used inference pipeline
APREF_VAL_SET_AGE_INFERENCE_SOURCE = "serial" # "serial" / "parallel"

PRODUCE_CONFUSION_MATRIX = False