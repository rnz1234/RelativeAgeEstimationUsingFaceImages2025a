
ENABLE_DEBUG_BREAKPOINTS = False #True

USE_GPU = True #True
MULTI_GPU = False #True
RANDOM_SEED = 42
RANDOM_SPLIT = False
TEST_SIZE_FOR_RS = 0.2


DIFF_TRAIN_SIZE_RATIO = 0.75

TASK_TYPE = "rangeall" # "range4to10"
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
elif TASK_TYPE == "rangeall":
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI = 35      # the uppermost of the range for the apref validation group 
    APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO = 0      # the lowermost of the range for the apref validation group
    AGE_DIFF_LEARN_RADIUS_HI = 35 #35 #10            # this is the uppermost value of the range the model learns
    AGE_DIFF_LEARN_RADIUS_LO = 0                    # DO NOT CHANGE !
else:
    exit()


AGE_DIFF_LO = 4
AGE_DIFF_HI = 10

BATCH_SIZE = 16 #32 #32
NUM_EPOCHS = 200 #60
NUM_ITERS = int(1.5e5)
LEARNING_RATE = 0.0001 #03

UNFREEZE_FEATURE_EXT_ON_RLVNT_EPOCH = True

if SETTING_TYPE == "test":
    UNFREEZE_FEATURE_EXT_EPOCH = 0
    SMALL_DATA = True
    VALIDATION_PERIOD_ITERS = 2
elif SETTING_TYPE == "real":
    UNFREEZE_FEATURE_EXT_EPOCH = 9 #15
    SMALL_DATA = False
    VALIDATION_PERIOD_ITERS = 3900 #600
else:
    exit()
VALIDATE_AT_END_OF_EPOCH = True

SAVE_ALL_MODEL_METADATA = True

NUM_OF_COPIES_AGE_PREDICT = 10
MID_FEATURE_SIZE_AGE_PREDICT = 1024

NUM_OF_WORKERS_DATALOADER = 16
if ENABLE_DEBUG_BREAKPOINTS:
    NUM_OF_WORKERS_DATALOADER = 0


OPTIMIZER = "RangerLars" #"RangerLars" # "RangerLars" / "Adam"
SCHEDULER = "CosineAnnealingLR+GradualWarmupScheduler" # "CosineAnnealingLR+GradualWarmupScheduler" # "NoScheduler" # "MultiStepLR"

CHECK_DIFF_BASED = True #True

NUM_OF_REFS = 5

from ep1_model import DiffModelConfigType

DIFF_MODEL_CONFIG_TYPE = DiffModelConfigType.AddedEmbeddingAndMlpHeadWithDiffHead #AddedEmbeddingAndMlpHead

ADDED_EMBED_LAYER_SIZE = 256
FC_2ND_LAYER_SIZE = 128

USE_EMBEDDINGS = True #True
EMBEDDINGS_KNN = 10
EMBEDDINGS_BASED_RATIO = 1.0 # 1.0 for avoiding non-embeddings in train set. In valid sets if USE_EMBEDDINGS == True --> just embedding based (q,r) pairs are used
EMBEDDINGS_FAR_KNN = False # This means KNN are not really KNN for train set, but with usually with some distance (i.e. D + 0, ... D + EMBEDDINGS_KNN nearest neighbours - see code)

# This flag means we simualte that we can tell when the original AgePredict model error is in the 
# range APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_LO and APREF_VLD_SET_AGE_DIFF_LEARN_RADIUS_HI when 
# evaluating
SIMULATE_ERROR_RANGE_IS_KNOWN = True

ENFORCE_CPU_ON_INFERENCE = False