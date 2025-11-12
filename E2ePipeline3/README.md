# Ep3 Pipeline - Formal Differential Age Estimation Solution

This project implements the age estimation scheme described in the paper.

Current best result: MAE=2.45 for Morph2

## Sources

ep3_config.py - configuration settings
ep3_dataset.py - dataset definition ([Q,R[]] collections)
ep3_model.py - deep learning model definition
ep3_train.py - training module (train and evaluate methods)
ep3_main_training.py - main file to run model training
ep3_infer.py - main file to run model inference and produce results into json file

For Train split mode (using an SE part of test set only for BAR error distribution estimation - USED IN PAPER AND REPORTED EVALUATION):
ep3_dist_subset_good_eval.ipynb

For Test split mode (using an SE part of test set only for BAR error distribution estimation - NOT USED IN PAPER AND REPORTED EVALUATION):
ep3_advanced_stats.ipynb
ep3_test_set_split.py

Early stage work - already converged to more robust code (some moved to Common) or not used:
ep3_dist_subset.ipynb
ep3_infer_iterated.py
ep3_advanced_stats_iter_infer.ipynb




## Usage

### Intro: Explanation On Inputs 

Tl;dr: if you already have the background and just want to run, jump to next section "Setup"

Note: dataset_name can be "Morph2" or "CACD"

1. Good Evaluation Setting: Train set split SE to dist and actual training set.
This is the setting used for the paper. It uses the same test set as the BAR for accurate model results comparison. The SE split also used for BAR training, and is done in another repo.

In this case, the algorithm is fed from:
- data & metadata:
	- Data is fed using the DataParser class (Morph 2) / CacdDataParser class (CACD)  
	- The estimation file paths of the BAR model are fed via config.py/INPUT_ESTIMATION_FILE_NAME_TRAIN and INPUT_ESTIMATION_FILE_NAME_TEST constants.
	(good_eval/{dataset_name}im2age_map_test*.json - original DeepAge2 results of different runs.
	good_eval_isol_iterative_improvement_results_Morph2/isol_iterative_improvement_results_{dataset_name} - results of using our model as BAR - iterative improvement approach)
	- In order to do the BAR's distribution estimation we have SE division of the MORPHII/CACD train sets. The index for the distribution estimation indexes are in {dataset_name}_dist_indexes.pkl while the indexes for isolated training set are in {dataset_name}_isolated_train_indexed.pkl.
	- embeddings: {dataset_name}_face2emb_arr_trn_recog.npy and {dataset_name}_face2emb_arr_tst_recog.npy
- various configuration (in config.py)
	- The APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL should be True to enforce the usage of 
	  the dist indexes and isolated train set. 
	- APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL should be False


2. Vanilla Setting [Not Used For Paper!]: Test set split SE to dist and actual training set.
This setting is not used for the paper, because here the split is done in the test set: dist and actual test set (SE). This is OK from validity perspective but for accurate evaluation and academic reporting - we use [1]. 

In this case, the algorithm is fed from:
- data & metadata:
	- Data is fed using the DataParser class (Morph 2) / CacdDataParser class (CACD)  
	- The estimation file path of the BAR model are fed via config.py/INPUT_ESTIMATION_FILE_NAME_TEST constant.
	({dataset_name}im2age_map_test.json - original DeepAge2 results. 
	isol_iterative_improvement_results_{dataset_name} - results of using our model as BAR - iterative improvement approach)
	- In order to do the BAR's distribution estimation we have SE division of the MORPHII/CACD test sets. The index for the distribution estimation indexes are in {dataset_name}_dist_indexes.pkl while the indexes for isolated testing are in {dataset_name}_isolated_test_indexed.pkl.
	- embeddings: {dataset_name}_face2emb_arr_trn_recog.npy and {dataset_name}_face2emb_arr_tst_recog.npy
- various configuration (in config.py)
	- The APPLY_TEST_SET_SPLIT_FOR_DIST_AND_ISOL if True enforce the usage of 
	  the dist indexes and isolated test set. 
	- APPLY_TRAIN_SET_SPLIT_FOR_DIST_AND_ISOL should be False


### Setup

Install all dependencies using requirements.txt
The project flows below run in python, based on Pytorch. The HW for the project was NVIDIA V100

### Training

For running training:
1. In config.py:
- set correctly DATASET_SELECT (CACD / Morph2)
- set the backbone correctly (e.g. USE_EFFICIENTNET)
- Set the correct estimation file path (original BAR or iterative improvement) - 
INPUT_ESTIMATION_FILE_NAME_TRAIN and INPUT_ESTIMATION_FILE_NAME_TEST constants.
2. Run 
```bash
python ep3_main_training.py
``` 
Or can run in background using 
```bash
nohup python ep3_main_training.py &
```
3. For reviewing results, follow the stdout/log file (nohup) and also  run tensorboard to view plots:
```bash
tensorboard --logdir ./logs --port 6003
```
To open the tensorboard UI in browser:
```bash
http://localhost:6003
```

### Inference

For running inference:
1. In config.py:
- set correctly DATASET_SELECT (CACD / Morph2)
- set the backbone correctly (e.g. USE_EFFICIENTNET)
- set the path for the weights of the trained model using INFERENCE_MODEL_WEIGHTS_PATH.
- set the input estimation file name to be used for the testing (if it is not so already), INPUT_ESTIMATION_FILE_NAME_TEST. 
- set the path model inference results will be written to (json file),
INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH.
(you may need to also set INFERENCE_BASED_ON_F according whether to use the F or R  head for inferenc - likely take the one who gave best validation/test result during training)
2. Run 
```bash
python ep3_infer.py.
```
3. The results are available in the file pointed by INFERENCE_MODEL_RESULTS_BASE_SAVE_PATH and stdout.

### Inference Analysis

To analyze the inference results (error histogram, together with original BAR's) use the ep3_advanced_stats.ipynb notebook.

### Iterated Improvement Flow

In iterated improvement, the model is re-trained with it's previous training results' json.

Do it as follows:
1. Run training (see above)
2. Run inference (see above)
3. Take new created JSON files, feed through config.INPUT_ESTIMATION_FILE_NAME_TRAIN and INPUT_ESTIMATION_FILE_NAME_TEST.
4. Return to 1, or break if no actual improvement anymore.


## BAR Estimation Results Info

im2age_map_*_Morph2_good_eval_time_07_09_2024_12_55_40_weights_147000_2.5563 - Results for Morph2 of BAR with transformer, good eval setting (using the actual training set for training, dist set is already not there)

im2age_map_*_CACD_good_eval_time_19_09_2024_01_21_19_weights_79500_5.2991 - Results for CACD of BAR WIHTOUT transformer (it is the results of unified_main.py run), good eval setting (using the actual training set for training, dist set is already not there)

## Bias Analysis 

Follow same guidelines to set ep3_config.py for inference. 
Run bias_analysis.ipynb - make sure to change the results file name created.
Use bia_analysis_graphs to generate various graphs.