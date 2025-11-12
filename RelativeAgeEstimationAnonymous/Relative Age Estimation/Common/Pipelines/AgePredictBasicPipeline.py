import os
import torch
import numpy as np
import json
from PIL import Image

import shutil

# importing the sys module
import sys        
 
# appending the directory of mod.py
# in the sys.path list
sys.path.append('../')   

#from global_config import cfg


# MIMIC = False

# if MIMIC:
# 	import config_mimic as cfg
# else:
# 	import diff_pipeline1__config as cfg
# 	#import config as cfg


# UnifiedClassificaionAndRegressionAgeModel
# TransformerModel
# AgeTransformer
# AgeDiffModel

def get_age_transformer(device, 
						num_classes, 
						age_interval, 
						min_age, 
						max_age, 
						mid_feature_size,
						unified_model,
						transformer_model,
						age_transformer,
						cfg):
	#pretrained_model = UnifiedClassificaionAndRegressionAgeModel(7, 10, 15, 80)
	
	pretrained_model = unified_model(num_classes, age_interval, min_age, max_age)
	#pretrained_model_path = 'weights/Morph2/unified/RangerLars_lr_5e4_4096_epochs_60_batch_32_mean_var_vgg16_pretrained_recognition_bin_10_more_augs_RandomApply_warmup_cosine_recreate'
	
	# # Use this for starting from backbone weights trained for e2e task with transformer:
	# pretrained_model_path = 'weights/Morph2/transformer/encoder/time_10_01_2022_09_08_38bin_1_layers_4_heads_4_lr0.0003_batch_8_copies_10_mid_feature_size_1024_augs_at_val_context4'
	
	# Use this for starting from backbone weights trained for e2e task with no transformer:
	#pretrained_model_path = 'weights/Morph2/unified/iter/RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_1_2'

	
	# pretrained_model = UnifiedClassificaionAndRegressionAgeModel(4, 8, 12, 43)
	# pretrained_model_path = 'weights/AFAD/unified/iter/RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_8_stronger_augs_2'

	#pretrained_model_file = os.path.join(pretrained_model_path, "weights.pt")
	#pretrained_model.load_state_dict(torch.load(pretrained_model_file), strict=False)

	num_features = pretrained_model.num_features
	backbone = pretrained_model.base_net
	#backbone.train()
	backbone.to(device)

	# backbone = InceptionResnetV1(pretrained='vggface2')
	# num_features = 512

	transformer = transformer_model(
		age_interval, min_age, max_age,
		mid_feature_size, mid_feature_size,
		num_outputs=num_classes,
		n_heads=4, n_encoders=4, dropout=0.3,
		mode='context').to(device)
	age_transformer_inst = age_transformer(backbone, transformer, num_features, mid_feature_size)

	if not cfg.ENFORCE_CPU_ON_INFERENCE:
		if cfg.USE_GPU and cfg.MULTI_GPU:
			if torch.cuda.device_count() > 1:
				print("Using multiple GPUs (" + str(torch.cuda.device_count()) + ")")
				age_transformer_inst = torch.nn.DataParallel(age_transformer_inst)
		
	age_transformer_inst.to(device)


	age_transformer_pretrained_model_path = '../Common/Weights/Morph2/transformer/mae2.56'
	#age_transformer_pretrained_model_path = 'weights/Morph2/transformer/mae2.56_sure'
	age_transformer_pretrained_model_file = os.path.join(age_transformer_pretrained_model_path, "weights.pt")
	#age_transformer_pretrained_model_file = 'C:/Users/proj/thesis/thesis_pytorch/weights/Morph2/transformer/encoder/time_15_01_2022_11_48_46bin_1_layers_4_heads_4_lr0.00028_batch_16_copies_10_mid_feature_size_1024_augs_at_val_context4/weights.pt'
	# import pdb
	# pdb.set_trace()
	
	if cfg.ENFORCE_CPU_ON_INFERENCE:
		from collections import OrderedDict

		file_dict = torch.load(age_transformer_pretrained_model_file)
		new_file_dict = OrderedDict()
		for x in file_dict:
			new_file_dict[x.split('module.')[1]] = file_dict[x] #new_file_dict[]

		age_transformer_inst.load_state_dict(new_file_dict)
		# w = torch.load(age_transformer_pretrained_model_file, map_location=torch.device('cpu'))
		# age_transformer_inst.load_state_dict(w)#, strict=False)
		# age_transformer_inst = age_transformer_inst.module.to(device)
	else:
		# import pdb
		# pdb.set_trace()
		if not cfg.MULTI_GPU:
			from collections import OrderedDict

			file_dict = torch.load(age_transformer_pretrained_model_file)
			new_file_dict = OrderedDict()
			for x in file_dict:
				new_file_dict[x.split('module.')[1]] = file_dict[x] #new_file_dict[]

			age_transformer_inst.load_state_dict(new_file_dict)
		else:
			age_transformer_inst.load_state_dict(torch.load(age_transformer_pretrained_model_file))#, map_location=torch.device('cuda:0')))#, strict=False)
			# if not cfg.MULTI_GPU:
			# 	age_transformer_inst = age_transformer_inst.module.to(device)

	return age_transformer_inst


def get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, age_diff_model, is_ordinal_reg, cfg, age_diff_model_path_arg=None, age_diff_model_file_name_arg=None):
	#age_diff_model_inst = age_diff_model(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, is_ordinal_reg)
	if isinstance(age_diff_model, type):
		# it is only a class - need to create instance
		age_diff_model_inst = age_diff_model(age_interval, min_age, max_age, age_radius, device) #, config_type, added_embed_layer_size, diff_embed_layer_size, is_ordinal_reg)
	else:
		# it is the actual object - just take it
		age_diff_model_inst = age_diff_model

	if not cfg.ENFORCE_CPU_ON_INFERENCE:
		if cfg.USE_GPU and cfg.MULTI_GPU:
			if torch.cuda.device_count() > 1:
				print("Using multiple GPUs (" + str(torch.cuda.device_count()) + ")")
				age_diff_model_inst = torch.nn.DataParallel(age_diff_model_inst)
	if age_diff_model_path_arg is None:
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_13_04_2022_01_09_30_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms'
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_23_04_2022_15_01_56_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms' # best radius=3
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_22_05_2022_09_15_59_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms' # best radius = 5
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_29_05_2022_01_47_40_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms' # best radius = 5, trained with normal dist for data
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_17_05_2022_12_33_41_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms' # best radius=10
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_14_05_2022_19_58_36_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms' # for radius = 3 with outliers 
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_15_05_2022_14_34_50_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms' # radius=7
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_09_06_2022_18_23_58_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs15_batch16_vgg16_unfreezecnnon5_transforms'
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_10_06_2022_03_54_21_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs30_batch16_vgg16_unfreezecnnon5_transforms' # radius=3, ordinal reg, unfreeze on epoch 5, run for 15 epochs
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_11_06_2022_03_28_52_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs45_batch16_vgg16_unfreezecnnon5_transforms' # radius=3, ordinal reg, unfreeze on epoch 15, run for ~35 epochs, deeper model (diff embedding is deeper), 2X data
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_30_06_2022_09_50_34_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch16_vgg16_unfreezecnnon5_transforms' # radius=3, ordinal reg but with CondorOrdinalCrossEntropy, unfreeze on epoch 12, run for ~28 epochs, deeper model (diff embedding is deeper), 1X data, no Dropout
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_01_07_2022_11_44_17_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch16_vgg16_unfreezecnnon5_transforms' # radius=3, ordinal reg but with CondorOrdinalCrossEntropy, unfreeze on epoch 12, run for ~28 epochs, no diff embedding, 1X data, no Dropout
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_02_07_2022_18_22_37_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch16_vgg16_unfreezecnnon5_transforms' # radius=3, ordinal reg, unfreeze on epoch 12, run for ~28 epochs, deeper model (diff embedding is deeper), 1X data, no Dropout, diff embedding is batch parallel
		#age_diff_model_path ='weights/Morph2Diff/unified/iter/time_04_07_2022_21_35_51_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch8_vgg16_unfreezecnnon5_transforms' # radius=3, ordinal reg, unfreeze on epoch 5, run for 30 ? epochs, deeper model (diff embedding is deeper), 1X data, no Dropout, diff embedding is batch parallel, 20 refs with batch=8
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_13_07_2022_11_43_49_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch16_vgg16_unfreezecnnon5_transforms' # radius = 7, mimic
		#age_diff_model_path = 'D:/new_morph2_diff/weights/Morph2Diff/unified/iter/time_28_07_2022_10_55_27_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch64_vgg16_unfreezecnnon5_transforms'
		#age_diff_model_path = 'D:/new_morph2_diff/weights/Morph2Diff/unified/iter/time_28_07_2022_10_55_27_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch64_vgg16_unfreezecnnon5_transforms'
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline1/weights/Morph2Diff/unified/iter/time_06_08_2022_02_53_53'
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline1/weights/Morph2Diff/unified/iter/time_06_08_2022_13_31_28'
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline2/weights/Morph2Diff/unified/iter/time_07_08_2022_10_23_16'
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline2/weights/Morph2Diff/unified/iter/time_07_08_2022_17_40_51'
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline2/weights/Morph2Diff/unified/iter/time_08_08_2022_01_40_18'
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline4/weights/Morph2Diff/unified/iter/time_23_08_2022_22_33_01'  # specific range learning : radius 0-2. shallow
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline4/weights/Morph2Diff/unified/iter/time_23_08_2022_21_29_57'  # specific range learning : radius 0-2. shallow but just for 1 epoch
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline4/weights/Morph2Diff/unified/iter/time_13_08_2022_13_43_08'  # specific range learning : radius 0-2. deep
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline4/weights/Morph2Diff/unified/iter/time_14_08_2022_22_26_51' # specific range learning : radius 2-4
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline4/weights/Morph2Diff/unified/iter/time_16_08_2022_10_04_03' # specific range learning : radius 4-10
		age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline4/weights/Morph2Diff/unified/iter/time_23_09_2022_20_10_15' # specific range learning : radius 4-10
		#age_diff_model_path = 'D:/projects/age_estimation/DiffPipeline4/weights/Morph2Diff/unified/iter/time_17_08_2022_01_44_29' # specific range learning : radius 10-35
		# weights/Morph2Diff/unified/iter/
		# weights/Morph2Diff/unified/iter/
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_27_06_2022_14_44_47_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch16_vgg16_unfreezecnnon5_transforms' # radius=4, ordinal reg, unfreeze on epoch 15, run for ~35 epochs, deeper model (diff embedding is deeper), 2X data, no Dropout
		#age_diff_model_path = 'weights/Morph2Diff/unified/iter/time_24_06_2022_22_33_33_RangerLars_CosineAnnealingLR+GradualWarmupScheduler_lr0.0003_epochs60_batch16_vgg16_unfreezecnnon5_transforms' # radius=3, ordinal reg, unfreeze on epoch 12, run for ~28 epochs, deeper model (diff embedding is deeper), 2X data, no Dropout
		age_diff_model_file = os.path.join(age_diff_model_path, "weights.pt")
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_400_1.1953.pt")
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_2300_6.3726.pt") #
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_3596_4.7118.pt") #time_06_08_2022_13_31_28
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_172000_1.4337.pt") # radius=3, ordinal reg, unfreeze on epoch 15, run for ~35 epochs, deeper model (diff embedding is deeper), 2X data
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_116256_1.4292.pt") # radius=3, ordinal reg but with CondorOrdinalCrossEntropy, unfreeze on epoch 12, run for ~28 epochs, deeper model (diff embedding is deeper), 1X data, no Dropout
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_82000_1.4373.pt") # radius=3, ordinal reg but with CondorOrdinalCrossEntropy, unfreeze on epoch 12, run for ~28 epochs, no diff embedding, 1X data, no Dropout
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_77476_1.4376.pt") # radius=3, ordinal reg, unfreeze on epoch 12, run for ~28 epochs, deeper model (diff embedding is deeper), 1X data, no Dropout, diff embedding is batch parallel 
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_156000_1.3966.pt") # radius=3, ordinal reg, unfreeze on epoch 5, run for 30 ? epochs, deeper model (diff embedding is deeper), 1X data, no Dropout, diff embedding is batch parallel, 20 refs with batch=8
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_22000_softage0.2927.pt") # radius=7, mimic
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_16800_2.4308.pt") # time_28_07_2022_10_55_27_RangerLars_CosineAnnealingLR
		#age_diff_model_file = os.path.join(age_diff_model_path, "")
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_132864_1.4216.pt") # radius=3, ordinal reg, unfreeze on epoch 15, run for ~35 epochs, deeper model (diff embedding is deeper), 2X data, no Dropout
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_96000_1.7166.pt") # radius=4, ordinal reg, unfreeze on epoch 15, run for ~35 epochs, deeper model (diff embedding is deeper), 2X data, no Dropout
		
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_34000_1.5071.pt")
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_49806_2.4882.pt") # for radius = 10 best
		#age_diff_model_file = os.path.join(age_diff_model_path, "weights_60000_2.5902.pt") # for radius = 3 with outliers 
	else:
		print("Age diff model path is given from notebook")
		age_diff_model_path = age_diff_model_path_arg
		age_diff_model_file = os.path.join(age_diff_model_path, age_diff_model_file_name_arg)

	if cfg.ENHANCED_CHECKPOINT_ORIGIN:
		# import pdb
		# pdb.set_trace()
		age_diff_model_inst.load_state_dict(torch.load(age_diff_model_file)['model_state_dict'])
	elif cfg.ENFORCE_CPU_ON_INFERENCE:
		from collections import OrderedDict

		file_dict = torch.load(age_diff_model_file)
		new_file_dict = OrderedDict()
		for x in file_dict:
			new_file_dict[x.split('module.')[1]] = file_dict[x] #new_file_dict[]

		age_diff_model_inst.load_state_dict(new_file_dict)
	else:
		if not cfg.MULTI_GPU:
			from collections import OrderedDict

			file_dict = torch.load(age_diff_model_file)
			new_file_dict = OrderedDict()
			for x in file_dict:
				new_file_dict[x.split('module.')[1]] = file_dict[x] #new_file_dict[]

			age_diff_model_inst.load_state_dict(new_file_dict)
		else:
			age_diff_model_inst.load_state_dict(torch.load(age_diff_model_file))#, strict=False)
	age_diff_model_inst.to(device)

	return age_diff_model_inst

	


class RefsQuerier:
	def __init__(self, metadata, num_references):
		self.metadata = metadata
		self.num_references = num_references
		self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.metadata])

	def _find_ref_image(self, query_age, ages):
		# import pdb
		# pdb.set_trace()
		# setting the range of possible diffs in order to select a specific one.
		# we don't want to select one that won't have possible references in dataset (ages not in range)
		
		# we look for a diff that will have examples in our dataset. We randomly choose from the range.
		# if the selected diff doesn't have candidates at all, we try a different diff from range. We are supposed 
		# to find something. If we find, we continue. If we don't find, we abort since the given age cannot be used for the task
		# (but there a very small chance this will happen)s
		not_found = True
		on_first = True
		# find all pool of potential references for this age diff
		idxs = np.where(ages == query_age)

		if len(idxs[0]) == 0: 
			on_first = False
			print("zero ref images with age: ", query_age)
			#print("ABORTING...")
			return [], 0
		#else:
			#print("found ref images for age ", query_age)

		# we found a diff we have one or more candidates. In case we have less than the needed 
		# num of references, we will replicate (probablistically). Else, we can just sample the num of references 
		# of unique references.
		if len(idxs[0]) < self.num_references:
			# import pdb
			# pdb.set_trace()
			#print(query_age, idxs)
			#print("not enough ref images in radius! replicating")
			selected_idxs = np.random.choice(idxs[0], self.num_references)
			return selected_idxs, 1
		else:
			selected_idxs = np.random.choice(idxs[0], self.num_references, replace=False)
			return selected_idxs, 1

	def query(self, query_age):
		return self._find_ref_image(query_age, self.ages)


class AgePredictBasicPipeline:
	def __init__(self, 
					device,
					num_classes, 
					age_interval, 
					min_age, 
					max_age, 
					age_radius,
					mid_feature_size, 
					images_train_db,
					metadata,
					deep, 
					num_references, 
					config_type, 
					added_embed_layer_size, 
					diff_embed_layer_size,
					transform,
					copies,
					unified_model,
					transformer_model,
					age_transformer,
					age_diff_model,
					is_ordinal_reg,
					cfg,
					no_age_transformer_init=False):
		self.device = device
		self.cfg = cfg
		self.images_train_db = images_train_db
		self.metadata = metadata
		self.copies = copies
		self.transform = transform
		if no_age_transformer_init:
			self.age_transformer_inst = None
		else:
			self.age_transformer_inst = get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size, unified_model, transformer_model, age_transformer, cfg)
		self.ref_querier = RefsQuerier(metadata, num_references)
		self.age_diff_predictor = get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, age_diff_model, is_ordinal_reg, cfg)
		if not no_age_transformer_init:
			self.age_transformer_inst.eval()
		self.age_diff_predictor.eval()

	def predict_diff_only(self, inputs):
		with torch.no_grad():
			classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(inputs)
		return classification_logits, age_diff_pred_hard, age_diff_pred_soft

	# def predict_iterative(self, query, query_indep, ref_age=-100, compare_post_round=False):
		
	# 	self.predict_stochastic(query, query_indep, ref_age=, compare_post_round)

	def predict_stochastic(self, query, query_indep, ref_age=-100, compare_post_round=False):
		image = query

		with torch.no_grad():
			_, query_est_age = self.age_transformer_inst(image)

		age_pred_raw = query_est_age.cpu()
		age_pred_round = int(np.round(age_pred_raw))

		ref_pred_ages = []
		for i in range(10):
			valid = 0
			while not valid:
				stochastic_delta = np.random.normal(0, 2)

				age_pred = int(np.round(age_pred_raw + stochastic_delta))

				ref_image_idx, valid = self.ref_querier.query(age_pred)

				if valid:
					image = query_indep[0]
					
					#metadata = json.loads(self.metadata[idx])

					ref_image_arr = self.images_train_db[ref_image_idx]
					ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
					ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
					ref_image_arr_metadata = [json.loads(self.metadata[idx]) for idx in ref_image_idx]
					ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
					
					image_vec = torch.stack(tuple([image.cpu()] + ref_image_arr))

					image_vec.to(self.device)
					image_vec = image_vec.view(1,image_vec.shape[0],image_vec.shape[1],image_vec.shape[2],image_vec.shape[3])

					
					with torch.no_grad():
						classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(image_vec)
					
					diff_select = age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_soft

					ref_pred_ages.append(float(diff_select.cpu()) + age_pred)
				# else:
				# 	return -1, -1

		avg_pred_age = np.mean(np.array(ref_pred_ages))

		if np.abs(age_pred_round - ref_age.cpu()) <= self.cfg.AGE_RADIUS: # TODO : remove this after research !!!
		#if np.abs((age_pred + diff_select - ref_age).cpu()) <= cfg.AGE_RADIUS:
			if compare_post_round:
				return 1, torch.Tensor([np.round(avg_pred_age)]).to(self.device)
			else:
				return 1, torch.Tensor([avg_pred_age]).to(self.device)
		else:
			if compare_post_round:
				return -2, torch.Tensor([np.round(avg_pred_age)]).to(self.device)
			else:
				return -2, torch.Tensor([avg_pred_age]).to(self.device)




	def predict(self, query, query_indep, tranform_query=False, bypass_diff=False, ref_age=-100, compare_post_round=False):
		if tranform_query:
			orig_image = Image.fromarray(query)
			if self.copies > 1:
				images = []
				for i in range(self.copies):
					images.append(self.transform(orig_image))
				image = torch.stack(images)
			else:
				image = self.transform(orig_image)
		else:
			image = query

		with torch.no_grad():
			_, query_est_age = self.age_transformer_inst(image)
		

		# import pdb
		# pdb.set_trace()
		if bypass_diff:
			age_pred_raw = query_est_age.cpu()
			age_pred = int(np.round(age_pred_raw)) #query_est_age
			if np.abs(age_pred - ref_age.cpu()) <= self.cfg.AGE_RADIUS: # TODO : remove this after research !!!
				if compare_post_round:
					return 1, torch.Tensor([age_pred]).to(self.device)
				else:
					return 1, query_est_age					 
			else:
				if compare_post_round:
					return -2, torch.Tensor([age_pred]).to(self.device)
				else:
					return -2, query_est_age
			#return torch.Tensor([age_pred]).to(self.device)
		else:
			#if np.abs(int(np.round(query_est_age.cpu())) - ref_age.cpu()) <= cfg.AGE_RADIUS: # TODO : remove this after research !!!
			age_pred = int(np.round(query_est_age.cpu()))
			# iteration = 0
			# done = False
			# while True:
			#	age_pred = int(age_pred)
			loe = []
			for noe in range(1):
				
				ref_image_idx, valid = self.ref_querier.query(age_pred)
				if valid:
					if tranform_query:
						orig_image = query
						
						orig_image = Image.fromarray(orig_image)
						image = orig_image
						# if self.copies > 1:
						# 	images = []
						# 	for i in range(self.copies):
						# 		images.append(self.transform(orig_image))
						# 	image = torch.stack(images)
						# else:
						image = self.transform(orig_image)
					else:
						# image = Image.fromarray(query_original)
						# image = self.transform(image) #query[0,0,:,:,:]
						image = query_indep[0]
					
					#metadata = json.loads(self.metadata[idx])

					ref_image_arr = self.images_train_db[ref_image_idx]
					ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
					ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
					ref_image_arr_metadata = [json.loads(self.metadata[idx]) for idx in ref_image_idx]
					ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
					#print(ref_age, age_pred, ref_image_arr_age)

					# import matplotlib.pyplot as plt
					# plt.imshow(np.transpose(ref_image_arr[0].numpy(), (1, 2, 0)))
					# plt.show()

					#ref_image_label = ref_image_arr_age[0] #// self.age_interval
					
					
					# import pdb
					# pdb.set_trace()
					image_vec = torch.stack(tuple([image.cpu()] + ref_image_arr))

					image_vec.to(self.device)
					image_vec = image_vec.view(1,image_vec.shape[0],image_vec.shape[1],image_vec.shape[2],image_vec.shape[3])

					

					# import pdb
					# pdb.set_trace()

					#print(pair_label)

					
					with torch.no_grad():
						classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(image_vec)
					
					# import pdb
					# pdb.set_trace()

					#return query_est_age + age_diff_pred_soft
					# torch.Tensor([age_pred]).to(self.device)
					
					#print(age_diff_pred_soft, age_pred - ref_age.cpu())
					#if iteration == 0:
					#print(age_diff_pred_hard, age_diff_pred_soft)
					diff_select = torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_soft
					#diff_select = age_diff_pred_soft
					# import pdb
					# pdb.set_trace()
					loe.append(float(diff_select.cpu()))
				
			if len(loe) != 0:
				diff_select = np.mean(np.array(loe))
				are_pred_with_fix = int(np.round(age_pred + diff_select)) #.cpu()
				if np.abs(age_pred - ref_age.cpu()) <= self.cfg.AGE_RADIUS: # TODO : remove this after research !!!
				#if np.abs((age_pred + diff_select - ref_age).cpu()) <= cfg.AGE_RADIUS:
					if compare_post_round:
						return 1, torch.Tensor([are_pred_with_fix]).to(self.device)
					else:
						return 1, torch.Tensor([age_pred + diff_select]).to(self.device)
				else:
					if compare_post_round:
						return -2,  torch.Tensor([are_pred_with_fix]).to(self.device)
					else:
						return -2, torch.Tensor([age_pred + diff_select]).to(self.device)

				# if age_diff_pred_soft > 2.5:
				# 	age_pred += age_diff_pred_soft
				# elif age_diff_pred_soft < -2.5:
				# 	age_pred -= age_diff_pred_soft
				# else:
				# 	if np.abs(age_pred - ref_age.cpu()) <= cfg.AGE_RADIUS: # TODO : remove this after research !!!
				# 		return 1, age_pred + age_diff_pred_soft
				# 	else:
				# 		return -2, age_pred + age_diff_pred_soft

				# iteration += 1
			else:
				return -1, -1
			# else:
			# 	return -2

	def predict_diff(self, query, query_indep, ref_age=-100, compare_post_round=False):
		image = query

		with torch.no_grad():
			_, query_est_age = self.age_transformer_inst(image)

		#if np.abs(int(np.round(query_est_age.cpu())) - ref_age.cpu()) <= cfg.AGE_RADIUS: # TODO : remove this after research !!!
		age_pred = int(np.round(query_est_age.cpu()))
		# iteration = 0
		# done = False
		# while True:
		#	age_pred = int(age_pred)
		loe = []
		for noe in range(1):
			
			ref_image_idx, valid = self.ref_querier.query(age_pred)
			if valid:
				image = query_indep[0]
				
				#metadata = json.loads(self.metadata[idx])

				ref_image_arr = self.images_train_db[ref_image_idx]
				ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
				ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
				ref_image_arr_metadata = [json.loads(self.metadata[idx]) for idx in ref_image_idx]
				ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
				#print(ref_age, age_pred, ref_image_arr_age)

				# import matplotlib.pyplot as plt
				# plt.imshow(np.transpose(ref_image_arr[0].numpy(), (1, 2, 0)))
				# plt.show()

				#ref_image_label = ref_image_arr_age[0] #// self.age_interval
				
				
				# import pdb
				# pdb.set_trace()
				image_vec = torch.stack(tuple([image.cpu()] + ref_image_arr))

				image_vec.to(self.device)
				image_vec = image_vec.view(1,image_vec.shape[0],image_vec.shape[1],image_vec.shape[2],image_vec.shape[3])

				

				# import pdb
				# pdb.set_trace()

				#print(pair_label)

				
				with torch.no_grad():
					classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(image_vec)
				
				# import pdb
				# pdb.set_trace()

				#return query_est_age + age_diff_pred_soft
				# torch.Tensor([age_pred]).to(self.device)
				
				#print(age_diff_pred_soft, age_pred - ref_age.cpu())
				#if iteration == 0:
				#print(age_diff_pred_hard, age_diff_pred_soft)
				#diff_select = torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_soft
				diff_select = age_diff_pred_soft
				# import pdb
				# pdb.set_trace()
				loe.append(float(diff_select.cpu()))
			
		if len(loe) != 0:
			diff_select = np.mean(np.array(loe))
			diff_select_with_fix = int(np.round(diff_select)) #.cpu()
			if np.abs(query_est_age.cpu() - ref_age.cpu()) <= self.cfg.AGE_RADIUS: # TODO : remove this after research !!!
			#if np.abs((age_pred + diff_select - ref_age).cpu()) <= cfg.AGE_RADIUS:
				if compare_post_round: 
					return 1, torch.Tensor([diff_select_with_fix]).to(self.device), torch.Tensor([int(np.round(ref_age.cpu() - query_est_age.cpu()))]).to(self.device)
				else:
					return 1, torch.Tensor([diff_select]).to(self.device), ref_age - query_est_age
			else:
				if compare_post_round:
					return -2,  torch.Tensor([diff_select_with_fix]).to(self.device), torch.Tensor([int(np.round(ref_age.cpu() - query_est_age.cpu()))]).to(self.device)
				else:
					return -2, torch.Tensor([diff_select]).to(self.device), ref_age - query_est_age


# This pipeline is supposed to get an age and additive noise (dev) and predict an age estimate based on
# the additively noised age (input age + dev). This is supposed to model the algorithm, where we use 
# the base age predict pipeline as a first estimate and we know it has some noise (MAE=~2.53).
# This class enables us to see, for modeled noise how much can we improve - "reduce" the uncertainty that
# is represented by dev.
class DiffPredictCheckPipeline:
	def __init__(self, 
					device,
					num_classes, 
					age_interval, 
					min_age, 
					max_age, 
					age_radius,
					mid_feature_size, 
					images_train_db,
					metadata,
					deep, 
					num_references, 
					config_type, 
					added_embed_layer_size, 
					diff_embed_layer_size,
					transform,
					copies,
					unified_model,
					transformer_model,
					age_transformer,
					age_diff_model):
		self.device = device
		self.images_train_db = images_train_db
		self.metadata = metadata
		self.copies = copies
		self.transform = transform
		self.age_transformer_inst = get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size, unified_model, transformer_model, age_transformer, cfg)
		self.ref_querier = RefsQuerier(metadata, num_references)
		self.age_diff_predictor = get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, age_diff_model, cfg)
		self.age_transformer_inst.eval()
		self.age_diff_predictor.eval()

		

	def predict(self, query, dev):
		age_pred = ref_age + dev
		#age_pred = int(age_pred)
		age_pred = int(np.round(age_pred.cpu()))
		valid = 0
		ref_image_idx, valid = self.ref_querier.query(age_pred)

		if valid:
			image = query[0,0,:,:,:]
			
			#metadata = json.loads(self.metadata[idx])

			ref_image_arr = self.images_train_db[ref_image_idx]
			ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
			ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
			ref_image_arr_metadata = [json.loads(self.metadata[idx]) for idx in ref_image_idx]
			ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
			#print(ref_age, age_pred, ref_image_arr_age)

			# import matplotlib.pyplot as plt
			# plt.imshow(np.transpose(ref_image_arr[0].numpy(), (1, 2, 0)))
			# plt.show()

			#ref_image_label = ref_image_arr_age[0] #// self.age_interval
			
			
			
			
			image_vec = torch.stack(tuple([image.cpu()] + ref_image_arr))

			image_vec.to(self.device)
			image_vec = image_vec.view(1,image_vec.shape[0],image_vec.shape[1],image_vec.shape[2],image_vec.shape[3])

			

			# import pdb
			# pdb.set_trace()

			#print(pair_label)

			
			with torch.no_grad():
				classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(image_vec)
			
			# import pdb
			# pdb.set_trace()

			#return query_est_age + age_diff_pred_soft
			# torch.Tensor([age_pred]).to(self.device)
			
			#print(age_diff_pred_soft, age_pred - ref_age.cpu())
			return 1, age_pred + age_diff_pred_soft
		else:
			return -1, -1




class AgePredictOriginalPipeline:
	def __init__(self, 
					device,
					num_classes, 
					age_interval, 
					min_age, 
					max_age, 
					age_radius,
					mid_feature_size, 
					images_train_db,
					metadata,
					deep, 
					num_references, 
					config_type, 
					added_embed_layer_size, 
					diff_embed_layer_size,
					transform,
					copies,
					unified_model,
					transformer_model,
					age_transformer,
					is_round,
					cfg):
		self.device = device
		self.images_train_db = images_train_db
		self.metadata = metadata
		self.copies = copies
		self.transform = transform
		self.age_transformer_inst = get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size, unified_model, transformer_model, age_transformer, cfg)
		self.age_transformer_inst.eval()
		self.is_round = is_round

	def predict(self, query, tranform_query=False, with_logits=False):
		if tranform_query:
			orig_image = Image.fromarray(query)
			if self.copies > 1:
				images = []
				for i in range(self.copies):
					images.append(self.transform(orig_image))
				image = torch.stack(images)
			else:
				image = self.transform(orig_image)
		else:
			image = query

		with torch.no_grad():
			cls_logits, query_est_age = self.age_transformer_inst(image)

		if with_logits:
			if self.is_round:
				return cls_logits, torch.round(query_est_age.cpu())
			else:
				return cls_logits, query_est_age
		else:
			if self.is_round:
				return torch.round(query_est_age.cpu())
			else:
				return query_est_age



class AgePredictWrapper:
	def __init__(self, 
					device,
					num_classes, 
					age_interval, 
					min_age, 
					max_age, 
					age_radius,
					mid_feature_size, 
					unified_model,
					transformer_model,
					age_transformer,
					cfg):
		
		self.age_transformer_inst = get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size, unified_model, transformer_model, age_transformer, cfg)
		self.age_transformer_inst.eval()

	def get_model(self):
		return self.age_transformer_inst