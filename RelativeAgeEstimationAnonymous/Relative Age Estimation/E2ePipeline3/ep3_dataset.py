##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline3
#	Date		:	28.10.2023
# 	Description	: 	Datasets file. Provided relevant pytorch-based datasets
#					For the task.
##############################################################################

import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import random
import numpy as np

from ep3_config import USE_GENDER, IS_GENDER_IS_DATASET, ERROR_SAT_RANGE_MIN, ERROR_SAT_RANGE_MAX, DIST_APPROX_METHOD

"""
Dataset:
Creates examples of [Q, R[0],...,R[N-1]]
Where R are from taken the closest in emebeddings space (closest_embeddings_to_choose_from). 
"""
class QueryAndMultiAgeRefsDataset(Dataset):
	# Constructor
	def __init__(self, 
			min_age,
			max_age,
			age_interval,
			transform,
			num_references,
			embeddings_knn,
			base_data_set_images,                
			base_data_set_metadata,   
			base_data_set_embeddings,            
			ref_data_set_images,                
			ref_data_set_metadata,              
			ref_data_set_embeddings,
			dataset_size_factor,
			base_set_is_ref_set,
			disable_same_ref_being_query,
			knn_reduced_pool_size,
			sample_knn_reduced_pool,
			base_model_distribution,
			im2age_map,
			mode_select,
			distribution_approximation_method=None,
			use_knn=True
			):
		# attributes
		self.min_age = min_age
		self.max_age = max_age
		self.age_interval = age_interval
		self.transform = transform
		self.num_references = num_references
		self.embeddings_knn = embeddings_knn
		self.base_data_set_images = base_data_set_images[:, :, :, [2, 1, 0]]
		self.base_data_set_metadata = base_data_set_metadata            
		self.base_data_set_embeddings = base_data_set_embeddings.reshape(base_data_set_embeddings.shape[0], base_data_set_embeddings.shape[2])
		self.ref_data_set_images = ref_data_set_images[:, :, :, [2, 1, 0]]
		self.ref_data_set_metadata = ref_data_set_metadata  
		self.ref_data_set_age = np.array([float(json.loads(ref_image_metadata)['age']) for ref_image_metadata in self.ref_data_set_metadata])
		self.ref_data_set_embeddings = ref_data_set_embeddings.reshape(ref_data_set_embeddings.shape[0], ref_data_set_embeddings.shape[2])
		self.dataset_size_factor = dataset_size_factor
		self.base_set_is_ref_set = base_set_is_ref_set
		self.disable_same_ref_being_query = disable_same_ref_being_query
		self.knn_reduced_pool_size = knn_reduced_pool_size
		self.sample_knn_reduced_pool = sample_knn_reduced_pool

		self.base_model_distribution = base_model_distribution
		self.im2age_map = im2age_map
		self.mode_select = mode_select
		self.distribution_approximation_method = distribution_approximation_method
		
		self.use_knn = use_knn

		print(self.mode_select)

	# core method to find references per query image
	def _find_ref_image(self, query_idx, query_age):
		#print(f"time start _find_ref_image:{time.time()}")
		# get the ref indexes sorted by embeddings cosine similarity
		base_data_set_embeddings_query = self.base_data_set_embeddings[query_idx]
		base_data_set_embeddings_query = base_data_set_embeddings_query.reshape(1, base_data_set_embeddings_query.shape[0])
		orig_idxs = np.arange(len(self.ref_data_set_embeddings))
		if self.mode_select == "apply_distribution": 
			# Getting ref age according error distribution (model emulation)
			selected = False
			while not selected:
				# adding noise
				if self.distribution_approximation_method == "gaussian_based":
					query_age_noised = np.round(query_age + np.random.normal(loc=self.base_model_distribution["mean"], scale=self.base_model_distribution["std"]))
				elif self.distribution_approximation_method == "kde_based":
					query_age_noised = np.round(query_age - self.base_model_distribution["kde"].resample(size=1)[0][0])
				elif self.distribution_approximation_method == "kde_based_saturated":
					error_sample = self.base_model_distribution["kde"].resample(size=1)[0][0]
					if error_sample < ERROR_SAT_RANGE_MIN:
						error_sample = ERROR_SAT_RANGE_MIN
					elif error_sample > ERROR_SAT_RANGE_MAX:
						error_sample = ERROR_SAT_RANGE_MAX
					query_age_noised = np.round(query_age - error_sample)
				elif self.distribution_approximation_method == "uniform_based":
					query_age_noised = np.round(query_age + np.random.randint(low=-3, high=3))
				else:
					print("invalid distribution approximation method")
					exit(1)
				# if query_age < self.base_model_distribution["mid_value"]:
				# 	query_age_noised = np.round(query_age + np.random.normal(loc=self.base_model_distribution["low"]["mean"], scale=self.base_model_distribution["low"]["std"]))
				# else:
				# 	query_age_noised = np.round(query_age + np.random.normal(loc=self.base_model_distribution["high"]["mean"], scale=self.base_model_distribution["high"]["std"]))
				# safety limits
				if query_age_noised > self.max_age - 1:
					query_age_noised = self.max_age - 1
				elif query_age_noised < self.min_age + 1:
					query_age_noised = self.min_age + 1
				# filter
				orig_idxs_filtered = np.where(self.ref_data_set_age == query_age_noised)[0]
				# debug notification
				selected = True
				if len(orig_idxs_filtered) == 0:
					selected = False
					print("apply distribution: empty refs pool, trying again...")
				elif len(orig_idxs_filtered) == 1:
					print("apply distribution: only single")
					if self.base_set_is_ref_set:
						if self.disable_same_ref_being_query:
							if query_age_noised == query_age:
								selected = False
								print("apply distribution: only single which is self")
				
		elif self.mode_select == "apply_map":
			# Getting ref age according map (model)
			# age predicted on image by model
			query_age_noised = self.im2age_map[str(query_idx)]
			# round in order to get an actual age
			ref_age = np.round(query_age_noised)
			query_age_noised = ref_age
			# automatically fix in case we are out of range
			if ref_age > self.max_age - 1:
				ref_age = self.max_age - 1
			elif ref_age < self.min_age + 1:
				ref_age = self.min_age + 1
			# get all idxs is ref age
			orig_idxs_filtered = np.where(self.ref_data_set_age == ref_age)[0]

			if len(orig_idxs_filtered) == 0:
				print("Not ref found - adding some small noise")
			
			# in case no refs found, no choice but to get other ref - for different diff 
			while len(orig_idxs_filtered) == 0:
				# take a sample in radius around the original point
				ref_age = np.round(query_age_noised + np.random.normal(0, 2))
				# automatically fix in case we are out of range
				if ref_age > self.max_age - 1:
					ref_age = self.max_age - 1
				elif ref_age < self.min_age + 1:
					ref_age = self.min_age + 1
				orig_idxs_filtered = np.where(self.ref_data_set_age == ref_age)[0]
				# debug notification
				if len(orig_idxs_filtered) == 0:
					print("apply map: empty refs pool, trying again...")
		else: # "regular"
			print("unsupported mode")
			exit(1)
			#query_age_noised = query_age
			#orig_idxs_filtered = orig_idxs
		
		if self.use_knn:
			if self.sample_knn_reduced_pool:
				knn_reduced_pool_idxs = np.random.choice(orig_idxs_filtered, self.knn_reduced_pool_size)
			else:
				knn_reduced_pool_idxs = orig_idxs_filtered
			actual_ref_set_embeddings = self.ref_data_set_embeddings[knn_reduced_pool_idxs]
		
			base_data_set_embeddings_query_torch = torch.from_numpy(base_data_set_embeddings_query)
			ref_data_set_embeddings_torch = torch.from_numpy(actual_ref_set_embeddings)
			
			ref_idx_sort_by_sim_pre = np.argsort(torch.linalg.norm(base_data_set_embeddings_query_torch - ref_data_set_embeddings_torch, axis=1))
			ref_idx_sort_by_sim = knn_reduced_pool_idxs[ref_idx_sort_by_sim_pre]
			#ref_idx_sort_by_sim = np.argsort(np.linalg.norm(base_data_set_embeddings_query - actual_ref_set_embeddings, axis=1))
			#ref_idx_sort_by_sim = np.argsort(cosine_similarity(base_data_set_embeddings_query, actual_ref_set_embeddings)[0])
			# remove query from ref
			if self.base_set_is_ref_set:
				if self.disable_same_ref_being_query:
					ref_idx_sort_by_sim = ref_idx_sort_by_sim[ref_idx_sort_by_sim != query_idx]
			# take the knn
			ref_idx_sort_by_sim_knn = ref_idx_sort_by_sim[:self.embeddings_knn]
			if len(ref_idx_sort_by_sim_knn) == 0:
				print("no references! failure.")
				exit()
			# randomly selecting indexes
			if len(ref_idx_sort_by_sim_knn) < self.num_references:
				selected_idxs = np.random.choice(ref_idx_sort_by_sim_knn, self.num_references)
			else:
				selected_idxs = np.random.choice(ref_idx_sort_by_sim_knn, self.num_references, replace=False)
			#print(f"time end _find_ref_image:{time.time()}")
			# print("-------------------")
			# print(query_age_noised)
			# print(query_age)
		else:
			ref_idx_sort_by_sim = orig_idxs_filtered #knn_reduced_pool_idxs

			# remove query from ref
			if self.base_set_is_ref_set:
				if self.disable_same_ref_being_query:
					ref_idx_sort_by_sim = ref_idx_sort_by_sim[ref_idx_sort_by_sim != query_idx]

			if len(ref_idx_sort_by_sim) < self.num_references:
				selected_idxs = np.random.choice(ref_idx_sort_by_sim, self.num_references)
			else:
				selected_idxs = np.random.choice(ref_idx_sort_by_sim, self.num_references, replace=False)
		return selected_idxs, query_age_noised
			
	# dataset size
	def __len__(self):
		return self.dataset_size_factor*len(self.base_data_set_metadata)

	# iterator method
	def __getitem__(self, idx):
		#print(f"time start getitem:{time.time()}")
		# get the actual query index
		actual_query_idx = idx % len(self.base_data_set_metadata)
		# get the query image
		query_image = self.base_data_set_images[actual_query_idx]
		query_image = Image.fromarray(query_image)
		# apply transforms on the query image
		query_image = self.transform(query_image)
		# get the query's age
		query_metadata = json.loads(self.base_data_set_metadata[actual_query_idx])
		query_age = float(query_metadata['age'])
		
		if USE_GENDER:
			if query_metadata['gender'] == 'M':
				query_gender = 0
			else:
				query_gender = 1
			raw_query_gender = query_metadata['gender']
			raw_query_race = query_metadata['race']
		else:
			query_gender = -1
			if IS_GENDER_IS_DATASET:
				raw_query_gender = query_metadata['gender']
				raw_query_race = query_metadata['race']
			else:
				raw_query_gender = "NA"
				raw_query_race = "NA"
		# get the ref indexes
		ref_image_idx, query_age_noised = self._find_ref_image(actual_query_idx, query_age)
		# get the ref images 
		ref_image_arr = self.ref_data_set_images[ref_image_idx]
		ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
		# apply transforms on the ref images
		ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
		# get the ref ages
		ref_image_arr_metadata = [json.loads(self.ref_data_set_metadata[idx_i]) for idx_i in ref_image_idx]
		ref_image_arr_age = torch.Tensor([float(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata])
		# prepare batch input (images)
		image_vec = torch.stack(tuple([query_image] + ref_image_arr))
		# prepare batch output (labels)
		age_diffs_for_reg = torch.Tensor([query_age - ref_age for ref_age in ref_image_arr_age])
		if DIST_APPROX_METHOD == "kde_based_saturated":
			offset = -ERROR_SAT_RANGE_MIN
			age_diffs_for_cls_list = []
			age_minus_diffs_for_cls_list = []
			for ref_age in ref_image_arr_age:
				if query_age - ref_age < ERROR_SAT_RANGE_MIN:
					q_r_error = ERROR_SAT_RANGE_MIN
				elif query_age - ref_age > ERROR_SAT_RANGE_MAX:
					q_r_error = ERROR_SAT_RANGE_MAX
				else:
					q_r_error = query_age - ref_age
				age_diffs_for_cls_list.append(q_r_error+offset)
				age_minus_diffs_for_cls_list.append(-q_r_error+offset)
			age_diffs_for_cls = torch.Tensor(age_diffs_for_cls_list)
			age_minus_diffs_for_cls = torch.Tensor(age_minus_diffs_for_cls_list)
		else:
			offset = self.max_age - self.min_age
			age_diffs_for_cls = torch.Tensor([query_age - ref_age + offset for ref_age in ref_image_arr_age])
			age_minus_diffs_for_cls = torch.Tensor([ref_age - query_age + offset for ref_age in ref_image_arr_age])
		# build returned batch
		ret_batch = {
			'image_vec' : image_vec,
			'query_age' : query_age,
			'query_age_noised' : query_age_noised,
			'general_query_idx' : idx,
			'actual_query_idx' : actual_query_idx,
			'ref_idxs' : ref_image_idx,
			'age_diffs_for_reg' : age_diffs_for_reg,
			'age_diffs_for_cls' : age_diffs_for_cls,
			'age_minus_diffs_for_cls' : age_minus_diffs_for_cls,
			'age_refs' : ref_image_arr_age,
			'gender' : query_gender,
			'gender_raw' : raw_query_gender,
			'race' : raw_query_race
		}
		#print(f"time end getitem: {time.time()}")
		# return ret_batch
		return ret_batch