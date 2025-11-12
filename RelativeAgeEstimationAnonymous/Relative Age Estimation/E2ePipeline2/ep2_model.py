##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline2
#	Date		:	28.10.2023
# 	Description	: 	Models file. Provides relevant pytorch-based models
#					For the task.
##############################################################################

import os

import torch
from torch import nn as nn
from torch.nn import Linear, functional as F, Dropout
import torchvision.models as models

from Common.Models.ArcMarginClassifier import ArcMarginClassifier
from Common.Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel
# from Models.inception_resnet_v1 import InceptionResnetV1

from condor_pytorch.dataset import levels_from_labelbatch
from condor_pytorch.losses import condor_negloglikeloss
from condor_pytorch.dataset import logits_to_label
from condor_pytorch.activations import ordinal_softmax
from condor_pytorch.metrics import earth_movers_distance
from condor_pytorch.metrics import ordinal_accuracy
from condor_pytorch.metrics import mean_absolute_error

"""
Wrapper for feature extractor
"""
class FeatureExtractionVgg16(nn.Module):
	# Constructor
	def __init__(self,
			  	min_age,
				max_age,
				age_interval,
				pretrained_model_path,
				pretrained_model_file_name="weights.pt",
				):
		super(FeatureExtractionVgg16, self).__init__()
		
		# attributes
		self.base_net = models.vgg16(pretrained=True)
		self.min_age = min_age
		self.max_age = max_age
		self.age_interval = age_interval
		self.pretrained_model_path = pretrained_model_path
		self.pretrained_model_file_name = pretrained_model_file_name
		pretrained_model_file = os.path.join(pretrained_model_path, pretrained_model_file_name)
		
		# init backbone from existing network
		num_classes = int((max_age - min_age) / age_interval + 1)
		self.recog_model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age) #ArcMarginClassifier(10976)
		self.recog_model.load_state_dict(torch.load(pretrained_model_file), strict=False)
		self.base_net = self.recog_model.base_net
		self.base_net.classifier = self.base_net.base_net.classifier[:6]

	# Forward pass
	def forward(self, input):
		x = self.base_net.base_net.features(input)
		x = self.base_net.base_net.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.base_net.base_net.classifier(x)
		
		return x
	
"""
Main diff-based detection model
"""
class DiffBasedAgeDetectionModel(nn.Module):
	# Constructor
	def __init__(self,
			  		device,
			  		min_age,
					max_age,
					age_interval,
					num_references,
					pretrained_model_path,
					pretrained_model_file_name,
					dropout_p=0.5,
					num_of_fc_layers=3,
					age_embedding_size=128,
					is_ordinal=True,
					min_age_diff=-35,
					max_age_diff=35,
					num_classes_diff=71,
					age_diff_interval=1,
					regressors_diff_head=False
					):
		super(DiffBasedAgeDetectionModel, self).__init__()

		# attributes
		self.device = device
		self.min_age = min_age
		self.max_age = max_age
		self.age_interval = age_interval
		self.age_diff_interval = age_diff_interval
		self.num_references = num_references
		self.dropout_p = dropout_p
		self.num_of_fc_layers = num_of_fc_layers
		self.age_embedding_size = age_embedding_size
		self.is_ordinal = is_ordinal
		self.regressors_diff_head = regressors_diff_head

		# feature extractor
		self.base_net = FeatureExtractionVgg16(
			min_age=min_age,
			max_age=max_age,
			age_interval=age_interval,
			pretrained_model_path=pretrained_model_path,
			pretrained_model_file_name=pretrained_model_file_name
		)

		self.embeddings_size = self.base_net.base_net.classifier[3].out_features
		
		self.base_act = nn.LeakyReLU()
		self.base_dropout = nn.Dropout(self.dropout_p)

		self.fc_weight_head = [Linear(2*(self.embeddings_size+self.age_embedding_size), 2048)]
		self.fc_weight_head.append(nn.LeakyReLU())
		self.fc_weight_head.append(nn.Dropout(self.dropout_p))
		self.fc_diff_head = [Linear(2*(self.embeddings_size+self.age_embedding_size), 2048)]
		self.fc_diff_head.append(nn.LeakyReLU())
		self.fc_diff_head.append(nn.Dropout(self.dropout_p))
		for i in range(self.num_of_fc_layers-1):
			if i == self.num_of_fc_layers-2:
				self.fc_weight_head.append(Linear(int(2048/(2**i)), 1))
				self.fc_diff_head.append(Linear(int(2048/(2**i)), 1))
			else:
				self.fc_weight_head.append(Linear(int(2048/(2**i)), int(2048/(2**(i+1)))))
				self.fc_weight_head.append(nn.LeakyReLU())
				self.fc_weight_head.append(nn.Dropout(self.dropout_p))
				self.fc_diff_head.append(Linear(int(2048/(2**i)), int(2048/(2**(i+1)))))
				self.fc_diff_head.append(nn.LeakyReLU())
				self.fc_diff_head.append(nn.Dropout(self.dropout_p))
		self.fc_weight_head = nn.Sequential(*self.fc_weight_head) #nn.Sequential(*self.fc_weight_head)
		self.fc_diff_head = nn.ModuleList(self.fc_diff_head) #nn.Sequential(*self.fc_diff_head)

		self.age_embedding = nn.Embedding(num_embeddings=max_age-min_age, embedding_dim=self.age_embedding_size)

		
		last_layer_width = int(2048/(2**(self.num_of_fc_layers-2)))
	
		#self.batch_norm_layer_w = nn.BatchNorm1d(last_layer_width)
		#self.batch_norm_layer_d = nn.BatchNorm1d(last_layer_width)
		#self.batch_norm_layer_base = nn.BatchNorm1d(self.embeddings_size+self.age_embedding_size, eps=0.4)

		if self.is_ordinal:
			self.class_head = Linear(last_layer_width, num_classes_diff-1)
		else:
			self.class_head = Linear(last_layer_width, num_classes_diff)
		
		if self.regressors_diff_head:
			self.labels = range(num_classes_diff)

			self.regression_heads = []
			# self.classification_heads = []
			self.centers = []
			for i in self.labels:
				self.regression_heads.append(Linear(last_layer_width, 1))
				# self.classification_heads.append(Linear(k, int(self.age_intareval*2)))
				center = min_age_diff + 0.5 * age_diff_interval + i * age_diff_interval
				self.centers.append(center)

			self.regression_heads = nn.ModuleList(self.regression_heads)


	# freeze backbone
	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze
	
	# Forward pass
	def forward(self, input_images, query_noisy_age, input_ref_ages): #
		#############################################
		# Core
		#############################################
		ref_age_emb = self.age_embedding(input_ref_ages-self.min_age)
		query_age_emb = self.age_embedding(query_noisy_age-self.min_age)
		age_emb = torch.cat([query_age_emb.reshape(query_age_emb.shape[0],1,query_age_emb.shape[1]), ref_age_emb], dim=1)
		unpacked_age_emb = age_emb.view(age_emb.shape[0] * age_emb.shape[1], age_emb.shape[2])
		#torch.concat([ref_age_emb, packed_embs_b], axis=1)
		# flattening all batches, because we are going to just calculate embeddings on them which is the same 
		# operation for all inputs images
		# current input structure: batch, collection, channel, x, y
		unpacked_input = input_images.view(input_images.shape[0] * input_images.shape[1], input_images.shape[2], input_images.shape[3], input_images.shape[4])
        # calculate image embeddings
		base_embedding_a = self.base_net(unpacked_input)
		base_embedding_a1 = torch.cat([base_embedding_a, unpacked_age_emb], dim=1)
		base_embedding_b = self.base_act(base_embedding_a1) # F.leaky_relu(base_embedding_a1)
		#base_embedding_b = self.batch_norm_layer_base(base_embedding_b)
		base_embedding_b = self.base_dropout(base_embedding_b) #Dropout(self.dropout_p)(base_embedding_b)
		# get back to division to batches, since we want to continue separate calculation per batch
		packed_embs_a_pre = base_embedding_b.view(input_images.shape[0], input_images.shape[1], -1)
		# repeat the query embedding for number of references
		query_reshaped = packed_embs_a_pre[:, 0].view(packed_embs_a_pre.shape[0], 1, packed_embs_a_pre.shape[2])
		query_repeated = query_reshaped.repeat((1, self.num_references, 1))
		# create pairs of (Qe, Re[j]), for each Re[j] in the references embeddings, and flatten each pair
		flattened_query_ref_pairs = torch.stack((query_repeated, packed_embs_a_pre[:, 1:]), dim=2).view(packed_embs_a_pre.shape[0], self.num_references, 2*(self.embeddings_size+self.age_embedding_size))
		flattened_ref_query_pairs = torch.stack((packed_embs_a_pre[:, 1:], query_repeated), dim=2).view(packed_embs_a_pre.shape[0], self.num_references, 2*(self.embeddings_size+self.age_embedding_size))

		flattened_both_dir_pairs = torch.stack((flattened_query_ref_pairs, flattened_ref_query_pairs))
		fc_out_w = flattened_both_dir_pairs
		fc_out_d = flattened_both_dir_pairs

		#########################################
		# Old weight-diff code block
		#########################################
		# for i in range(self.num_of_fc_layers):
		# 	#############################################
		# 	# Weights
		# 	#############################################	
		# 	# fc_out_w = self.fc_weight_head[i](fc_out_w) 	# flattened_query_ref_pairs
		# 	# if i != self.num_of_fc_layers-1:
		# 	# 	fc_out_w = F.leaky_relu(fc_out_w)
		# 	# 	# if i == self.num_of_fc_layers-2:
		# 	# 	# 	fc_out_w = self.batch_norm_layer_w(fc_out_w)
		# 	# 	fc_out_w = Dropout(self.dropout_p)(fc_out_w)
				
		# 	#############################################
		# 	# Diff heads
		# 	#############################################
		# 	fc_out_d = self.fc_diff_head[i](fc_out_d) 	# flattened_query_ref_pairs
		# 	if i != self.num_of_fc_layers-1:
		# 		fc_out_d = F.leaky_relu(fc_out_d)
		# 		# if i == self.num_of_fc_layers-2:
		# 		# 	fc_out_d = self.batch_norm_layer_d(fc_out_d)
		# 		fc_out_d = Dropout(self.dropout_p)(fc_out_d)
		# 		if i == self.num_of_fc_layers-2:
		# 			fc_out_d_pre_reg = fc_out_d

		
		#########################################
		# New weight-diff code block
		#########################################
		fc_out_w = self.fc_weight_head(fc_out_w)
		for i, cur_diff_head_layer in enumerate(self.fc_diff_head):
			fc_out_d = cur_diff_head_layer(fc_out_d)
			if i == len(self.fc_diff_head) - 2:
				fc_out_d_pre_reg = fc_out_d

		weights = nn.Softmax(dim=2)(fc_out_w) 						# dim=1
		age_diffs = fc_out_d
		
		

		#############################################
		# Age head
		#############################################

		# ref_ages_tensor = torch.transpose(torch.stack(tuple(input_ref_ages)), 0, 1)
		ref_ages_tensor = input_ref_ages.reshape(input_ref_ages.shape[0], input_ref_ages.shape[1], 1)


		age_estimations_f = ref_ages_tensor + age_diffs[0]
		age_estimations_r = ref_ages_tensor - age_diffs[1]

		#print((weights[0]*age_estimations_f).sum(dim=1))
		age_pred_f = (weights[0]*age_estimations_f).sum(dim=1) # / torch.sum(weights[0], dim=1) # seems no need for div - since each value is 1
		# print(age_pred_f)
		# print(torch.sum(weights[0], dim=1))
		# print("--------------------------")
		age_pred_r = (weights[1]*age_estimations_r).sum(dim=1) # / torch.sum(weights[1], dim=1) # seems no need for div - since each value is 1
		

		# import math
		# for i in range(len(age_pred)):
		# 	if math.isnan(age_pred[i]):
		# 		import pdb
		# 		pdb.set_trace()

		
		age_diffs_f = age_diffs[0]
		age_diffs_r = age_diffs[1]

		# clasification
		classification_logits = self.class_head(fc_out_d_pre_reg[0])
		#classification_logits_minus = self.class_head(fc_out_d_pre_reg[1])

		if self.regressors_diff_head:
			if self.is_ordinal:
				weights_cls_diff_head = []
				for i in range(self.num_references):
					weights_cls_diff_head.append(ordinal_softmax(classification_logits[:,i,:], device=self.device).float())
				weights_cls_diff_head_t = torch.stack(weights_cls_diff_head, dim=1)
			else:
				weights_cls_diff_head_t = nn.Softmax()(classification_logits)
			
			t = []
			for i in self.labels:
				t.append(torch.squeeze(self.regression_heads[i](fc_out_d_pre_reg[0]) + self.centers[i]).to(self.device) * weights_cls_diff_head_t[:, :, i])

			age_diff_pred_cls = torch.stack(t, dim=0).sum(dim=0) / torch.sum(weights_cls_diff_head_t, dim=2)

		#return age_pred.reshape(age_pred.shape[0], age_pred.shape[1]), age_diffs.reshape(age_diffs.shape[0], age_diffs.shape[1])
		if self.regressors_diff_head:
			return age_pred_f.reshape(age_pred_f.shape[0], age_pred_f.shape[1]), \
					age_pred_r.reshape(age_pred_r.shape[0], age_pred_r.shape[1]), \
					age_diffs_f.reshape(age_diffs_f.shape[0], age_diffs_f.shape[1]), \
					age_diffs_r.reshape(age_diffs_r.shape[0], age_diffs_r.shape[1]), \
					classification_logits, age_diff_pred_cls
					#classification_logits, classification_logits_minus, age_diff_pred_cls
		else:
			return age_pred_f.reshape(age_pred_f.shape[0], age_pred_f.shape[1]), \
					age_pred_r.reshape(age_pred_r.shape[0], age_pred_r.shape[1]), \
					age_diffs_f.reshape(age_diffs_f.shape[0], age_diffs_f.shape[1]), \
					age_diffs_r.reshape(age_diffs_r.shape[0], age_diffs_r.shape[1]), \
					classification_logits