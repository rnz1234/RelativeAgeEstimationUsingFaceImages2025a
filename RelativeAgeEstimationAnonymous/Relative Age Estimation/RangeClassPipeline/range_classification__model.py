import os

import math 

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


class FeatureExtractionVgg16(nn.Module):
	def __init__(self, min_age, max_age, age_interval):
		super(FeatureExtractionVgg16, self).__init__()

		# self.base_net = models.vgg16(pretrained=True)
		age_interval = 1
		min_age = 15
		max_age = 80
		num_classes = int((max_age - min_age) / age_interval + 1)
		
		recog_model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age) #ArcMarginClassifier(10976)
		pretrained_model_path = '../Common/Weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64'
		#pretrained_model_path = '../Common/Weights/Morph2/transformer/mae2.56'
		pretrained_model_file = os.path.join(pretrained_model_path, "weights.pt")

		file_dict = torch.load(pretrained_model_file)
		from collections import OrderedDict
		new_file_dict = OrderedDict()
		# for x in file_dict:
		# 	new_file_dict[x.split('module.')[1]] = file_dict[x] #new_file_dict[]
		# new_file_dict2 = OrderedDict()
		# for x in new_file_dict:
		# 	if 'backbone' in x:
		# 		new_file_dict2['base_net.' + x.split('backbone.')[1]] = new_file_dict[x]

		recog_model.load_state_dict(file_dict, strict=False) #new_file_dict2

		#recog_model.load_state_dict(torch.load(pretrained_model_file), strict=False)

		self.base_net = recog_model.base_net

		#self.base_net.classifier = self.base_net.classifier[:6]

	def forward(self, input):
		# x = self.base_net.features(input)
		# x = self.base_net.avgpool(x)
		# x = torch.flatten(x, 1)
		# x = self.base_net.classifier(x)
		x = self.base_net(input)

		return x



class RangeClassificationModel(nn.Module):
	def __init__(self, 
					age_interval,
					min_age,
					max_age,
					age_radius=3, 
					device=torch.device("cuda:0")):
		super(RangeClassificationModel, self).__init__()

		self.device = device
		self.age_interval = age_interval
		self.age_radius = age_radius

		# num_features = 512
		# self.base_net = InceptionResnetV1(pretrained='vggface2')
		# self.base_net = FeatureExtractionResnet34()

		self.num_features = 4096
		self.base_net = FeatureExtractionVgg16(min_age=min_age, max_age=max_age, age_interval=age_interval)

		num_of_classes = 3#4
		
		
		

		N_FCL = 5
		N_IN  = 2*self.num_features
		N_OUT = num_of_classes
		# self.fc_layers = []
		# for i in range(N_FCL):
		# 	cur_layer_in = int(math.ceil(N_IN - i*(N_IN - N_OUT)/N_FCL))
		# 	cur_layer_out = int(math.ceil(N_IN - (i + 1)*(N_IN - N_OUT)/N_FCL))
		# 	self.fc_layers.append(Linear(cur_layer_in, cur_layer_out))
		#self.bn_layer = nn.BatchNorm1d(2*self.num_features)
		self.fc_layer0 = Linear(2*self.num_features, int(1.5*self.num_features))
		self.fc_layer1 = Linear(int(1.5*self.num_features), int(0.8*self.num_features))
		self.fc_layer2 = Linear(int(0.8*self.num_features), int(0.4*self.num_features))
		self.fc_layer3 = Linear(int(0.4*self.num_features), int(0.4*self.num_features))
		self.fc_layer4 = Linear(int(0.4*self.num_features), int(0.4*self.num_features))
		self.fc_layer5 = Linear(int(0.4*self.num_features), int(0.4*self.num_features))
		self.fc_layer6 = Linear(int(0.4*self.num_features), num_of_classes)

		#self.fc_first_layer = Linear(2*self.num_features, self.num_features)

		#self.class_head = Linear(2*self.num_features, num_of_classes)

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze

	def forward(self, input_images, p=0.5):
		# base_image = input_images[:,0]
		# ref_images = input_images[:,1]

		# flattening all batches, because we are going to just calculate embeddings on them which is the same 
		# operation for all inputs images
		# current input structure: batch, collection, channel, x, y
		unpacked_input = input_images.view(input_images.shape[0] * input_images.shape[1], input_images.shape[2], input_images.shape[3], input_images.shape[4])
		
		# calculate image embeddings
		base_embedding_a = self.base_net(unpacked_input)
		base_embedding_b = F.leaky_relu(base_embedding_a)

		# get back to division to batches, since we want to continue separate calculation per batch
		packed_embs_a = base_embedding_b.view(input_images.shape[0], input_images.shape[1], -1) #base_embedding_d


		packed_embs_b = packed_embs_a.view(packed_embs_a.shape[0], packed_embs_a.shape[1]*packed_embs_a.shape[2])
		
		#packed_embs_b = Dropout(p)(packed_embs_b)

		#packed_embs_b = self.fc_first_layer(packed_embs_b)
		#packed_embs_b = F.leaky_relu(packed_embs_b)
		#packed_embs_b = Dropout(p)(packed_embs_b)

		
		# import pdb
		# pdb.set_trace()

		#classification_logits = self.class_head(packed_embs_b)

		fc_res = packed_embs_b
		# # N_FCL = 5
		# # for i in range(N_FCL):
		# # 	fc_res = self.fc_layers[i](fc_res)
		# #fc_res = self.bn_layer(fc_res)
		fc_res = self.fc_layer0(fc_res)	
		# #fc_res = F.normalize(fc_res, dim=1, p=2)
		fc_res = F.leaky_relu(fc_res)
		c_res = Dropout(p)(fc_res)
		fc_res = self.fc_layer1(fc_res)	
		fc_res = F.leaky_relu(fc_res)
		fc_res = self.fc_layer2(fc_res)
		fc_res = F.leaky_relu(fc_res)	
		fc_res = self.fc_layer3(fc_res)
		fc_res = F.leaky_relu(fc_res)	
		fc_res = self.fc_layer4(fc_res)
		fc_res = F.leaky_relu(fc_res)	
		fc_res = self.fc_layer5(fc_res)
		fc_res = F.leaky_relu(fc_res)	
		fc_res = self.fc_layer6(fc_res)
		#fc_res = F.relu(fc_res)		
		classification_logits = fc_res


		_, preds = torch.max(classification_logits, 1)

		return classification_logits, preds



class DiffModelConfigType:
    MlpHead = 0
    AddedEmbeddingAndMlpHead = 1 
    AddedEmbeddingAndMlpHeadWithDiffHead = 2





class RangeClassificationDeepModel(nn.Module):
	DEEP2 = True
	def __init__(self, 
					age_interval,
					min_age,
					max_age,
					age_radius=3, 
					device=torch.device("cuda:0"),
					config_type=DiffModelConfigType.AddedEmbeddingAndMlpHeadWithDiffHead, #MlpHead, 
					added_embed_layer_size=512, 
					diff_embed_layer_size=512):
		super(RangeClassificationDeepModel, self).__init__()
		self.device = device
		self.age_interval = age_interval
		self.min_age = min_age
		self.max_age = max_age
		self.age_radius = age_radius
		self.num_references = 1

		self.config_type = config_type 

		# num_features = 512
		# self.base_net = InceptionResnetV1(pretrained='vggface2')
		# self.base_net = FeatureExtractionResnet34()

		
		self.num_features = 4096
		self.num_features_emb_head = added_embed_layer_size # 1024 #1024
		self.num_features_diff_emb = diff_embed_layer_size
		self.base_net = FeatureExtractionVgg16(min_age=min_age, max_age=max_age, age_interval=age_interval)

		self.embedding_head_layer = Linear(self.num_features, self.num_features_emb_head) #2*self.num_features_emb_head) #
		#self.embedding_head_layer2 = Linear(2*self.num_features_emb_head, self.num_features_emb_head)
		if self.DEEP2:
			self.diff_embedding_layer =  Linear(2*self.num_features_emb_head, int(1.5*self.num_features_diff_emb))		
			self.diff_embedding_layer2 =  Linear(int(1.5*self.num_features_diff_emb), self.num_features_diff_emb)
			#self.diff_embedding_layer =  Linear(2*self.num_features_emb_head, int(1.7*self.num_features_diff_emb))		
			#self.diff_embedding_layer2 =  Linear(int(1.7*self.num_features_diff_emb), int(1.5*self.num_features_diff_emb))
			#self.diff_embedding_layer3 =  Linear(int(1.5*self.num_features_diff_emb),  int(self.num_features_diff_emb))
		else:
			self.diff_embedding_layer =  Linear(2*self.num_features_emb_head, self.num_features_diff_emb)

		# self.embedding_head_layers = []
		# for i in range(self.num_references+1):
		# 	self.embedding_head_layers.append(Linear(self.num_features, self.num_features_emb_head))
		
		if self.config_type == DiffModelConfigType.MlpHead:
			k = self.num_features #4096
			self.fc_first_stage = Linear((self.num_references+1)*self.num_features, k) # num_features_emb_head
		elif self.config_type == DiffModelConfigType.AddedEmbeddingAndMlpHead:
			k = self.num_features_emb_head #1024 
			self.fc_first_stage = Linear((self.num_references+1)*self.num_features_emb_head, k) # num_features_emb_head
		elif self.config_type == DiffModelConfigType.AddedEmbeddingAndMlpHeadWithDiffHead:
			k = self.num_features_diff_emb #1024 
			self.fc_first_stage = Linear(self.num_references*self.num_features_diff_emb, k) # num_features_emb_head
		else:
			print("bad config type")
			exit()

		#self.class_head = Linear(k, 2*age_radius+2)

		num_of_classes = 3#4

		
		self.class_head = Linear(k, num_of_classes)

		# m = 2048
		# self.fc_second_stage = Linear(k, m)

		# self.class_head = Linear(m, 2*age_radius+2)

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze

		# for param in self.base_net2.parameters():
		# 	param.requires_grad = not should_freeze

	def forward(self, input_images, p=0.5):
		# base_image = input_images[:,0]
		# ref_images = input_images[:,1]

		# flattening all batches, because we are going to just calculate embeddings on them which is the same 
		# operation for all inputs images
		# current input structure: batch, collection, channel, x, y
		unpacked_input = input_images.view(input_images.shape[0] * input_images.shape[1], input_images.shape[2], input_images.shape[3], input_images.shape[4])
        
		# calculate image embeddings
		base_embedding_a = self.base_net(unpacked_input)
		base_embedding_b = F.leaky_relu(base_embedding_a)
		#base_embedding_b = Dropout(p)(base_embedding_b)
		if self.config_type == DiffModelConfigType.MlpHead:
			# get back to division to batches, since we want to continue separate calculation per batch
			packed_embs_a = base_embedding_b.view(input_images.shape[0], input_images.shape[1], -1) #base_embedding_d
		elif self.config_type == DiffModelConfigType.AddedEmbeddingAndMlpHead:
			# additional embedding layer ("head layer") for dimensionality reduction (trades off with amount of references possible)
			base_embedding_c = self.embedding_head_layer(base_embedding_b)
			base_embedding_d = F.leaky_relu(base_embedding_c)
			# get back to division to batches, since we want to continue separate calculation per batch
			packed_embs_a = base_embedding_d.view(input_images.shape[0], input_images.shape[1], -1) #base_embedding_d
		elif self.config_type == DiffModelConfigType.AddedEmbeddingAndMlpHeadWithDiffHead:
			# additional embedding layer ("head layer") for dimensionality reduction (trades off with amount of references possible) 
			base_embedding_c = self.embedding_head_layer(base_embedding_b)
			base_embedding_f = F.leaky_relu(base_embedding_c)
			#base_embedding_c1 = F.leaky_relu(base_embedding_c)
			#base_embedding_d = Dropout(p)(base_embedding_c1)
			#base_embedding_e = self.embedding_head_layer2(base_embedding_c1)
			#base_embedding_e1 = F.leaky_relu(base_embedding_e)
			#base_embedding_f = Dropout(p)(base_embedding_e1)
			
			####################################################
			#			OLD WAY - UNROLLED 
			####################################################
			# get back to division to batches, since we want to continue separate calculation per batch
			packed_embs_a_pre = base_embedding_f.view(input_images.shape[0], input_images.shape[1], -1) #base_embedding_e
			#packed_embs_a_pre = Dropout(p)(packed_embs_a_pre)
			# diff_embs_list = []
			# query_repeated = packed_embs_a_pre[:][0].repeat((self.num_references,1))
			# for i in range(input_images.shape[0]):
			# 	# import pdb
			# 	# pdb.set_trace()
			# 	# repeat the query embedding Qe
			# 	query_repeated = packed_embs_a_pre[i][0].repeat((self.num_references,1))
			# 	# create pairs of (Qe, Re[j]), for each Re[j] in the references embeddings, and flatten each pair
			# 	flattened_query_ref_pairs = torch.stack((query_repeated, packed_embs_a_pre[i][1:]), dim=1).view(self.num_references, 2*self.num_features_emb_head)
			# 	diff_embeddings = self.diff_embedding_layer(flattened_query_ref_pairs)
			# 	if self.DEEP2:
			# 		# diff_embeddings_a = F.leaky_relu(diff_embeddings)
			# 		# diff_embeddings_b = self.diff_embedding_layer2(diff_embeddings_a)
			# 		# diff_embeddings_b1 = F.leaky_relu(diff_embeddings_b)
			# 		# diff_embeddings_b2 = self.diff_embedding_layer3(diff_embeddings_b1)
			# 		diff_embeddings_b3 = F.leaky_relu(diff_embeddings)
			# 		diff_embeddings_b4 = self.diff_embedding_layer2(diff_embeddings_b3)
			# 		diff_embeddings_c = F.leaky_relu(diff_embeddings_b4)
			# 	else:
			# 		diff_embeddings_c = F.leaky_relu(diff_embeddings) #(diff_embeddings_b)
			# 	diff_embs_list.append(diff_embeddings_c) #(diff_embeddings_a)
			# # import pdb
			# # pdb.set_trace()
			# # Tensorize the list of all batches
			# packed_embs_a = torch.stack(diff_embs_list)

			####################################################
			#			NEW WAY - BATCH-PARALLEL (MORE CORRECT) 
			####################################################

			# repeat the query embedding Qe
			# import pdb
			# pdb.set_trace()
			query_reshaped = packed_embs_a_pre[:, 0].view(packed_embs_a_pre.shape[0], 1, packed_embs_a_pre.shape[2])
			query_repeated = query_reshaped.repeat((1, self.num_references, 1))
			# create pairs of (Qe, Re[j]), for each Re[j] in the references embeddings, and flatten each pair
			flattened_query_ref_pairs = torch.stack((query_repeated, packed_embs_a_pre[:, 1:]), dim=2).view(packed_embs_a_pre.shape[0], self.num_references, 2*self.num_features_emb_head)
			diff_embeddings = self.diff_embedding_layer(flattened_query_ref_pairs)
			if self.DEEP2:
				# diff_embeddings_a = F.leaky_relu(diff_embeddings)
				# diff_embeddings_b = self.diff_embedding_layer2(diff_embeddings_a)
				# diff_embeddings_b1 = F.leaky_relu(diff_embeddings_b)
				# diff_embeddings_b2 = self.diff_embedding_layer3(diff_embeddings_b1)
				diff_embeddings_b3 = F.leaky_relu(diff_embeddings)
				diff_embeddings_b4 = self.diff_embedding_layer2(diff_embeddings_b3)
				# diff_embeddings_b5 = F.leaky_relu(diff_embeddings_b4)
				# diff_embeddings_b6 = self.diff_embedding_layer3(diff_embeddings_b5)
				# diff_embeddings_c = F.leaky_relu(diff_embeddings_b6)
				diff_embeddings_c = F.leaky_relu(diff_embeddings_b4)
			else:
				diff_embeddings_c = F.leaky_relu(diff_embeddings) #(diff_embeddings_b)

			packed_embs_a = diff_embeddings_c
		else:
			print("bad config type")
			exit()

		packed_embs_b = packed_embs_a.view(packed_embs_a.shape[0], packed_embs_a.shape[1]*packed_embs_a.shape[2])
		#packed_embs_b = Dropout(p)(packed_embs_b)
		# import pdb
		# pdb.set_trace()

		#repacked_input = base_embedding.view(input_images.shape[0], input_images.shape[1], input_images.shape[2], input_images.shape[3], input_images.shape[4])

		# Identical logic:
		# base_embedding1 = self.base_net1(base_image)
		# base_embedding1 = F.leaky_relu(base_embedding1)
		# base_embedding2 = self.base_net1(ref_images)
		# base_embedding2 = F.leaky_relu(base_embedding2)
		
		#base_embedding1 = F.relu(base_embedding1)
		# base_embedding = Dropout(p)(base_embedding)

		#base_embedding2 = self.base_net2(ref_images)

		
		
		#base_embedding2 = F.relu(base_embedding2)
		# base_embedding = Dropout(p)(base_embedding)

		# import pdb
		# pdb.set_trace()
		#base_embedding = torch.cat((base_embedding1, base_embedding2), dim=1)

		#base_embedding = #torch.cat(base_embedding1, dim=1)

		
		packed_embs_b = Dropout(p)(packed_embs_b)
		
		# first stage
		class_embed = self.fc_first_stage(packed_embs_b)
		#class_embed = F.normalize(class_embed, dim=1, p=2)
		#x = F.relu(class_embed)
		x = F.leaky_relu(class_embed)
		#x = Dropout(p)(x)

		# x = self.fc_second_stage(x)

		# x = F.leaky_relu(x)
		classification_logits = self.class_head(x)

		_, preds = torch.max(classification_logits, 1)

		return classification_logits, preds