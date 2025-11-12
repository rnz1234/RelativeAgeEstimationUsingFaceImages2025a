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

import dp7_config as cfg


DEEP_HEAD = False

class FeatureExtractionVgg16(nn.Module):
	def __init__(self):
		super(FeatureExtractionVgg16, self).__init__()

		# self.base_net = models.vgg16(pretrained=True)
		age_interval = 1
		min_age = 15
		max_age = 80
		num_classes = int((max_age - min_age) / age_interval + 1)
		
		recog_model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age) #ArcMarginClassifier(10976)
		#pretrained_model_path = '../Common/Weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64'
		#pretrained_model_path = '../Common/Weights/Morph2/transformer/mae2.56'
		#pretrained_model_path = 'weights/Morph2Diff/unified/iter/from_my_host'
		#pretrained_model_path = 'weights/Morph2Diff/unified/iter/time_07_07_2023_01_13_05'
		pretrained_model_path = '../Common/Weights/bjj1_time_07_07_2023_01_13_05'
		pretrained_model_file = os.path.join(pretrained_model_path, "weights.pt")
		recog_model.load_state_dict(torch.load(pretrained_model_file), strict=False)

		self.base_net = recog_model.base_net

		#self.base_net.classifier = self.base_net.classifier[:6]

	def forward(self, input):
		# x = self.base_net.features(input)
		# x = self.base_net.avgpool(x)
		# x = torch.flatten(x, 1)
		# x = self.base_net.classifier(x)
		x = self.base_net(input)

		return x


class AgeDiffRegAndClsModel(nn.Module):
	def __init__(self, 
					age_diff_interval,
					min_age_diff, 
					max_age_diff,
					num_classes,
					device=torch.device("cuda:0"), 
					num_references=1, 
					added_embed_layer_size=512, 
					diff_embed_layer_size=512,
					fc_2nd_layer_size=256, 
					agg_type="fc"): #="maxpool"):
		super(AgeDiffRegAndClsModel, self).__init__()
		self.device = device
		self.age_diff_interval = age_diff_interval
		self.min_age_diff = min_age_diff
		self.max_age_diff = max_age_diff
		self.num_references = num_references
		self.agg_type = agg_type

		# num_features = 512
		# self.base_net = InceptionResnetV1(pretrained='vggface2')
		# self.base_net = FeatureExtractionResnet34()
		
		self.num_features = 4096
		self.num_features_emb_head = added_embed_layer_size # 1024 #1024
		self.num_features_diff_emb = diff_embed_layer_size
		self.fc_2nd_layer_size = fc_2nd_layer_size
		self.base_net = FeatureExtractionVgg16()

		self.embedding_head_layer = Linear(self.num_features, self.num_features_emb_head)
		self.diff_embedding_layer =  Linear(2*self.num_features_emb_head, self.num_features_diff_emb)
		
		self.aggregation_stage = nn.MaxPool1d(self.num_references, self.num_references)
		self.fc_first_stage = Linear(self.num_references*self.num_features_diff_emb, self.num_features_diff_emb)
		self.fc_second_stage = Linear(self.num_features_diff_emb, self.fc_2nd_layer_size)
		# self.fc_second_stage1 = Linear(self.fc_2nd_layer_size, int(0.75*self.fc_2nd_layer_size))
		# self.fc_second_stage2 = Linear(self.fc_2nd_layer_size, int(0.5*self.fc_2nd_layer_size))
		# self.fc_second_stage3 = Linear(self.fc_2nd_layer_size, int(0.25*self.fc_2nd_layer_size))
		# self.class_head = Linear(0.25*self.fc_2nd_layer_size, num_classes)

		if cfg.IS_ORDINAL:
			self.class_head = Linear(self.fc_2nd_layer_size, num_classes-1)
		else:
			self.class_head = Linear(self.fc_2nd_layer_size, num_classes)



		self.labels = range(num_classes)
		

		self.regression_heads = []
		# self.classification_heads = []
		self.centers = []
		for i in self.labels:
			self.regression_heads.append(Linear(self.fc_2nd_layer_size, 1))
			# self.classification_heads.append(Linear(k, int(self.age_intareval*2)))
			center = min_age_diff + 0.5 * age_diff_interval + i * age_diff_interval
			self.centers.append(center)

		self.regression_heads = nn.ModuleList(self.regression_heads)

		

		
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
		#base_embedding_b = Dropout(p)(base_embedding_b)
		
		# additional embedding layer ("head layer") for dimensionality reduction (trades off with amount of references possible) 
		base_embedding_c = self.embedding_head_layer(base_embedding_b)
		base_embedding_f = F.leaky_relu(base_embedding_c)
		#base_embedding_f = Dropout(p)(base_embedding_f)
		
		# get back to division to batches, since we want to continue separate calculation per batch
		packed_embs_a_pre = base_embedding_f.view(input_images.shape[0], input_images.shape[1], -1) #base_embedding_e
		
		# repeat the query embedding Qe
		# import pdb
		# pdb.set_trace()
		query_reshaped = packed_embs_a_pre[:, 0].view(packed_embs_a_pre.shape[0], 1, packed_embs_a_pre.shape[2])
		query_repeated = query_reshaped.repeat((1, self.num_references, 1))
		# create pairs of (Qe, Re[j]), for each Re[j] in the references embeddings, and flatten each pair
		flattened_query_ref_pairs = torch.stack((query_repeated, packed_embs_a_pre[:, 1:]), dim=2).view(packed_embs_a_pre.shape[0], self.num_references, 2*self.num_features_emb_head)
		diff_embeddings = self.diff_embedding_layer(flattened_query_ref_pairs)
		
		diff_embeddings_c = F.leaky_relu(diff_embeddings) #(diff_embeddings_b)
		#diff_embeddings_c = Dropout(p)(diff_embeddings_c)

		if self.agg_type == "fc":
			packed_embs_b = diff_embeddings_c.view(diff_embeddings_c.shape[0], diff_embeddings_c.shape[1]*diff_embeddings_c.shape[2])
			x = self.fc_first_stage(packed_embs_b)
			x = F.leaky_relu(x)
		elif self.agg_type == "maxpool":
			x = self.aggregation_stage(diff_embeddings_c.transpose(1,2))
			x = x.view(x.shape[0], x.shape[1])
			x = Dropout(p)(x)
		
		x = self.fc_second_stage(x)
		x = F.leaky_relu(x)
		# x = self.fc_second_stage1(x)
		# x = F.leaky_relu(x)
		# x = self.fc_second_stage2(x)
		# x = F.leaky_relu(x)
		# x = self.fc_second_stage3(x)
		# x = F.leaky_relu(x)
		#x = Dropout(p)(x)

		classification_logits = self.class_head(x)

		if cfg.IS_ORDINAL:
			weights = ordinal_softmax(classification_logits, device=self.device).float()
		else:
			weights = nn.Softmax()(classification_logits)

		# import pdb
		# pdb.set_trace()

		t = []
		for i in self.labels:
			# t.append(torch.squeeze(self.regression_heads[i](x)) * weights[:, i])
			t.append(torch.squeeze(self.regression_heads[i](x) + self.centers[i]).to(self.device) * weights[:, i])
			# _, local_res = torch.max(self.classification_heads[i](x), 1)
			# t.append(torch.squeeze(local_res - int(self.age_intareval*2) / 2 + self.centers[i]) * weights[:, i])


		age_diff_pred = torch.stack(t, dim=0).sum(dim=0) / torch.sum(weights, dim=1)


		

		return classification_logits, age_diff_pred