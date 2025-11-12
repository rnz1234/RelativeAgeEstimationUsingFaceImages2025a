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


class FeatureExtractionVgg16(nn.Module):
	def __init__(self, min_age, max_age, age_interval):
		super(FeatureExtractionVgg16, self).__init__()

		# self.base_net = models.vgg16(pretrained=True)
		age_interval = 1
		min_age = 15
		max_age = 80
		num_classes = int((max_age - min_age) / age_interval + 1)
		
		recog_model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age) #ArcMarginClassifier(10976)
		#pretrained_model_path = 'weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64'
		pretrained_model_path = '../Common/Weights/Morph2/transformer/mae2.56'
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



class DiffPipeline3Model(nn.Module):
	def __init__(self, 
					age_interval,
					min_age,
					max_age,
					age_radius=3, 
					device=torch.device("cuda:0")):
		super(DiffPipeline3Model, self).__init__()

		self.device = device
		self.age_interval = age_interval
		self.age_radius = age_radius

		# num_features = 512
		# self.base_net = InceptionResnetV1(pretrained='vggface2')
		# self.base_net = FeatureExtractionResnet34()

		self.num_features = 4096
		self.base_net = FeatureExtractionVgg16(min_age=min_age, max_age=max_age, age_interval=age_interval)

		num_of_classes = 2*age_radius

		self.fc_first_layer = Linear(2*self.num_features, self.num_features)

		self.class_head = Linear(self.num_features, num_of_classes)

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
		
		packed_embs_b = Dropout(p)(packed_embs_b)

		packed_embs_b = self.fc_first_layer(packed_embs_b)
		classification_logits = self.class_head(packed_embs_b)

		
		# import pdb
		# pdb.set_trace()
		age_diff_pred_hard = logits_to_label(classification_logits).float()
		#age_diff_pred_hard = age_diff_pred_hard - self.age_radius
		predicted_probs = ordinal_softmax(classification_logits, device=self.device).float()
		age_diff_pred_soft = torch.matmul(predicted_probs.to(self.device), torch.Tensor(list(range(-self.age_radius,self.age_radius+1))).to(self.device))
		
		return classification_logits, age_diff_pred_hard, age_diff_pred_soft