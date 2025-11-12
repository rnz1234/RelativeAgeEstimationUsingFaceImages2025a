##############################################################################
#	Project		:	Age Estimation
#	Pipeline	:	E2ePipeline3
#	Date		:	28.10.2023
# 	Description	: 	Models file. Provides relevant pytorch-based models
#					For the task.
##############################################################################

import os

import torch
from torch import nn as nn
from torch.nn import Linear, functional as F, Dropout
import torchvision.models as models
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights
import timm

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

from ep3_config import USE_GENDER, USE_DIFF_CLS_AND_REG, DIST_APPROX_METHOD, ERROR_SAT_RANGE_MIN, ERROR_SAT_RANGE_MAX


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
	
class BaseRecogClassifier(nn.Module):
	def __init__(self):
		super(BaseRecogClassifier, self).__init__()

		self.base_net = models.vgg16(pretrained=True)
		num_features = self.base_net.classifier[6].in_features
		# Add a new fully connected layer with the desired number of output units
		del self.base_net.classifier[1:]
		
		self.BaseEmbAct = nn.LeakyReLU()
		self.DropoutLayer = nn.Dropout(p=0.5)
		self.FaceBatchNorm1d = nn.BatchNorm1d(num_features, affine=False)

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze

	def forward(self, input):
		BaseEmbedding = self.base_net(input)
		IdEmbed = self.BaseEmbAct(BaseEmbedding)
		IdEmbed = self.FaceBatchNorm1d(IdEmbed)
		output = self.DropoutLayer(IdEmbed)
		
		return output
	
class BaseRecogClassifier2(nn.Module):
	def __init__(self):
		super(BaseRecogClassifier2, self).__init__()

		self.base_net = models.vgg16(pretrained=True)
		num_features = 84*7*7 
		self.AdaptiveAvgPoolLayer = nn.AdaptiveAvgPool2d(output_size=(7, 7))
		self.ReducLayer = nn.Conv2d(in_channels=512, out_channels=84, kernel_size=1)
		self.FlatLayer = nn.Flatten()
		
		self.BaseEmbAct = nn.LeakyReLU()
		self.DropoutLayer = nn.Dropout(p=0.5)
		self.FaceBatchNorm1d = nn.BatchNorm1d(num_features, affine=False)


	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze

	def forward(self, input_images, device=torch.device("cuda:0")):
		PreBaseEmbedding = self.base_net.features(input_images)

		PreBaseEmbedding1 = self.AdaptiveAvgPoolLayer(PreBaseEmbedding)
		PreBaseEmbedding2 = self.ReducLayer(PreBaseEmbedding1)
		BaseEmbedding = self.FlatLayer(PreBaseEmbedding2)

		IdEmbed = self.BaseEmbAct(BaseEmbedding)
		IdEmbed = self.FaceBatchNorm1d(IdEmbed)
		output = self.DropoutLayer(IdEmbed)
		
		return output
	
class BaseRecogClassifierWrapper(nn.Module):
	def __init__(self,
			  	pretrained_model_path,
				pretrained_model_file_name="weights.pt",
				load_pretrained=False):
		super(BaseRecogClassifierWrapper, self).__init__()

		#self.base_net = BaseRecogClassifier2()
		self.base_net = BaseRecogClassifier()

		self.pretrained_model_path = pretrained_model_path
		self.pretrained_model_file_name = pretrained_model_file_name
		if load_pretrained:
			pretrained_model_file = os.path.join(pretrained_model_path, pretrained_model_file_name)
			self.base_net.load_state_dict(torch.load(pretrained_model_file), strict=False)

	def freeze_base_cnn(self, should_freeze=True):
		self.base_net.freeze_base_cnn(should_freeze)

	def forward(self, input):
		return self.base_net(input)
	



class ViTRecogClassifier(nn.Module):
	def __init__(self):
		super(ViTRecogClassifier, self).__init__()

		# Load pretrained Vision Transformer
		weights = ViT_B_16_Weights.IMAGENET1K_V1
		self.base_net = vit_b_16(weights=weights, dropout=0.3)
		#self.base_net = timm.create_model(
		# 	'vit_base_patch16_224',
		# 	pretrained=True,
		# 	features_only=True,
		# 	drop_rate=0.2,
		# 	attn_drop_rate=0.1,
		# 	drop_path_rate=0.1,
		# )

        # weights = ViT_L_16_Weights.IMAGENET1K_V1
        # self.base_net = vit_l_16(weights=weights)


		# Remove the classification head to get feature embeddings
		self.base_net.heads = nn.Identity()

		# Get the number of output features
		self.num_features = self.base_net.hidden_dim  # usually 768

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze

	def forward(self, input):
		base_embedding = self.base_net(input)
		return base_embedding  # shape: [batch_size, 768]
	
class ConvNeXtRecogClassifier(nn.Module):
    def __init__(self):
        super(ConvNeXtRecogClassifier, self).__init__()
        
        # Create ConvNeXt base backbone without classifier head
        self.base_net = timm.create_model(
            'convnext_base',
            pretrained=True,
            num_classes=0  # removes classifier head, outputs pooled features
        )
        
        # Output feature size of convnext_base is 1024
        self.num_features = 1024
    
    def freeze_base_cnn(self, should_freeze=True):
        for param in self.base_net.parameters():
            param.requires_grad = not should_freeze

    def forward(self, input):
        # input shape: (B, C, H, W)
        base_embedding = self.base_net(input)  # output shape: (B, 1024)
        return base_embedding
	
class EfficientNetV2MRecogClassifier(nn.Module):
    def __init__(self):
        super(EfficientNetV2MRecogClassifier, self).__init__()
        
        # Create EfficientNetV2-M backbone without classifier head
        self.base_net = timm.create_model(
            'tf_efficientnetv2_m', #'tf_efficientnetv2_m', 'tf_efficientnetv2_s'
            pretrained=True,
            num_classes=0  # removes classifier head, outputs pooled features
        )
        
        # Output feature size for tf_efficientnetv2_m is 1280
        self.num_features = 1280
    
    def freeze_base_cnn(self, should_freeze=True):
        for param in self.base_net.parameters():
            param.requires_grad = not should_freeze

    def forward(self, input):
        # input shape: (B, C, H, W)
        base_embedding = self.base_net(input)  # output shape: (B, 1280)
        return base_embedding


class ResNet51QBackbone(nn.Module):
    def __init__(self):
        super(ResNet51QBackbone, self).__init__()
        
        # Load resnet51q from timm with no classifier head
        self.base_net = timm.create_model(
            'resnet51q',
            pretrained=True,
            num_classes=0  # removes classifier, outputs pooled feature vector
        )
        
        # Number of output features after global pooling
        self.num_features = self.base_net.num_features  # typically 2048

    def freeze_base_cnn(self, should_freeze=True):
        for param in self.base_net.parameters():
            param.requires_grad = not should_freeze

    def forward(self, input):
        # Input: [B, 3, H, W], Output: [B, num_features]
        return self.base_net(input)

	
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
					load_pretrained,
					dropout_p=0.5,
					num_of_fc_layers=3,
					age_embedding_size=128,
					is_ordinal=True,
					min_age_diff=-35,
					max_age_diff=35,
					num_classes_diff=71,
					age_diff_interval=1,
					regressors_diff_head=False,
					fc_head_base_layer_size=2048,
					use_vit=False,
					use_convnext=False,
					use_efficientnet=False,
					use_resnet51q=False
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
		self.fc_head_base_layer_size = fc_head_base_layer_size
		self.use_vit = use_vit
		self.use_convnext = use_convnext
		self.use_efficientnet = use_efficientnet
		self.use_resnet51q = use_resnet51q

		# feature extractor
		# self.base_net = FeatureExtractionVgg16(
		# 	min_age=min_age,
		# 	max_age=max_age,
		# 	age_interval=age_interval,
		# 	pretrained_model_path=pretrained_model_path,
		# 	pretrained_model_file_name=pretrained_model_file_name
		# )
		# 
		# self.embeddings_size = self.base_net.base_net.classifier[3].out_features

		if self.use_vit:
			self.base_net = ViTRecogClassifier()
			self.embeddings_size = self.base_net.num_features
		elif self.use_convnext:
			self.base_net = ConvNeXtRecogClassifier()
			self.embeddings_size = self.base_net.num_features
		elif self.use_efficientnet:
			self.base_net = EfficientNetV2MRecogClassifier()
			self.embeddings_size = self.base_net.num_features
		elif self.use_resnet51q:
			self.base_net = ResNet51QBackbone()
			self.embeddings_size = self.base_net.num_features
		else:
			self.base_net = BaseRecogClassifierWrapper(
				pretrained_model_path=pretrained_model_path,
				pretrained_model_file_name=pretrained_model_file_name,
				load_pretrained=load_pretrained
			)

			self.embeddings_size = self.base_net.base_net.FaceBatchNorm1d.num_features
		
		self.base_act = nn.LeakyReLU()
		self.base_dropout = nn.Dropout(self.dropout_p)

		self.fc_weight_head = [Linear(2*(self.embeddings_size+self.age_embedding_size), self.fc_head_base_layer_size)]
		self.fc_weight_head.append(nn.LeakyReLU())
		self.fc_weight_head.append(nn.Dropout(self.dropout_p))
		self.fc_diff_head = [Linear(2*(self.embeddings_size+self.age_embedding_size), self.fc_head_base_layer_size)]
		self.fc_diff_head.append(nn.LeakyReLU())
		self.fc_diff_head.append(nn.Dropout(self.dropout_p))
		for i in range(self.num_of_fc_layers-1):
			if i == self.num_of_fc_layers-2:
				self.fc_weight_head.append(Linear(int(self.fc_head_base_layer_size/(2**i)), 1))
				#self.fc_diff_head.append(Linear(int(self.fc_head_base_layer_size/(2**i)), 1))
			else:
				self.fc_weight_head.append(Linear(int(self.fc_head_base_layer_size/(2**i)), int(self.fc_head_base_layer_size/(2**(i+1)))))
				self.fc_weight_head.append(nn.LeakyReLU())
				self.fc_weight_head.append(nn.Dropout(self.dropout_p))
				self.fc_diff_head.append(Linear(int(self.fc_head_base_layer_size/(2**i)), int(self.fc_head_base_layer_size/(2**(i+1)))))
				self.fc_diff_head.append(nn.LeakyReLU())
				self.fc_diff_head.append(nn.Dropout(self.dropout_p))
		self.fc_weight_head = nn.Sequential(*self.fc_weight_head) #nn.Sequential(*self.fc_weight_head)
		self.fc_diff_head = nn.Sequential(*self.fc_diff_head) #nn.Sequential(*self.fc_diff_head)

		self.age_embedding = nn.Embedding(num_embeddings=max_age-min_age, embedding_dim=self.age_embedding_size)

		
		last_layer_width = int(self.fc_head_base_layer_size/(2**(self.num_of_fc_layers-2)))
	
		#self.batch_norm_layer_w = nn.BatchNorm1d(last_layer_width)
		#self.batch_norm_layer_d = nn.BatchNorm1d(last_layer_width)
		#self.batch_norm_layer_base = nn.BatchNorm1d(self.embeddings_size+self.age_embedding_size, eps=0.4)

		if self.is_ordinal:
			self.class_head = Linear(last_layer_width, num_classes_diff-1)
		else:
			self.class_head = Linear(last_layer_width, num_classes_diff)

		self.class_head_diff_main = Linear(last_layer_width, num_classes_diff)

		# mixture of experts regression for the diff
		self.labels = range(num_classes_diff)

		self.regression_diff_main_heads = []
		# self.classification_heads = []
		self.centers_diff_main = []
		for i in self.labels:
			self.regression_diff_main_heads.append(Linear(num_classes_diff, 1))
			# self.classification_heads.append(Linear(k, int(self.age_intareval*2)))
			centers_diff_main = min_age_diff + 0.5 * age_diff_interval + i * age_diff_interval
			self.centers_diff_main.append(centers_diff_main)

		self.regression_diff_main_heads = nn.ModuleList(self.regression_diff_main_heads)
		
		# "side car" mixture of experts diff regression head
		if self.regressors_diff_head:
			self.regression_heads = []
			# self.classification_heads = []
			self.centers = []
			for i in self.labels:
				self.regression_heads.append(Linear(last_layer_width, 1))
				# self.classification_heads.append(Linear(k, int(self.age_intareval*2)))
				center = min_age_diff + 0.5 * age_diff_interval + i * age_diff_interval
				self.centers.append(center)

			self.regression_heads = nn.ModuleList(self.regression_heads)

		# additional head for gender classification
		if USE_GENDER:
			self.fc_gender_head = nn.Sequential(nn.LeakyReLU(),
									  		nn.Dropout(self.dropout_p),
											Linear(self.embeddings_size, 2048),
											nn.LeakyReLU(),
									  		nn.Dropout(self.dropout_p),
									  		Linear(2048, 1))
		

	# freeze backbone
	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze
	
	# Forward pass
	def forward(self, input_images, query_noisy_age, input_ref_ages): #
		# if input_images.shape[0] == 1:
		# 	import pdb
		# 	pdb.set_trace()
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
		fc_out_d_pre_reg = self.fc_diff_head(fc_out_d)

		

		fc_out_d_cls_logits = self.class_head_diff_main(fc_out_d_pre_reg)

		weights = nn.Softmax(dim=2)(fc_out_w) 						# dim=1

		weights_main_reg_diff_head_t = nn.Softmax(dim=3)(fc_out_d_cls_logits)
		
		
		if USE_DIFF_CLS_AND_REG:
			#print("-----------------")
			t = []
			for i in self.labels:
				t.append(torch.squeeze(self.regression_diff_main_heads[i](fc_out_d_cls_logits) + self.centers_diff_main[i], dim=(0,3)) * weights_main_reg_diff_head_t[:, :, :, i]) # .to(self.device)
				#t.append(torch.squeeze(self.regression_diff_main_heads[i](fc_out_d_cls_logits) + self.centers_diff_main[i], dim=(0,2,3)) * weights_main_reg_diff_head_t[:, :, :, i]) # .to(self.device)
				# if i == 0:
				# 	print((self.regression_diff_main_heads[i](fc_out_d_cls_logits) + self.centers_diff_main[i]).shape)
				# 	print(weights_main_reg_diff_head_t[:, :, :, i].shape)
				# 	print((torch.squeeze(self.regression_diff_main_heads[i](fc_out_d_cls_logits) + self.centers_diff_main[i], dim=(0,2,3)) * weights_main_reg_diff_head_t[:, :, :, i]).shape)
			#print("----------------")

			# if input_images.shape[0] == 1:
			# 	import pdb
			# 	pdb.set_trace()
			fc_out_d = torch.stack(t, dim=0).sum(dim=0) # / torch.sum(weights_main_reg_diff_head_t, dim=3)
			

			age_diffs = fc_out_d
		else:
			_, diff_preds_idx = torch.max(weights_main_reg_diff_head_t, 3)
			if DIST_APPROX_METHOD == "kde_based_saturated":
				age_diffs = diff_preds_idx + ERROR_SAT_RANGE_MIN
			else:
				age_diffs = diff_preds_idx + self.min_age - self.max_age			
		

		#############################################
		# Age head
		#############################################

		# ref_ages_tensor = torch.transpose(torch.stack(tuple(input_ref_ages)), 0, 1)
		ref_ages_tensor = input_ref_ages.reshape(input_ref_ages.shape[0], input_ref_ages.shape[1], 1)

		# print(age_diffs[0].shape)
		# print(age_diffs[1].shape)

		#age_estimations_f = ref_ages_tensor + age_diffs[0].view(age_diffs[0].shape[0], age_diffs[1].shape[1], 1)
		age_diffs_f = age_diffs[0].view(age_diffs[0].shape[0], age_diffs[1].shape[1], 1)
		#age_estimations_r = ref_ages_tensor - age_diffs[1].view(age_diffs[1].shape[0], age_diffs[1].shape[1], 1)
		age_diffs_r = age_diffs[1].view(age_diffs[1].shape[0], age_diffs[1].shape[1], 1)

		# print(ref_ages_tensor[:,0,:].shape)
		# print(weights[0].shape)
		# print(age_diffs_f.shape)
		# print((weights[0]*age_diffs_f).sum(dim=1).shape)

		#age_pred_f = (weights[0]*age_estimations_f).sum(dim=1) # / torch.sum(weights[0], dim=1) # seems no need for div - since each value is 1
		age_pred_f = ref_ages_tensor[:,0,:] + (weights[0]*age_diffs_f).sum(dim=1) # / torch.sum(weights[0], dim=1) # seems no need for div - since each value is 1
		#age_pred_r = (weights[1]*age_estimations_r).sum(dim=1) # / torch.sum(weights[1], dim=1) # seems no need for div - since each value is 1
		age_pred_r = ref_ages_tensor[:,0,:] - (weights[1]*age_diffs_r).sum(dim=1) # / torch.sum(weights[1], dim=1) # seems no need for div - since each value is 1
		
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

		# gender head
		if USE_GENDER:
			base_embedding_query_img_reshaped = base_embedding_a.view(input_images.shape[0], input_images.shape[1], -1)[:,0]
			gender_head_cls_pre_sigmoid = self.fc_gender_head(base_embedding_query_img_reshaped)

		#return age_pred.reshape(age_pred.shape[0], age_pred.shape[1]), age_diffs.reshape(age_diffs.shape[0], age_diffs.shape[1])
		if USE_GENDER:
			if self.regressors_diff_head:
				return age_pred_f.reshape(age_pred_f.shape[0], age_pred_f.shape[1]), \
						age_pred_r.reshape(age_pred_r.shape[0], age_pred_r.shape[1]), \
						age_diffs_f.reshape(age_diffs_f.shape[0], age_diffs_f.shape[1]), \
						age_diffs_r.reshape(age_diffs_r.shape[0], age_diffs_r.shape[1]), \
						classification_logits, age_diff_pred_cls, fc_out_d_cls_logits[0], fc_out_d_cls_logits[1], \
						gender_head_cls_pre_sigmoid
						#classification_logits, classification_logits_minus, age_diff_pred_cls
			else:
				return age_pred_f.reshape(age_pred_f.shape[0], age_pred_f.shape[1]), \
						age_pred_r.reshape(age_pred_r.shape[0], age_pred_r.shape[1]), \
						age_diffs_f.reshape(age_diffs_f.shape[0], age_diffs_f.shape[1]), \
						age_diffs_r.reshape(age_diffs_r.shape[0], age_diffs_r.shape[1]), \
						classification_logits, fc_out_d_cls_logits[0], fc_out_d_cls_logits[1], \
						gender_head_cls_pre_sigmoid
		else:
			if self.regressors_diff_head:
				return age_pred_f.reshape(age_pred_f.shape[0], age_pred_f.shape[1]), \
						age_pred_r.reshape(age_pred_r.shape[0], age_pred_r.shape[1]), \
						age_diffs_f.reshape(age_diffs_f.shape[0], age_diffs_f.shape[1]), \
						age_diffs_r.reshape(age_diffs_r.shape[0], age_diffs_r.shape[1]), \
						classification_logits, age_diff_pred_cls, fc_out_d_cls_logits[0], fc_out_d_cls_logits[1]
						#classification_logits, classification_logits_minus, age_diff_pred_cls
			else:
				return age_pred_f.reshape(age_pred_f.shape[0], age_pred_f.shape[1]), \
						age_pred_r.reshape(age_pred_r.shape[0], age_pred_r.shape[1]), \
						age_diffs_f.reshape(age_diffs_f.shape[0], age_diffs_f.shape[1]), \
						age_diffs_r.reshape(age_diffs_r.shape[0], age_diffs_r.shape[1]), \
						classification_logits, fc_out_d_cls_logits[0], fc_out_d_cls_logits[1]
						



class PerNoisyRangeAgeModel(nn.Module):
	def __init__(self, model_range1, model_range2, device, threshold=50):
		super(PerNoisyRangeAgeModel, self).__init__()
		self.model_range1 = model_range1
		self.model_range2 = model_range2
		self.device = device
		self.threshold = threshold

	def freeze_base_cnn(self, should_freeze=True):
		self.model_range1.freeze_base_cnn(should_freeze)
		self.model_range2.freeze_base_cnn(should_freeze)
		
	def forward(self, input_images, query_noisy_age, input_ref_ages):
		# TODO : need to do this vectorized (i.e. apply the right model per entry)
		mask = query_noisy_age > self.threshold
		# import pdb
		# pdb.set_trace()
		if len(query_noisy_age[mask]) == len(query_noisy_age):
			output = self.model_range1(input_images[mask], query_noisy_age[mask], input_ref_ages[mask])
		elif len(query_noisy_age[~mask]) == len(query_noisy_age):
			output = self.model_range2(input_images[~mask], query_noisy_age[~mask], input_ref_ages[~mask])
		else:
			output1 = self.model_range1(input_images[mask], query_noisy_age[mask], input_ref_ages[mask])
			output2 = self.model_range2(input_images[~mask], query_noisy_age[~mask], input_ref_ages[~mask])
			output_l = []
			for i in range(len(output1)):
				req_size = [query_noisy_age.shape[0]] + list(output1[i].shape)[1:]
				if output1[i].shape[0] > 0:
					cur_output_dtype = output1[i].dtype
				else:
					cur_output_dtype = output2[i].dtype
				cur_output = torch.zeros(*req_size).to(output1[i].device).to(cur_output_dtype)
				try:
					cur_output[mask] = output1[i]
					cur_output[~mask] = output2[i]
				except RuntimeError:
					import pdb
					pdb.set_trace()
				# if i == 0:
				# 	age_pred_f = cur_output
				# elif i == 1:
				# 	age_pred_r = cur_output
				# elif i == 2:
				# 	age_diff_preds_f = cur_output
				# elif i == 3:
				# 	age_diff_preds_r = cur_output
				# elif i == 4:
				# 	classification_logits = cur_output
				# elif i == 5:
				# 	classification_logits_main_diff = cur_output
				# elif i == 6:
				# 	gender_head_cls_pre_sigmoid = cur_output
				output_l.append(cur_output)

			output = tuple(output_l)
		
		return output
		#return age_pred_f, age_pred_r, age_diff_preds_f, age_diff_preds_r, classification_logits, classification_logits_main_diff, gender_head_cls_pre_sigmoid
		# if query_noisy_age > self.threshold:
		# 	return self.model_range1(input_images, query_noisy_age, input_ref_ages)
		# else:
		# 	return self.model_range2(input_images, query_noisy_age, input_ref_ages)