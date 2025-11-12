import json
import numpy as np
import torch
from PIL import Image
from Common.Pipelines.AgePredictBasicPipeline import get_age_transformer, get_age_diff_predictor
# importing the sys module
import sys        
 
# appending the directory of mod.py
# in the sys.path list
sys.path.append('../')   

#from global_config import cfg

SELECT_WITH_EMBS = False
EMBS_PERCENTILE = 10 #25
K_NN_EMBS = 3 #5
N_REFS = 1

from condor_pytorch.activations import ordinal_softmax


class SingleRefQuerier:
    def __init__(self, metadata, min_age, max_age):
        self.metadata = metadata
        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.metadata])
        self.min_age = min_age
        self.max_age = max_age

    def _find_ref_image(self, idx, query_age, ages, embs_trn_rsh, embs_vld_rsh):
        # round in order to get an actual age
        ref_age = query_age
        # automatically fix in case we are out of range
        if ref_age > self.max_age - 1:
            ref_age = self.max_age - 1
        elif ref_age < self.min_age + 1:
            ref_age = self.min_age + 1
        # get all idxs is ref age
        idxs = np.where(ages == ref_age)[0]

        if len(idxs) == 0:
            print("Not ref found - adding some small noise")
        
        # in case no refs found, no choice but to get other ref - for different diff 
        while len(idxs) == 0:
            # take a sample in radius around the original point
            ref_age = np.round(query_age + np.random.normal(0, 2))
            # automatically fix in case we are out of range
            if ref_age > self.max_age - 1:
                ref_age = self.max_age - 1
            elif ref_age < self.min_age + 1:
                ref_age = self.min_age + 1
            idxs = np.where(ages == ref_age)[0]

        if SELECT_WITH_EMBS:
            #near_ne_idxs_of_idxs = np.where(np.linalg.norm(embs_vld_rsh[idx] - embs_trn_rsh[idxs], axis=1) <= np.percentile(np.linalg.norm(embs_vld_rsh[idx] - embs_trn_rsh[idxs], axis=1), EMBS_PERCENTILE))[0]
            near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(embs_vld_rsh[idx] - embs_trn_rsh[idxs], axis=1))[0:K_NN_EMBS]


            near_ne_in_range = idxs[near_ne_idxs_of_idxs]
        else:
            near_ne_in_range = idxs
            

        #############################
        # Select from the references (randomly)
        if N_REFS <= len(near_ne_in_range):
            selected_idxs = np.random.choice(near_ne_in_range, N_REFS, replace=False)
        else:
            selected_idxs = np.random.choice(near_ne_in_range, N_REFS, replace=True)
        return selected_idxs, ref_age

    def query(self, idx, query_age, embs_trn_rsh, embs_vld_rsh):
        return self._find_ref_image(idx, query_age, self.ages, embs_trn_rsh, embs_vld_rsh)
        #return self._find_ref_image(query_age, self.ages)








class AgePredictSingleRefPipeline:
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
                    no_age_transformer_init=False
                    ):
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
        self.ref_querier = SingleRefQuerier(metadata, min_age, max_age)
        self.age_diff_predictor = get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, cfg, age_diff_model, is_ordinal_reg)
        if not no_age_transformer_init:
            self.age_transformer_inst.eval()
        self.age_diff_predictor.eval()

    def predict(self, query, query_indep, bypass_diff=False):
        image = query

        if bypass_diff:
            with torch.no_grad():
                _, query_est_age = self.age_transformer_inst(image)
            
            return query_est_age
        else:
            with torch.no_grad():
                _, query_est_age = self.age_transformer_inst(image)

                age_pred = int(np.round(query_est_age.cpu()))
                        
                ref_image_idx, _ = self.ref_querier.query(age_pred)
                
                image = query_indep[0]

                # orig_image = query
                            
                # orig_image = Image.fromarray(orig_image)
                # image = orig_image
                # # if self.copies > 1:
                # # 	images = []
                # # 	for i in range(self.copies):
                # # 		images.append(self.transform(orig_image))
                # # 	image = torch.stack(images)
                # # else:
                # image = self.transform(orig_image)
            
                #metadata = json.loads(self.metadata[idx])

                 

                ref_image_arr = self.images_train_db[ref_image_idx]
                # import pdb
                # pdb.set_trace()
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
                
                diff_select = torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - self.cfg.AGE_RADIUS #age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_soft
                    
                return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device)





# with internal check of the range of error of original age predict model
class AgePredictSingleRefEnhancedPipeline:
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
                    no_age_transformer_init=False,
                    age_diff_model_path_arg=None,
                    age_diff_model_file_name_arg=None,
                    embs_trn=None,
                    embs_vld=None
                    ):
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
        self.ref_querier = SingleRefQuerier(metadata, min_age, max_age)
        if age_diff_model is not None:
            self.age_diff_predictor = get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, age_diff_model, is_ordinal_reg, cfg, age_diff_model_path_arg, age_diff_model_file_name_arg)
        if not no_age_transformer_init:
            self.age_transformer_inst.eval()
        if age_diff_model is not None:
            self.age_diff_predictor.eval()

        self.embs_trn_rsh = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        self.embs_vld_rsh = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])


    def predict_diff_only(self, inputs):
        with torch.no_grad():
            classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(inputs)
        return classification_logits, age_diff_pred_hard, age_diff_pred_soft

    def get_diff_model(self):
        return self.age_diff_predictor

    def get_age_orig_model(self):
        return self.age_transformer_inst

    def predict(self, query, idx, query_indep=None, real_age=-1, range_in=-1, range_out=-1, bypass_diff=False, check_ranges=True, pred_db_val=None, check_against_pred_db=False):
        image = query

        if bypass_diff:
            #print("using original AgePredict (not diff-based)")
            if check_against_pred_db:
                query_est_age = torch.Tensor([pred_db_val]).to(self.device)
            else:
                with torch.no_grad():
                    _, query_est_age = self.age_transformer_inst(image)
            

            if check_ranges:
                age_pred_raw = query_est_age.cpu()
                age_pred = age_pred_raw #int(np.round(age_pred_raw))
                if (np.abs(age_pred - real_age.cpu()) >= range_in) and (np.abs(age_pred - real_age.cpu()) <= range_out):
                    return query_est_age, True
                else:
                    return query_est_age, False
            else:
                return query_est_age
        else:
            with torch.no_grad():
                if check_against_pred_db:
                    query_est_age = torch.Tensor([pred_db_val]).to(self.device)
                else:
                    _, query_est_age = self.age_transformer_inst(image)

                age_pred = int(np.round(query_est_age.cpu()))
                        
                #ref_image_idx = self.ref_querier.query(age_pred)

                # import pdb
                # pdb.set_trace() 

                #print("using diff-based")

                age_ests = []
                for age_pred_shift in [-1, 0, 1]:
                    ref_image_idx, ref_age = self.ref_querier.query(idx=int(idx), 
                                                            query_age=age_pred + age_pred_shift, 
                                                            embs_trn_rsh=self.embs_trn_rsh, 
                                                            embs_vld_rsh=self.embs_vld_rsh)

                    image = query_indep[0]

                    # orig_image = query
                                
                    # orig_image = Image.fromarray(orig_image)
                    # image = orig_image
                    # # if self.copies > 1:
                    # # 	images = []
                    # # 	for i in range(self.copies):
                    # # 		images.append(self.transform(orig_image))
                    # # 	image = torch.stack(images)
                    # # else:
                    # image = self.transform(orig_image)
                
                    #metadata = json.loads(self.metadata[idx])

                    ref_image_arr = self.images_train_db[ref_image_idx]
                    # import pdb
                    # pdb.set_trace()
                    ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
                    ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
                    ref_image_arr_metadata = [json.loads(self.metadata[idx]) for idx in ref_image_idx]
                    ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
                    #print(ref_age, age_pred, ref_image_arr_age)

                    # import matplotlib.pyplot as plt
                    # plt.imshow(np.transpose(ref_image_arr[0].numpy(), (1, 2, 0)))
                    # plt.show()

                    #ref_image_label = ref_image_arr_age[0] #// self.age_interval
                


                
                # # import pdb
                # # pdb.set_trace()
                # diff_select = 0.0
                # for ref_image in ref_image_arr:
                    ref_image = ref_image_arr[0]
                    image_vec = torch.stack(tuple([image.cpu()] + [ref_image]))

                    image_vec.to(self.device)
                    image_vec = image_vec.view(1,image_vec.shape[0],image_vec.shape[1],image_vec.shape[2],image_vec.shape[3])

                    


                    # import pdb
                    # pdb.set_trace()

                    #print(pair_label)

                    
                    with torch.no_grad():
                        classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(image_vec)
                    
                    #predicted_probs = ordinal_softmax(classification_logits, device=self.device).float()

                    #print(age_diff_pred_soft)
                    diff_select_i = age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - self.cfg.AGE_RADIUS #age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_soft

                    # if diff_select_i != 0.0:
                    #     print(diff_select_i)
                    #diff_select += diff_select_i    

                    age_ests.append(ref_age + float(diff_select_i.cpu()))

                    #print(ref_age, float(diff_select_i.cpu()))

                avg_age_est = np.mean(age_ests)

                # print(avg_age_est)
                # print(real_age.cpu())

                # print("---------------------------")

                # import pdb
                # pdb.set_trace() 
                    
                #diff_select = diff_select / N_REFS


                age_pred_raw = query_est_age.cpu()

                #age_pred_raw#int(np.round(age_pred_raw))
                # if (np.abs(age_pred_raw - real_age.cpu()) >= range_in) and (np.abs(age_pred_raw - real_age.cpu()) <= range_out):
                #     #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), True, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                #     return torch.Tensor([ref_age + float(diff_select.cpu())]).to(self.device), True, diff_select, torch.Tensor([real_age-ref_age]).to(self.device)
                # else:
                #     #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), False, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                #     return torch.Tensor([ref_age + float(diff_select.cpu())]).to(self.device), False, diff_select, torch.Tensor([real_age-ref_age]).to(self.device)
                    
                if (np.abs(age_pred_raw - real_age.cpu()) >= range_in) and (np.abs(age_pred_raw - real_age.cpu()) <= range_out):
                    #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), True, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                    return torch.Tensor([avg_age_est]).to(self.device), True, torch.Tensor([avg_age_est-age_pred]).to(self.device), torch.Tensor([real_age-age_pred]).to(self.device)
                else:
                    #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), False, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                    return torch.Tensor([avg_age_est]).to(self.device), False, torch.Tensor([avg_age_est-age_pred]).to(self.device), torch.Tensor([real_age-age_pred]).to(self.device)
                    







class AgePredictSearchPipeline:
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
                    no_age_transformer_init=False,
                    age_diff_model_path_arg=None,           # compare model
                    age_diff_model_file_name_arg=None,      # compare model
                    embs_trn=None,
                    embs_vld=None
                    ):
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
        self.ref_querier = SingleRefQuerier(metadata, min_age, max_age)
        if age_diff_model is not None:
            self.age_diff_predictor = get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, age_diff_model, is_ordinal_reg, cfg, age_diff_model_path_arg, age_diff_model_file_name_arg)
        if not no_age_transformer_init:
            self.age_transformer_inst.eval()
        if age_diff_model is not None:
            self.age_diff_predictor.eval()

        self.embs_trn_rsh = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        self.embs_vld_rsh = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])

        self.min_age = min_age
        self.max_age = max_age
        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.metadata])


    def predict_diff_only(self, inputs):
        with torch.no_grad():
            classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(inputs)
        return classification_logits, age_diff_pred_hard, age_diff_pred_soft

    def get_diff_model(self):
        return self.age_diff_predictor

    def get_age_orig_model(self):
        return self.age_transformer_inst

    def predict(self, query, idx, query_indep=None, real_age=-1, range_in=-1, range_out=-1, bypass_diff=False, check_ranges=True, pred_db_val=None, check_against_pred_db=False):
        image = query

        if bypass_diff:
            #print("using original AgePredict (not diff-based)")
            if check_against_pred_db:
                query_est_age = torch.Tensor([pred_db_val]).to(self.device)
            else:
                with torch.no_grad():
                    _, query_est_age = self.age_transformer_inst(image)
            

            if check_ranges:
                age_pred_raw = query_est_age.cpu()
                age_pred = age_pred_raw #int(np.round(age_pred_raw))
                if (np.abs(age_pred - real_age.cpu()) >= range_in) and (np.abs(age_pred - real_age.cpu()) <= range_out):
                    return query_est_age, True
                else:
                    return query_est_age, False
            else:
                return query_est_age
        else:
            with torch.no_grad():
                if check_against_pred_db:
                    query_est_age = torch.Tensor([pred_db_val]).to(self.device)
                else:
                    _, query_est_age = self.age_transformer_inst(image)

                age_pred = int(np.round(query_est_age.cpu()))
                        
                #ref_image_idx = self.ref_querier.query(age_pred)

                # import pdb
                # pdb.set_trace() 

                #print("using diff-based")
                image = query_indep[0]

                ref_age = int((self.min_age + self.max_age) / 2)
                cur_search_space_size = (self.max_age - self.min_age + 1 ) / 2
                while cur_search_space_size > 1:
                    idxs = []
                    while len(idxs) == 0:
                        ref_age -= 1
                        idxs = np.where(self.ages == ref_age)[0]

                    ref_image_idx = np.random.choice(idxs, 1, replace=False)
        
                    ref_image_arr = self.images_train_db[ref_image_idx]
                    # import pdb
                    # pdb.set_trace()
                    ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
                    ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
                    ref_image_arr_metadata = [json.loads(self.metadata[idx]) for idx in ref_image_idx]
                    ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
                    #print(ref_age, age_pred, ref_image_arr_age)


                    ref_image = ref_image_arr[0]
                    image_vec = torch.stack(tuple([image.cpu()] + [ref_image]))

                    image_vec.to(self.device)
                    image_vec = image_vec.view(1,image_vec.shape[0],image_vec.shape[1],image_vec.shape[2],image_vec.shape[3])

                    
                    classification_logits, age_diff_pred_hard, _ = self.age_diff_predictor(image_vec)
                    

                    # import matplotlib.pyplot as plt
                    # plt.imshow(np.transpose(ref_image_arr[0].numpy(), (1, 2, 0)))
                    # plt.show()

                    #ref_image_label = ref_image_arr_age[0] #// self.age_interval

                    # TODO : may use the probs here and do a weighted recursive bnary search
                    # (i.e. in iteration i where we are in point=m, if p for <= and 1-p for <, the result of this step will be 
                    # p*search(points={<m}) + (1-p)*search(point={>m}))
                    cur_search_space_size = int(cur_search_space_size / 2)
                    if int(age_diff_pred_hard.cpu()) <= 1:
                        ref_age = ref_age - cur_search_space_size
                    else:
                        ref_age = ref_age + cur_search_space_size
                
                if (np.abs(age_pred - real_age.cpu()) >= range_in) and (np.abs(age_pred - real_age.cpu()) <= range_out):
                    #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), True, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                    return torch.Tensor([ref_age]).to(self.device), True, torch.Tensor([ref_age-age_pred]).to(self.device), torch.Tensor([real_age-age_pred]).to(self.device)
                else:
                    #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), False, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                    return torch.Tensor([ref_age]).to(self.device), False, torch.Tensor([ref_age-age_pred]).to(self.device), torch.Tensor([real_age-age_pred]).to(self.device)
                    

                    
                #return torch.Tensor([ref_age]).to(self.device)
            


# with internal check of the range of error of original age predict model
class AgePredictSingleRefComparePipeline:
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
                    no_age_transformer_init=False,
                    age_diff_model_path_arg=None,           # compare model
                    age_diff_model_file_name_arg=None,      # compare model
                    embs_trn=None,
                    embs_vld=None
                    ):
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
        self.ref_querier = SingleRefQuerier(metadata, min_age, max_age)
        if age_diff_model is not None:
            self.age_diff_predictor = get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, age_diff_model, is_ordinal_reg, cfg, age_diff_model_path_arg, age_diff_model_file_name_arg)
        if not no_age_transformer_init:
            self.age_transformer_inst.eval()
        if age_diff_model is not None:
            self.age_diff_predictor.eval()

        self.embs_trn_rsh = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        self.embs_vld_rsh = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])

        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.metadata])

        self.max_age = max_age
        self.min_age = min_age

    def predict_diff_only(self, inputs):
        with torch.no_grad():
            classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(inputs)
        return classification_logits, age_diff_pred_hard, age_diff_pred_soft

    def get_diff_model(self):
        return self.age_diff_predictor

    def get_age_orig_model(self):
        return self.age_transformer_inst

    def predict(self, query, idx, query_indep=None, real_age=-1, range_in=-1, range_out=-1, bypass_diff=False, check_ranges=True, pred_db_val=None, check_against_pred_db=False):
        image = query

        if bypass_diff:
            #print("using original AgePredict (not diff-based)")
            if check_against_pred_db:
                query_est_age = torch.Tensor([pred_db_val]).to(self.device)
            else:
                with torch.no_grad():
                    _, query_est_age = self.age_transformer_inst(image)
            

            if check_ranges:
                age_pred_raw = query_est_age.cpu()
                age_pred = age_pred_raw #int(np.round(age_pred_raw))
                if (np.abs(age_pred - real_age.cpu()) >= range_in) and (np.abs(age_pred - real_age.cpu()) <= range_out):
                    return query_est_age, True
                else:
                    return query_est_age, False
            else:
                return query_est_age
        else:
            with torch.no_grad():
                if check_against_pred_db:
                    query_est_age = torch.Tensor([pred_db_val]).to(self.device)
                else:
                    _, query_est_age = self.age_transformer_inst(image)

                #age_pred = int(np.round(query_est_age.cpu()))
                        
                age_pred = float(query_est_age.cpu())

                #ref_image_idx = self.ref_querier.query(age_pred)

                # import pdb
                # pdb.set_trace() 

                #print("using diff-based")

                # ref_image_idx, ref_age = self.ref_querier.query(idx=int(idx), 
                #                                         query_age=age_pred, 
                #                                         embs_trn_rsh=self.embs_trn_rsh, 
                #                                         embs_vld_rsh=self.embs_vld_rsh)

                if age_pred < self.max_age - 6:
                    idxs_bigger = np.where((self.ages > age_pred + 5) & (self.ages < age_pred + 35))[0]
                else:
                    idxs_bigger = np.where((self.ages > age_pred) & (self.ages < age_pred + 35))[0]
                
                if age_pred > self.min_age + 6:
                    idxs_smaller = np.where((self.ages < age_pred - 5) & (self.ages > age_pred - 35))[0]
                else:
                    idxs_smaller = np.where((self.ages < age_pred) & (self.ages > age_pred - 35))[0]

                if len(idxs_bigger) == 0 or len(idxs_smaller) == 0:
                    return query_est_age

                ref_image_idx_bigger = np.random.choice(idxs_bigger, 1, replace=False)
                ref_image_idx_smaller = np.random.choice(idxs_smaller, 1, replace=False)

                image = query_indep[0]

                # orig_image = query
                            
                # orig_image = Image.fromarray(orig_image)
                # image = orig_image
                # # if self.copies > 1:
                # # 	images = []
                # # 	for i in range(self.copies):
                # # 		images.append(self.transform(orig_image))
                # # 	image = torch.stack(images)
                # # else:
                # image = self.transform(orig_image)
            
                #metadata = json.loads(self.metadata[idx])

                ref_image_arr_bigger = self.images_train_db[ref_image_idx_bigger]
                # import pdb
                # pdb.set_trace()
                ref_image_arr_bigger = [Image.fromarray(ref_image) for ref_image in ref_image_arr_bigger]
                ref_image_arr_bigger = [self.transform(ref_image) for ref_image in ref_image_arr_bigger]
                ref_image_arr_metadata_bigger = [json.loads(self.metadata[idx]) for idx in ref_image_idx_bigger]
                ref_image_arr_age_bigger = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata_bigger]
                #print(ref_age, age_pred, ref_image_arr_age)

                # import matplotlib.pyplot as plt
                # plt.imshow(np.transpose(ref_image_arr[0].numpy(), (1, 2, 0)))
                # plt.show()

                #ref_image_label = ref_image_arr_age[0] #// self.age_interval

                ref_age_bigger = ref_image_arr_age_bigger[0]



                ref_image_arr_smaller = self.images_train_db[ref_image_idx_smaller]
                # import pdb
                # pdb.set_trace()
                ref_image_arr_smaller = [Image.fromarray(ref_image) for ref_image in ref_image_arr_smaller]
                ref_image_arr_smaller = [self.transform(ref_image) for ref_image in ref_image_arr_smaller]
                ref_image_arr_metadata_smaller = [json.loads(self.metadata[idx]) for idx in ref_image_idx_smaller]
                ref_image_arr_age_smaller = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata_smaller]
                #print(ref_age, age_pred, ref_image_arr_age)

                # import matplotlib.pyplot as plt
                # plt.imshow(np.transpose(ref_image_arr[0].numpy(), (1, 2, 0)))
                # plt.show()

                #ref_image_label = ref_image_arr_age[0] #// self.age_interval

                ref_age_smaller = ref_image_arr_age_smaller[0]


            
            # # import pdb
            # # pdb.set_trace()
            # diff_select = 0.0
            # for ref_image in ref_image_arr:
                ref_image_bigger = ref_image_arr_bigger[0]
                image_vec_b = torch.stack(tuple([image.cpu()] + [ref_image_bigger]))

                image_vec_b.to(self.device)
                image_vec_bigger = image_vec_b.view(1,image_vec_b.shape[0],image_vec_b.shape[1],image_vec_b.shape[2],image_vec_b.shape[3])

                ref_image_smaller = ref_image_arr_smaller[0]
                image_vec_s = torch.stack(tuple([image.cpu()] + [ref_image_smaller]))

                image_vec_s.to(self.device)
                image_vec_smaller = image_vec_s.view(1,image_vec_s.shape[0],image_vec_s.shape[1],image_vec_s.shape[2],image_vec_s.shape[3])

                



                # import pdb
                # pdb.set_trace()

                #print(pair_label)

                
                classification_logits_bigger, q_small_or_eq_r_bigger, _ = self.age_diff_predictor(image_vec_bigger)
                
                classification_logits_smaller, q_small_or_eq_r_smaller, _ = self.age_diff_predictor(image_vec_smaller)
                
                #predicted_probs = ordinal_softmax(classification_logits, device=self.device).float()

                # age_est = age_pred
                # if age_pred < ref_age:
                #     print("age_pred, age_est, real_age, ref_age, compare - age_pred < ref_age")
                #     if int(q_small_or_eq_r.cpu()) == 2:
                #         # q > r
                #         age_est += 1
                #     else:
                        
                # elif age_pred > ref_age:
                #     print("age_pred, age_est, real_age, ref_age, compare - age_pred > ref_age")
                #     if int(q_small_or_eq_r.cpu()) == 0:
                #         # q < r
                #         age_est -= 1
                # elif age_pred == ref_age:
                #     print("age_pred, age_est, real_age, ref_age, compare -age_pred = ref_age")
                #     if int(q_small_or_eq_r.cpu()) == 0:
                #         # q < r
                #         age_est -= 1
                #     elif int(q_small_or_eq_r.cpu()) == 2:
                #         # q > r
                #         age_est += 1

                age_est = age_pred

                if int(q_small_or_eq_r_bigger.cpu()) == 2:
                    # q > r > age_est
                    age_est += 1
                    if np.abs(real_age-age_est) < np.abs(real_age-age_pred):
                        print("real_age: {real_age}, age_est_pre: {age_est_pre}, age_est_post : {age_est_post}, ref_age : {ref_age}, good fix !".format(real_age=real_age,age_est_pre=age_pred,age_est_post=age_est,ref_age=ref_age_bigger))
                    else:
                        print("real_age: {real_age}, age_est_pre: {age_est_pre}, age_est_post : {age_est_post}, ref_age : {ref_age}, bad fix !".format(real_age=real_age,age_est_pre=age_pred,age_est_post=age_est,ref_age=ref_age_bigger))
                        
                elif int(q_small_or_eq_r_smaller.cpu()) == 0:
                    # q < r < age_est
                    age_est -= 1
                    if np.abs(real_age-age_est) < np.abs(real_age-age_pred):
                        print("real_age: {real_age}, age_est_pre: {age_est_pre}, age_est_post : {age_est_post}, ref_age : {ref_age}, good fix !".format(real_age=real_age,age_est_pre=age_pred,age_est_post=age_est,ref_age=ref_age_smaller))
                    else:
                        print("real_age: {real_age}, age_est_pre: {age_est_pre}, age_est_post : {age_est_post}, ref_age : {ref_age}, bad fix !".format(real_age=real_age,age_est_pre=age_pred,age_est_post=age_est,ref_age=ref_age_smaller))
                        

                # print(age_pred, age_est, float(real_age.cpu()), ref_age, int(q_small_or_eq_r.cpu()))
                    

                age_pred_raw = query_est_age.cpu()

                #age_pred_raw#int(np.round(age_pred_raw))
                # if (np.abs(age_pred_raw - real_age.cpu()) >= range_in) and (np.abs(age_pred_raw - real_age.cpu()) <= range_out):
                #     #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), True, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                #     return torch.Tensor([ref_age + float(diff_select.cpu())]).to(self.device), True, diff_select, torch.Tensor([real_age-ref_age]).to(self.device)
                # else:
                #     #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), False, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                #     return torch.Tensor([ref_age + float(diff_select.cpu())]).to(self.device), False, diff_select, torch.Tensor([real_age-ref_age]).to(self.device)
                    
                if (np.abs(age_pred_raw - real_age.cpu()) >= range_in) and (np.abs(age_pred_raw - real_age.cpu()) <= range_out):
                    #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), True, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                    return torch.Tensor([age_est]).to(self.device), True, torch.Tensor([age_est-age_pred_raw]).to(self.device), torch.Tensor([real_age-age_pred_raw]).to(self.device)
                else:
                    #return torch.Tensor([age_pred + float(diff_select.cpu())]).to(self.device), False, diff_select, torch.Tensor([real_age-age_pred]).to(self.device)
                    return torch.Tensor([age_est]).to(self.device), False, torch.Tensor([age_est-age_pred_raw]).to(self.device), torch.Tensor([real_age-age_pred_raw]).to(self.device)
                    



