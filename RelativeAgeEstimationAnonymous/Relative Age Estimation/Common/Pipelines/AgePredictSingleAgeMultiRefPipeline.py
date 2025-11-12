import json
import numpy as np
import torch
from PIL import Image
from Common.Pipelines.AgePredictBasicPipeline import get_age_transformer, get_age_diff_predictor
# importing the sys module
import sys        
 
from condor_pytorch.activations import ordinal_softmax


DEBUG_DATASET = False
DEBUG_REF_SIMILARITY = False
PRINT_TST_IDX_AND_AP_AGE = False

class SingleAgeMultiRefQuerier:
    def __init__(self, metadata, min_age, max_age, num_references, use_embs, embs_knn, batrn_set_images, batst_set_images):
        self.metadata = metadata
        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.metadata])
        self.min_age = min_age
        self.max_age = max_age
        self.num_references = num_references
        self.use_embs = use_embs
        self.embs_knn = embs_knn
        self.batrn_set_images = batrn_set_images[:, :, :, [2, 1, 0]]
        self.batst_set_images = batst_set_images[:, :, :, [2, 1, 0]]
        
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

        if self.use_embs:
            base_potential_idxs = idxs
            # import pdb
            # pdb.set_trace()
            near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(embs_vld_rsh[idx] - embs_trn_rsh[base_potential_idxs], axis=1))
            actual_idxs = base_potential_idxs[near_ne_idxs_of_idxs][0:self.embs_knn]
        else:
            actual_idxs = idxs

        if DEBUG_REF_SIMILARITY:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, len(actual_idxs) + 1, figsize=(15, 10))
            ax[0].imshow(self.batst_set_images[idx])#.reshape((224,224,3)))
            for i in range(len(actual_idxs)):
                ax[i+1].imshow(self.batrn_set_images[actual_idxs[i]])#.reshape((224,224,3)))
            q_r_distance = np.linalg.norm(embs_vld_rsh[idx] - embs_trn_rsh[actual_idxs], axis=1)
            print(f'q-r distance: {q_r_distance}')
            # print(embs_vld_rsh[idx])
            # for i in actual_idxs:
            #     print(embs_trn_rsh[i])
            plt.show()

        # import pdb
        # pdb.set_trace()
        if len(actual_idxs) < self.num_references:
            if DEBUG_DATASET:
                print(actual_idxs, query_age-ref_age)
                print("not enough ref images in radius! replicating")
            selected_idxs = np.random.choice(actual_idxs, self.num_references)
            if DEBUG_REF_SIMILARITY:
                print(f"test idx = {idx}")
                print(f"ref age {ref_age}")
                print(f"train idxs refs = {selected_idxs}")
            return selected_idxs, ref_age
        else:
            selected_idxs = np.random.choice(actual_idxs, self.num_references, replace=False)
            if DEBUG_REF_SIMILARITY:
                print(f"test idx = {idx}")
                print(f"ref age {ref_age}")
                print(f"train idxs refs = {selected_idxs}")
            return selected_idxs, ref_age

    def query(self, idx, query_age, embs_trn_rsh, embs_vld_rsh):
        return self._find_ref_image(idx, query_age, self.ages, embs_trn_rsh, embs_vld_rsh)
        #return self._find_ref_image(query_age, self.ages)





# with internal check of the range of error of original age predict model
class AgePredictSingleAgeMultiRefPipeline:
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
                    images_test_db,
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
        self.ref_querier = SingleAgeMultiRefQuerier(metadata, 
                                                        min_age, 
                                                        max_age, 
                                                        num_references=cfg.NUM_OF_REFS, 
                                                        use_embs=cfg.USE_EMBEDDINGS, 
                                                        embs_knn=cfg.EMBEDDINGS_KNN, 
                                                        batrn_set_images=images_train_db, 
                                                        batst_set_images=images_test_db)
        if age_diff_model is not None:
            self.age_diff_predictor = get_age_diff_predictor(age_interval, min_age, max_age, age_radius, device, deep, num_references, config_type, added_embed_layer_size, diff_embed_layer_size, age_diff_model, is_ordinal_reg, cfg, age_diff_model_path_arg, age_diff_model_file_name_arg)
        if not no_age_transformer_init:
            self.age_transformer_inst.eval()
        if age_diff_model is not None:
            self.age_diff_predictor.eval()

        self.embs_trn_rsh = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        self.embs_vld_rsh = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])

        self.count_moved_refs = 0
        self.last_printed = -1


    def predict_diff_only(self, inputs, age_ref=None):
        with torch.no_grad():
            if age_ref is None:
                res = self.age_diff_predictor(inputs)
            else:
                res = self.age_diff_predictor(input_images=inputs, age_ref=age_ref)
            if len(res) == 3:
                classification_logits = res[0]
                age_diff_pred_hard = res[1]
                age_diff_pred_soft = res[2]
                return classification_logits, age_diff_pred_hard, age_diff_pred_soft
            elif len(res) == 2:
                classification_logits = res[0]
                age_diff_pred = res[1]
                return classification_logits, age_diff_pred
            else:
                print("unsupported model")
                return None
        return classification_logits, age_diff_pred_hard, age_diff_pred_soft

    def get_diff_model(self):
        return self.age_diff_predictor

    def get_age_orig_model(self):
        return self.age_transformer_inst

    def predict(self, query, idx, query_indep=None, real_age=-1, range_in=-1, range_out=-1, bypass_diff=False, check_ranges=True, pred_db_val=None, check_against_pred_db=False, apply_diff_model_only_in_range=False, use_round_for_range_check=True):
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
            ####################################################################################
            #       Run original AgePredict
            ####################################################################################
            with torch.no_grad():
                if check_against_pred_db:
                    query_est_age = torch.Tensor([pred_db_val]).to(self.device)
                else:
                    _, query_est_age = self.age_transformer_inst(image)
            
            age_pred_raw = query_est_age.cpu()

            if PRINT_TST_IDX_AND_AP_AGE:
                print(f"PRINT_TST_IDX_AND_AP_AGE: query_idx={idx}, query_ap_age={float(age_pred_raw)}")
            

            ####################################################################################
            #       Do not apply diff model in case not in range
            ####################################################################################  
            if apply_diff_model_only_in_range:
                if use_round_for_range_check:
                    if not((np.abs(np.round(age_pred_raw) - real_age.cpu()) >= range_in) and (np.abs(np.round(age_pred_raw) - real_age.cpu()) <= range_out)):
                        # Note: In this case, we just return the original AgePredict result
                        # final age pred, is in range, predicted diff, real diff
                        return torch.Tensor([age_pred_raw]).to(self.device), False, torch.Tensor([0.0]).to(self.device), torch.Tensor([0.0]).to(self.device)
                else:
                    if not((np.abs(age_pred_raw - real_age.cpu()) >= range_in) and (np.abs(age_pred_raw - real_age.cpu()) <= range_out)):
                        # Note: In this case, we just return the original AgePredict result
                        # final age pred, is in range, predicted diff, real diff
                        return torch.Tensor([age_pred_raw]).to(self.device), False, torch.Tensor([0.0]).to(self.device), torch.Tensor([0.0]).to(self.device)

            ####################################################################################
            #       Applying diff model for fixing the original AgePredict
            ####################################################################################  
            age_pred = int(np.round(query_est_age.cpu()))

            
            # The original AgePredict model returns a result with some average MAE.
            # We take this result as an estimate and try to find references in that age. It because 
            # This the only age information we have on the input, and we know it is in a bounded average distance
            # from the real age, so inferring the difference between the ref and the input query image is likely to produce 
            # a result in good direction. It will of course work if we can know where the error is placed (in what range)
            # to simulate error range knowledge, we use the range_in, range_out and apply_diff_model_only_in_range.
            # Note that if no ref in found such that it's age == AgePredict(query), we go "nearby" and look for reference near the
            # prediction. It is not perfect but it's one way to get reference in close age to actual diff range. That is why
            # ref_age is returned from the call below.
            ref_image_idx, ref_age = self.ref_querier.query(idx=int(idx), 
                                                    query_age=age_pred, 
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
        


            ref_image = ref_image_arr[0]
            image_vec = torch.stack(tuple([image.cpu()] + ref_image_arr))

            image_vec.to(self.device)
            image_vec = image_vec.view(1,image_vec.shape[0],image_vec.shape[1],image_vec.shape[2],image_vec.shape[3])

            

            
            with torch.no_grad():
                classification_logits, age_diff_pred_hard, age_diff_pred_soft = self.age_diff_predictor(image_vec)
            
            diff_select_i = age_diff_pred_soft #age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - self.cfg.AGE_RADIUS #age_diff_pred_soft #torch.sum(torch.cumprod(torch.sigmoid(classification_logits),dim=1)>0.5,dim=1,keepdim=True,dtype=classification_logits.dtype) - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_hard - cfg.AGE_RADIUS #age_diff_pred_soft

            avg_age_est = ref_age + float(diff_select_i.cpu())

            if ref_age != age_pred:
                self.count_moved_refs += 1

            if (self.count_moved_refs % 50 == 0) and (self.last_printed < self.count_moved_refs): 
                self.last_printed = self.count_moved_refs
                self.print_moved_refs()

            if (np.abs(np.round(age_pred_raw) - real_age.cpu()) >= range_in) and (np.abs(np.round(age_pred_raw) - real_age.cpu()) <= range_out):
                # final age pred, is in range, predicted diff, real diff
                return torch.Tensor([avg_age_est]).to(self.device), True, torch.Tensor([float(diff_select_i.cpu())]).to(self.device), torch.Tensor([real_age-ref_age]).to(self.device)
            else:
                if apply_diff_model_only_in_range:
                    print("Error - can't be here if apply_diff_model_only_in_range is True")
                    exit()
                # final age pred, is in range, predicted diff, real diff
                return torch.Tensor([avg_age_est]).to(self.device), False, torch.Tensor([float(diff_select_i.cpu())]).to(self.device), torch.Tensor([real_age-ref_age]).to(self.device)

    def print_moved_refs(self):
        print(f'moved refs: {self.count_moved_refs}')