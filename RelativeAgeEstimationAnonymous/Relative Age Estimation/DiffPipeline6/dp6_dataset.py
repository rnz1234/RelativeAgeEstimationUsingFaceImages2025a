import json 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random


DEBUG_DATASET = False
ANALYZE_DATASET = False
DEBUG_REF_SIMILARITY = False #True
#RECORD_VALIDATION_DATASET = False

def arr_safe_norm(raw_arr):
    normed_arr = np.zeros(raw_arr.shape)
    arr_norm = np.linalg.norm(raw_arr, axis=1)
    normed_arr[arr_norm > 0] = raw_arr[arr_norm > 0] / arr_norm.reshape(arr_norm.shape[0],1)[arr_norm > 0]
    return normed_arr

class AgeDiffSameUniformDiffDataset(Dataset):
    def __init__(self,
                data_set_images,                # basic aligned * images 
                data_set_metadata,              # basic aligned * metadata (labels)
                min_age,
                age_interval,
                max_age,
                transform=None,
                copies=1,
                age_diff_learn_radius_hi=3,
                age_diff_learn_radius_lo=0,
                num_references=1,
                embs=None,
                use_embs=False,
                embs_knn=None,
                embs_far_knn=False,
                embeddings_based_ratio=1.0,
                embs_normalize=True,
                dataset_type="train"):
        
        self.data_set_images = data_set_images[:, :, :, [2, 1, 0]]
        self.data_set_metadata = data_set_metadata

        self.min_age = min_age
        self.age_interval = age_interval
        self.max_age = max_age
        self.transform = transform
        self.copies = copies
        self.age_diff_learn_radius_hi = age_diff_learn_radius_hi
        self.age_diff_learn_radius_lo = age_diff_learn_radius_lo
        self.use_embs = use_embs
        self.embs_knn = embs_knn
        self.embs_far_knn = embs_far_knn
        self.num_references = num_references
        self.embeddings_based_ratio = embeddings_based_ratio
        self.embs_normalize = embs_normalize
        self.dataset_type = dataset_type
        
        raw_embs = embs.reshape(embs.shape[0], embs.shape[2])
        # normalize embeddings
        if self.embs_normalize:
            self.embs_rsh = arr_safe_norm(raw_embs) 
        else:
            self.embs_rsh = raw_embs
            
    
        self.total_diffs = []
        self.len_of_potential_refs = []

        # additional attributes

        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.data_set_metadata])
        
    def _find_ref_image(self, idx):
        # import pdb
        # pdb.set_trace()
        # setting the range of possible diffs in order to select a specific one.
        # we don't want to select one that won't have possible references in dataset (ages not in range)
        if self.age_diff_learn_radius_lo == 0:
            #diff_range = np.arange(-min(self.ages[idx]-(self.min_age+1), self.age_diff_learn_radius_hi), min(self.max_age-self.ages[idx], self.age_diff_learn_radius_hi)+1)	
            diff_range = np.arange(-self.age_diff_learn_radius_hi, self.age_diff_learn_radius_hi+1)
        else:
            #neg_range = np.arange(-min(self.ages[idx]-(self.min_age+1), self.age_diff_learn_radius_hi), -self.age_diff_learn_radius_lo+1)
            #pos_range = np.arange(self.age_diff_learn_radius_lo, min(self.max_age-self.ages[idx], self.age_diff_learn_radius_hi)+1)	
            neg_range = np.arange(-self.age_diff_learn_radius_hi, -self.age_diff_learn_radius_lo+1)
            pos_range = np.arange(self.age_diff_learn_radius_lo, self.age_diff_learn_radius_hi)
            
            diff_range = np.concatenate((neg_range, pos_range)) 

        # we look for a diff that will have examples in our dataset. We randomly choose from the range.
        # if the selected diff doesn't have candidates at all, we try a different diff from range. We are supposed 
        # to find something. If we find, we continue. If we don't find, we abort since the given age cannot be used for the task
        # (but there a very small chance this will happen)s
        not_found = True
        on_first = True
        while not_found:
            # select specific diff in radius
            selected_diff = np.random.choice(diff_range, 1)
            # find all pool of potential references for this age diff
            idxs = np.where(self.ages[idx] - self.ages == selected_diff)

            if len(idxs[0]) == 0: 
                on_first = False
                if DEBUG_DATASET:
                    print("zero ref images for age ", self.ages[idx], " in radius ", selected_diff)
                diff_range = np.delete(diff_range, np.where(diff_range == selected_diff)[0][0])
                if len(diff_range) == 0:
                    print("zero ref images in any radius of age: ", self.ages[idx])
                    print("ABORTING...")
                    exit()
            else:
                if not on_first:
                    if DEBUG_DATASET:
                        print("found ref images for age ", self.ages[idx], " in radius ", selected_diff)
                not_found = False


        if ANALYZE_DATASET:
            self.total_diffs.append(selected_diff)
            self.len_of_potential_refs.append(len(idxs[0]))

        if self.use_embs:
            if (self.dataset_type == "valid") or (random.random() <= self.embeddings_based_ratio):
                base_potential_idxs = idxs[0]
                near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[base_potential_idxs], axis=1))
                if self.dataset_type == "train":
                    if self.embs_far_knn:
                        self.embs_knn = min(max(self.num_references, int(len(near_ne_idxs_of_idxs) / 8)), 150)
                #print(self.embs_knn, self.ages[idx])
                actual_idxs = base_potential_idxs[near_ne_idxs_of_idxs][0:self.embs_knn]
                if self.dataset_type == "train":
                    if idx in actual_idxs:
                        if selected_diff != 0:
                            print("found query idx in references idxs, but diff != 0")
                            print("ABORTING...")
                            exit()
                        else:
                            if len(actual_idxs) > 1:
                                # remove (q,q) pairs from training set
                                actual_idxs = np.delete(actual_idxs, np.where(actual_idxs == idx))
                                if len(base_potential_idxs[near_ne_idxs_of_idxs]) > self.embs_knn:
                                    actual_idxs = np.append(actual_idxs, base_potential_idxs[near_ne_idxs_of_idxs][self.embs_knn])
                            else:
                                actual_idxs = idxs[0]
            else:
                actual_idxs = idxs[0]
        else:
            actual_idxs = idxs[0]

        # we found a diff we have one or more candidates. In case we have less than the needed 
        # num of references, we will replicate (probablistically). Else, we can just sample the num of references 
        # of unique references.
        
        # if DEBUG_REF_SIMILARITY:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(1, len(actual_idxs) + 1, figsize=(15, 10))
        #     ax[0].imshow(self.data_set_images[idx])#.reshape((224,224,3)))
        #     for i in range(len(actual_idxs)):
        #         ax[i+1].imshow(self.data_set_images[actual_idxs[i]])#.reshape((224,224,3)))
        #     q_r_distance = np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[actual_idxs], axis=1)
        #     print(f'q-r distance: {q_r_distance}')
        #     plt.show()
            # import pdb
            # pdb.set_trace()

        if len(actual_idxs) < self.num_references:
            # import pdb
            # pdb.set_trace()
            if DEBUG_DATASET:
                print(self.ages[idx], actual_idxs, selected_diff, diff_range)
                print("not enough ref images in radius! replicating")
            selected_idxs = np.random.choice(actual_idxs, self.num_references)
            return selected_idxs
        else:
            selected_idxs = np.random.choice(actual_idxs, self.num_references, replace=False)
            return selected_idxs


    def __len__(self):
        return len(self.data_set_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        orig_image = self.data_set_images[idx]

        orig_image = Image.fromarray(orig_image)
        
        image = self.transform(orig_image)

        metadata = json.loads(self.data_set_metadata[idx])

        age = int(metadata['age'])
        label = age

        ref_image_idx = self._find_ref_image(idx)



        ref_image_arr = self.data_set_images[ref_image_idx]
        ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
        ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
        ref_image_arr_metadata = [json.loads(self.data_set_metadata[idx_i]) for idx_i in ref_image_idx]
        ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
        
        # import pdb
        # pdb.set_trace()

        ref_image_label = ref_image_arr_age[0] #// self.age_interval

        scaled_radius = self.age_diff_learn_radius_hi #// self.age_interval

        image_vec = torch.stack(tuple([image] + ref_image_arr))

        pair_label =  label - ref_image_label + scaled_radius

        sample = {'image_vec': image_vec, 'label': pair_label, 'age_diff': age - ref_image_arr_age[0], 'age_ref': ref_image_arr_age[0]}
        
        return sample

    def get_stats(self):
        return self.total_diffs, self.len_of_potential_refs

class AgeDiffMixedUniformDiffDataset(Dataset):
    def __init__(self,
                batrn_set_images,               # basic aligned train images 
                batrn_set_metadata,             # basic aligned train metadata (labels)
                batst_set_images,               # basic aligned test images
                batst_set_metadata,             # basic aligned test metadata (labels)
                min_age,
                age_interval,
                max_age,
                transform=None,
                copies=1,
                age_diff_learn_radius_hi=3,
                age_diff_learn_radius_lo=0,
                num_references=1,
                embs_trn=None,
                embs_vld=None,
                use_embs=False,
                embs_knn=1,
                embs_normalize=True):
        
        
        self.batrn_set_images = batrn_set_images[:, :, :, [2, 1, 0]]
        self.batrn_set_metadata = batrn_set_metadata

        self.batst_set_images = batst_set_images[:, :, :, [2, 1, 0]]
        self.batst_set_metadata = batst_set_metadata

        self.min_age = min_age
        self.age_interval = age_interval
        self.max_age = max_age
        self.transform = transform
        self.copies = copies
        self.age_diff_learn_radius_hi = age_diff_learn_radius_hi
        self.age_diff_learn_radius_lo = age_diff_learn_radius_lo
        
        
        self.use_embs = use_embs
        self.embs_knn = embs_knn
        self.embs_normalize = embs_normalize 

        self.num_references = num_references

        raw_embs_trn = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        raw_embs_vld = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])
        # normalize embeddings
        if self.embs_normalize:
            self.embs_trn_rsh = arr_safe_norm(raw_embs_trn)
            self.embs_vld_rsh = arr_safe_norm(raw_embs_vld) 
        else:
            self.embs_trn_rsh = raw_embs_trn
            self.embs_vld_rsh = raw_embs_vld


        # additional attributes
        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        self.ages_ba_test = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batst_set_metadata])
        
    def _find_ref_image(self, idx):
        # import pdb
        # pdb.set_trace()
        # setting the range of possible diffs in order to select a specific one.
        # we don't want to select one that won't have possible references in dataset (ages not in range)
        if self.age_diff_learn_radius_lo == 0:
            #diff_range = np.arange(-min(self.ages_ba_test[idx]-(self.min_age+1), self.age_diff_learn_radius_hi), min(self.max_age-self.ages_ba_test[idx], self.age_diff_learn_radius_hi)+1)	
            diff_range = np.arange(-self.age_diff_learn_radius_hi, self.age_diff_learn_radius_hi+1)
        else:
            #neg_range = np.arange(-min(self.ages[idx]-(self.min_age+1), self.age_diff_learn_radius_hi), -self.age_diff_learn_radius_lo+1)
            #pos_range = np.arange(self.age_diff_learn_radius_lo, min(self.max_age-self.ages[idx], self.age_diff_learn_radius_hi)+1)	
            neg_range = np.arange(-self.age_diff_learn_radius_hi, -self.age_diff_learn_radius_lo+1)
            pos_range = np.arange(self.age_diff_learn_radius_lo, self.age_diff_learn_radius_hi)
            
            diff_range = np.concatenate((neg_range, pos_range)) 

        # we look for a diff that will have examples in our dataset. We randomly choose from the range.
        # if the selected diff doesn't have candidates at all, we try a different diff from range. We are supposed 
        # to find something. If we find, we continue. If we don't find, we abort since the given age cannot be used for the task
        # (but there a very small chance this will happen)s
        not_found = True
        on_first = True
        while not_found:
            # select specific diff in radius
            selected_diff = np.random.choice(diff_range, 1)
            # find all pool of potential references for this age diff
            idxs = np.where(self.ages_ba_test[idx] - self.ages_ba_train == selected_diff)

            if len(idxs[0]) == 0: 
                on_first = False
                if DEBUG_DATASET:
                    print("zero ref images for age ", self.ages_ba_test[idx], " in radius ", selected_diff)
                diff_range = np.delete(diff_range, np.where(diff_range == selected_diff)[0][0])
                if len(diff_range) == 0:
                    print("zero ref images in any radius of age: ", self.ages_ba_test[idx])
                    print("ABORTING...")
                    exit()
            else:
                if not on_first:
                    if DEBUG_DATASET:
                        print("found ref images for age ", self.ages_ba_test[idx], " in radius ", selected_diff)
                not_found = False

        if self.use_embs:
            base_potential_idxs = idxs[0]
            near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[base_potential_idxs], axis=1))
            actual_idxs = base_potential_idxs[near_ne_idxs_of_idxs][0:self.embs_knn]
        else:
            actual_idxs = idxs[0]

        # import pdb
        # pdb.set_trace()
        
        # we found a diff we have one or more candidates. In case we have less than the needed 
        # num of references, we will replicate (probablistically). Else, we can just sample the num of references 
        # of unique references.
        if len(actual_idxs) < self.num_references:
            # import pdb
            # pdb.set_trace()
            if DEBUG_DATASET:
                print(self.ages_ba_test[idx], actual_idxs, selected_diff, diff_range)
                print("not enough ref images in radius! replicating")
            selected_idxs = np.random.choice(actual_idxs, self.num_references)
            return selected_idxs
        else:
            selected_idxs = np.random.choice(actual_idxs, self.num_references, replace=False)
            return selected_idxs

    def __len__(self):
        return len(self.batst_set_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        orig_image = self.batst_set_images[idx]

        orig_image = Image.fromarray(orig_image)
        
        image = self.transform(orig_image)
        metadata = json.loads(self.batst_set_metadata[idx])

        age = int(metadata['age'])
        label = age

        ref_image_idx = self._find_ref_image(idx)

        


        ref_image_arr = self.batrn_set_images[ref_image_idx]
        ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
        ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
        ref_image_arr_metadata = [json.loads(self.batrn_set_metadata[idx_i]) for idx_i in ref_image_idx]
        ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
        
        # import pdb
        # pdb.set_trace()

        ref_image_label = ref_image_arr_age[0] #// self.age_interval

        scaled_radius = self.age_diff_learn_radius_hi #// self.age_interval

        image_vec = torch.stack(tuple([image] + ref_image_arr))

        pair_label =  label - ref_image_label + scaled_radius

        sample = {'image_vec': image_vec, 'label': pair_label, 'age_diff': age - ref_image_arr_age[0], 'age_ref': ref_image_arr_age[0]}
        
        return sample





class AgeDiffMimicDiffDataset(Dataset):
    def __init__(self,
                batrn_set_images,               # basic aligned train images 
                batrn_set_metadata,             # basic aligned train metadata (labels)
                batst_set_images,               # basic aligned test images
                batst_set_metadata,             # basic aligned test metadata (labels)
                batst_set_indexes,              # 
                im2age_map_batst,               # image-to-age map according to original age est. model
                min_age,
                age_interval,
                max_age,
                transform=None,
                copies=1,
                age_radius=3,
                num_references=1,
                embs_trn=None,
                embs_vld=None,
                use_embs=False,
                embs_knn=1,
                embs_normalize=True):
        
        self.batrn_set_images = batrn_set_images[:, :, :, [2, 1, 0]]
        self.batrn_set_metadata = batrn_set_metadata

        self.batst_set_images = batst_set_images[:, :, :, [2, 1, 0]]
        self.batst_set_metadata = batst_set_metadata

        self.batst_set_indexes = batst_set_indexes
        
        self.im2age_map_batst = im2age_map_batst
        
        self.min_age = min_age
        self.age_interval = age_interval
        self.max_age = max_age
        self.transform = transform
        self.copies = copies
        self.age_radius = age_radius

        self.use_embs = use_embs
        self.embs_knn = embs_knn
        self.embs_normalize = embs_normalize

        raw_embs_trn = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        raw_embs_vld = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])
        # normalize embeddings
        if self.embs_normalize:
            self.embs_trn_rsh = arr_safe_norm(raw_embs_trn)
            self.embs_vld_rsh = arr_safe_norm(raw_embs_vld) 
        else:
            self.embs_trn_rsh = raw_embs_trn
            self.embs_vld_rsh = raw_embs_vld

        self.num_references = num_references

        self.total_diffs = []
        self.len_of_potential_refs = []

        # additional attributes

        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        
    def _find_ref_image(self, idx):
        #############################
        # Getting ref age which is in the diff according to model 
        # and finding all relevant indexes
        idxs = []
        # age predicted on image by model
        query_est_age = self.im2age_map_batst[str(self.batst_set_indexes[idx])]
        # round in order to get an actual age
        ref_age = np.round(query_est_age)
        # automatically fix in case we are out of range
        if ref_age > self.max_age - 1:
            ref_age = self.max_age - 1
        elif ref_age < self.min_age + 1:
            ref_age = self.min_age + 1
        # get all idxs is ref age
        idxs = np.where(self.ages_ba_train == ref_age)[0]

        if len(idxs) == 0:
            print("Not ref found - adding some small noise")
        
        # in case no refs found, no choice but to get other ref - for different diff 
        while len(idxs) == 0:
            # take a sample in radius around the original point
            ref_age = np.round(query_est_age + np.random.normal(0, 2))
            # automatically fix in case we are out of range
            if ref_age > self.max_age - 1:
                ref_age = self.max_age - 1
            elif ref_age < self.min_age + 1:
                ref_age = self.min_age + 1
            idxs = np.where(self.ages_ba_train == ref_age)[0]

        if DEBUG_DATASET:
            tst_metadata = json.loads(self.batst_set_metadata[idx])
            query_age = int(tst_metadata['age'])

        if ANALYZE_DATASET:
            self.total_diffs.append(query_age-ref_age)
            self.len_of_potential_refs.append(len(idxs))    

        if self.use_embs:
            base_potential_idxs = idxs

            near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[self.batst_set_indexes[idx]] - self.embs_trn_rsh[base_potential_idxs], axis=1))
            actual_idxs = base_potential_idxs[near_ne_idxs_of_idxs][0:self.embs_knn]
        else:
            actual_idxs = idxs

        # import pdb
        # pdb.set_trace()
        if DEBUG_REF_SIMILARITY:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, len(actual_idxs) + 1, figsize=(15, 10))
            ax[0].imshow(self.batst_set_images[idx])#.reshape((224,224,3)))
            for i in range(len(actual_idxs)):
                ax[i+1].imshow(self.batrn_set_images[actual_idxs[i]])#.reshape((224,224,3)))
            q_r_distance = np.linalg.norm(self.embs_vld_rsh[self.batst_set_indexes[idx]] - self.embs_trn_rsh[actual_idxs], axis=1)
            print(f'q-r distance: {q_r_distance}')
            # print(self.embs_vld_rsh[idx])
            # for i in actual_idxs:
            #     print(self.embs_trn_rsh[i])
            plt.show()

        if len(actual_idxs) < self.num_references:
            if DEBUG_DATASET:
                print(self.ages_ba_test[idx], actual_idxs, query_age-ref_age, diff_range)
                print("not enough ref images in radius! replicating")
            selected_idxs = np.random.choice(actual_idxs, self.num_references)
            if DEBUG_REF_SIMILARITY:
                print(f"test idx = {str(self.batst_set_indexes[idx])}")
                print(f"ref age {ref_age}")
                print(f"train idxs refs = {selected_idxs}")
            return selected_idxs
        else:
            selected_idxs = np.random.choice(actual_idxs, self.num_references, replace=False)
            if DEBUG_REF_SIMILARITY:
                print(f"test idx = {str(self.batst_set_indexes[idx])}")
                print(f"ref age {ref_age}")
                print(f"train idxs refs = {selected_idxs}")
            return selected_idxs

    def __len__(self):
        return len(self.batst_set_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        orig_image = self.batst_set_images[idx]

        orig_image = Image.fromarray(orig_image)
        
        image = self.transform(orig_image)
        metadata = json.loads(self.batst_set_metadata[idx])

        age = int(metadata['age'])
        label = age

        ref_image_idx = self._find_ref_image(idx)

        


        ref_image_arr = self.batrn_set_images[ref_image_idx]
        ref_image_arr = [Image.fromarray(ref_image) for ref_image in ref_image_arr]
        ref_image_arr = [self.transform(ref_image) for ref_image in ref_image_arr]
        ref_image_arr_metadata = [json.loads(self.batrn_set_metadata[idx_i]) for idx_i in ref_image_idx]
        ref_image_arr_age = [int(ref_image_metadata['age']) for ref_image_metadata in ref_image_arr_metadata]
        
        # import pdb
        # pdb.set_trace()

        ref_image_label = ref_image_arr_age[0] #// self.age_interval

        scaled_radius = self.age_radius #// self.age_interval

        image_vec = torch.stack(tuple([image] + ref_image_arr))

        if np.abs(label - ref_image_label) <= self.age_radius:
            pair_label =  label - ref_image_label + scaled_radius
        elif label - ref_image_label > self.age_radius:
            pair_label = 2*scaled_radius
        elif label - ref_image_label < -self.age_radius:
            pair_label = 0
        else:
            print("bug - aborting")
            exit()

        sample = {'image_vec': image_vec, 'label': pair_label, 'age_diff': age - ref_image_arr_age[0], 'age_ref': ref_image_arr_age[0]}
        
        return sample

    def get_stats(self):
        return self.total_diffs, self.len_of_potential_refs


def get_error_constrained_dataset(orig_dataset_images, orig_dataset_metadata, age_diff_learn_radius_lo, age_diff_learn_radius_hi, im2age_map_batst):
    filtered_dataset_images = []
    filtered_dataset_metadata = []
    orig_indexes = []
    for i in range(len(orig_dataset_metadata)):
        age = int(json.loads(orig_dataset_metadata[i])['age'])
        pred_age = im2age_map_batst[str(i)] 
        if age_diff_learn_radius_lo <= abs(pred_age - age) <= age_diff_learn_radius_hi:
            # import pdb
            # pdb.set_trace()
            filtered_dataset_images.append(orig_dataset_images[i])
            filtered_dataset_metadata.append(orig_dataset_metadata[i])
            orig_indexes.append(i)
    
    return np.stack(filtered_dataset_images),np.array(filtered_dataset_metadata), orig_indexes
            