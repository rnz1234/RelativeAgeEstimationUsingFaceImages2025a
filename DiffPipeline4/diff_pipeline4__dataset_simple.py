import json 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

SELECT_WITH_EMBS = False
EMBS_PERCENTILE = 10 # 25
K_NN_EMBS = 5

class DiffPipeline4SameUniformDiffDataset(Dataset):
    def __init__(self,
                data_set_images,                # basic aligned * images 
                data_set_metadata,              # basic aligned * metadata (labels)
                min_age,
                age_interval,
                max_age,
                transform=None,
                copies=1,
                age_radius=4,
                age_diff_learn_radius_lo=2,
                age_diff_learn_radius_hi=4,
                embs=None):
        
        self.data_set_images = data_set_images[:, :, :, [2, 1, 0]]
        self.data_set_metadata = data_set_metadata

        self.min_age = min_age
        self.age_interval = age_interval
        self.max_age = max_age
        self.transform = transform
        self.copies = copies
        self.age_radius = age_radius
        self.age_diff_learn_radius_lo = age_diff_learn_radius_lo
        self.age_diff_learn_radius_hi = age_diff_learn_radius_hi
        self.embs_rsh = embs.reshape(embs.shape[0], embs.shape[2])

        # additional attributes

        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.data_set_metadata])
        
    def _find_ref_image(self, idx):
        idxs = np.where((self.age_diff_learn_radius_lo <= abs(self.ages - self.ages[idx])) & (abs(self.ages - self.ages[idx]) <= self.age_diff_learn_radius_hi))[0]
        
        
        if SELECT_WITH_EMBS:
            #near_ne_idxs_of_idxs = np.where(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[idxs], axis=1) <= np.percentile(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[idxs], axis=1), EMBS_PERCENTILE))[0]
            near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[idxs], axis=1))[0:K_NN_EMBS]

            near_ne_in_range = idxs[near_ne_idxs_of_idxs]
        else:
            near_ne_in_range = idxs
        #near_ne_in_range = np.intersect1d(near_ne, idxs)

        # cur_p = 25
        # while len(near_ne_in_range) == 0:
        #     print("no found on percentile {}, increasing by 5".format(cur_p))
        #     cur_p += 5
        #     near_ne = np.where(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh, axis=1) <= np.percentile(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh, axis=1), cur_p))[0]

        #     near_ne_in_range = np.intersect1d(near_ne, idxs[0])

        #     if cur_p == 100:
        #         break
        
        selected_idxs = np.random.choice(near_ne_in_range, 1, replace=False)
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



class DiffPipeline4MixedUniformDiffDataset(Dataset):
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
                age_radius=3,
                age_diff_learn_radius_lo=2,
                age_diff_learn_radius_hi=4,
                embs_trn=None,
                embs_vld=None):
        
        
        self.batrn_set_images = batrn_set_images[:, :, :, [2, 1, 0]]
        self.batrn_set_metadata = batrn_set_metadata

        self.batst_set_images = batst_set_images[:, :, :, [2, 1, 0]]
        self.batst_set_metadata = batst_set_metadata

        self.min_age = min_age
        self.age_interval = age_interval
        self.max_age = max_age
        self.transform = transform
        self.copies = copies
        self.age_radius = age_radius
        self.age_diff_learn_radius_lo = age_diff_learn_radius_lo
        self.age_diff_learn_radius_hi = age_diff_learn_radius_hi

        self.embs_trn_rsh = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        self.embs_vld_rsh = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])


        # additional attributes
        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        self.ages_ba_test = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batst_set_metadata])
        
    def _find_ref_image(self, idx):
        idxs = np.where((self.age_diff_learn_radius_lo <= abs(self.ages_ba_train - self.ages_ba_test[idx])) & (abs(self.ages_ba_train - self.ages_ba_test[idx]) <= self.age_diff_learn_radius_hi))[0]

        if SELECT_WITH_EMBS:
            #near_ne_idxs_of_idxs = np.where(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1) <= np.percentile(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1), EMBS_PERCENTILE))[0]
            near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1))[0:K_NN_EMBS]

            near_ne_in_range = idxs[near_ne_idxs_of_idxs]
        else:
            near_ne_in_range = idxs

        # near_ne_in_range = np.intersect1d(near_ne, idxs)

        # cur_p = 25
        # while len(near_ne_in_range) == 0:
        #     print("no found on percentile {}, increasing by 5".format(cur_p))
        #     cur_p += 5
        #     near_ne = np.where(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh, axis=1) <= np.percentile(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh, axis=1), cur_p))[0]

        #     near_ne_in_range = np.intersect1d(near_ne, idxs[0])

        #     if cur_p == 100:
        #         break
        
        selected_idxs = np.random.choice(near_ne_in_range, 1, replace=False)
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





class DiffPipeline4MimicDiffDataset(Dataset):
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
                embs_trn=None,
                embs_vld=None):
        
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

        self.embs_trn_rsh = embs_trn.reshape(embs_trn.shape[0], embs_trn.shape[2])
        self.embs_vld_rsh = embs_vld.reshape(embs_vld.shape[0], embs_vld.shape[2])

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
            

        if SELECT_WITH_EMBS:
            #near_ne_idxs_of_idxs = np.where(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1) <= np.percentile(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1), EMBS_PERCENTILE))[0]
            near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1))[0:K_NN_EMBS]

            near_ne_in_range = idxs[near_ne_idxs_of_idxs]
        else:
            near_ne_in_range = idxs

        # near_ne_in_range = np.intersect1d(near_ne, idxs[0])

        # # import pdb
        # # pdb.set_trace()

        # cur_p = 25
        # while len(near_ne_in_range) == 0:
        #     print("no found on percentile {}, increasing by 5".format(cur_p))
        #     cur_p += 5
        #     near_ne = np.where(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh, axis=1) <= np.percentile(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh, axis=1), cur_p))[0]

        #     near_ne_in_range = np.intersect1d(near_ne, idxs[0])

        #     if cur_p == 100:
        #         break
        
        selected_idxs = np.random.choice(near_ne_in_range, 1, replace=False)
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
            