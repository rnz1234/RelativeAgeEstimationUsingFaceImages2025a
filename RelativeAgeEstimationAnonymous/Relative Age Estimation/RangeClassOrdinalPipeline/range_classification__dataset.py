import json 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class RangeClassificationSameUniformDiffDataset(Dataset):
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
                age_diff_learn_radius_hi=4):
        
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

        self.group0_upper = 4#10
        self.group1_upper = 10#20
        self.group2_upper = 100
            

        # additional attributes

        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.data_set_metadata])
        
    def _find_ref_image(self, idx):
        selected_grp = np.random.choice([0,1,2], 1, replace=False)
        
        if selected_grp == 0:
            idxs = np.where((0 <= abs(self.ages - self.ages[idx])) & (abs(self.ages - self.ages[idx]) < self.group0_upper))
            selected_idxs = np.random.choice(idxs[0], 1, replace=False)
        elif selected_grp == 1:
            idxs = np.where((self.group0_upper <= abs(self.ages - self.ages[idx])) & (abs(self.ages - self.ages[idx]) <= self.group1_upper))
            selected_idxs = np.random.choice(idxs[0], 1, replace=False)
        elif selected_grp == 2:
            idxs = np.where((self.group1_upper < abs(self.ages - self.ages[idx])) & (abs(self.ages - self.ages[idx]) <= self.group2_upper))
            selected_idxs = np.random.choice(idxs[0], 1, replace=False)
        
        return selected_idxs

        # #idxs = self.ages
        # ref_age = np.round(self.ages[idx] + np.random.normal(0, 3))
        # # automatically fix in case we are out of range
        # if ref_age > self.max_age - 1:
        #     ref_age = self.max_age - 1
        # elif ref_age < self.min_age + 1:
        #     ref_age = self.min_age + 1
        # # get all idxs is ref age
        # idxs = np.where(self.ages == ref_age)[0]

        # if len(idxs) == 0:
        #     print("Not ref found - adding some small noise")
        
        # # in case no refs found, no choice but to get other ref - for different diff 
        # while len(idxs) == 0:
        #     # take a sample in radius around the original point
        #     ref_age = np.round(self.ages[idx] + np.random.normal(0, 3))
        #     # automatically fix in case we are out of range
        #     if ref_age > self.max_age - 1:
        #         ref_age = self.max_age - 1
        #     elif ref_age < self.min_age + 1:
        #         ref_age = self.min_age + 1
        #     idxs = np.where(self.ages == ref_age)[0]

        # selected_idxs = np.random.choice(idxs, 1)
        # return selected_idxs

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

        if np.abs(label - ref_image_label) < self.group0_upper:
            pair_label = 0
        elif np.abs(label - ref_image_label) <= self.group1_upper:
            pair_label = 1
        # elif np.abs(label - ref_image_label) < 10:
        #     pair_label = 2
        else:
            pair_label = 2#3

        sample = {'image_vec': image_vec, 'label': pair_label, 'age_diff': age - ref_image_arr_age[0], 'age_ref': ref_image_arr_age[0]}
        
        return sample



class RangeClassificationMixedUniformDiffDataset(Dataset):
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
                age_diff_learn_radius_hi=4):
        
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

        # additional attributes
        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        self.ages_ba_test = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batst_set_metadata])
        
    def _find_ref_image(self, idx):
        idxs = self.ages_ba_train
        selected_idxs = np.random.choice(idxs[0], 1, replace=False)
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

        if np.abs(label - ref_image_label) < 2:
            pair_label = 0
        elif np.abs(label - ref_image_label) <= 4:
            pair_label = 1
        # elif np.abs(label - ref_image_label) < 10:
        #     pair_label = 2
        else:
            pair_label = 2#3

        sample = {'image_vec': image_vec, 'label': pair_label, 'age_diff': age - ref_image_arr_age[0], 'age_ref': ref_image_arr_age[0]}
        
        return sample





class RangeClassificationMimicDiffDataset(Dataset):
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
                age_radius=3):
        
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

        self.group0_upper = 4#10
        self.group1_upper = 10#20
        self.group2_upper = 100

        # additional attributes

        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        
    def _find_ref_image(self, idx):
        #############################
        # Getting ref age which is in the diff according to model 
        # and finding all relevant indexes
        idxs = []
        # age predicted on image by model
        try:
            query_est_age = self.im2age_map_batst[str(self.batst_set_indexes[idx])]
        except IndexError:
            import pdb
            pdb.set_trace()
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
            

        #############################
        # Select from the references (randomly)
        selected_idxs = np.random.choice(idxs, 1)
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

        # import pdb
        # pdb.set_trace()

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

        if np.abs(label - ref_image_label) < self.group0_upper:
            pair_label = 0
        elif np.abs(label - ref_image_label) <= self.group1_upper:
            pair_label = 1
        # elif np.abs(label - ref_image_label) < 10:
        #     pair_label = 2
        else:
            pair_label = 2#3

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
    
    # import pdb
    # pdb.set_trace()
    
    return np.stack(filtered_dataset_images),np.array(filtered_dataset_metadata), orig_indexes
            