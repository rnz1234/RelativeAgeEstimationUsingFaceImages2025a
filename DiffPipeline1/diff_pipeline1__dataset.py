import json 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class DiffPipeline1Dataset(Dataset):
    def __init__(self,
                dataset_type,                   # train / test
                batrn_set_images,               # basic aligned train images 
                batrn_set_metadata,             # basic aligned train metadata (labels)
                batst_set_images,               # basic aligned test images
                batst_set_metadata,             # basic aligned test metadata (labels)
                batst_set_idx,                  # basic aligned test indexes
                train_test_ratio,               # for batst, what is the ratio of created diff train & test sets
                im2age_map_batst,               # image-to-age map according to original age est. model
                min_age,
                age_interval,
                max_age,
                transform=None,
                copies=1,
                age_radius=3,
                dataset_size_factor=1):

        self.dataset_type = dataset_type
        
        self.batrn_set_images = batrn_set_images[:, :, :, [2, 1, 0]]
        self.batrn_set_metadata = batrn_set_metadata

        tst_dataset_size = len(batst_set_images)
        if self.dataset_type == "train":
            batst_set_images_trn_subset = batst_set_images[:int(train_test_ratio*tst_dataset_size)]
            self.actual_batst_set_images = batst_set_images_trn_subset[:, :, :, [2, 1, 0]]
            self.actual_batst_set_metadata = batst_set_metadata[:int(train_test_ratio*tst_dataset_size)]
            self.actual_batst_set_idx = batst_set_idx[:int(train_test_ratio*tst_dataset_size)]
        else:
            batst_set_images_test_subset = batst_set_images[int(train_test_ratio*tst_dataset_size):]
            self.actual_batst_set_images = batst_set_images_test_subset[:, :, :, [2, 1, 0]]
            self.actual_batst_set_metadata = batst_set_metadata[int(train_test_ratio*tst_dataset_size):]
            self.actual_batst_set_idx = batst_set_idx[int(train_test_ratio*tst_dataset_size):]

        self.im2age_map_batst = im2age_map_batst
        
        self.min_age = min_age
        self.age_interval = age_interval
        self.max_age = max_age
        self.transform = transform
        self.copies = copies
        self.age_radius = age_radius
        self.dataset_size_factor = dataset_size_factor

        # additional attributes

        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        self.ages_ba_test = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.actual_batst_set_metadata])

    def _find_ref_image(self, idx):
        #############################
        # Getting ref age which is in the diff according to model 
        # and finding all relevant indexes
        idxs = []
        # age predicted on image by model
        query_est_age = self.im2age_map_batst[str(self.actual_batst_set_idx[idx])]
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
        return len(self.actual_batst_set_metadata) * self.dataset_size_factor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_r = idx % len(self.actual_batst_set_metadata)

        orig_image = self.actual_batst_set_images[idx_r]

        orig_image = Image.fromarray(orig_image)
        
        image = self.transform(orig_image)
        metadata = json.loads(self.actual_batst_set_metadata[idx_r])

        age = int(metadata['age'])
        label = age

        ref_image_idx = self._find_ref_image(idx_r)



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