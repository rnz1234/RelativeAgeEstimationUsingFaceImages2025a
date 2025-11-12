import json 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

USE_RA2APED = False #True
SELECT_WITH_EMBS = False #True
USE_K_NN_EMBS = False
DIFF_SELECT = False #False
EMBS_PERCENTILE = 10
K_NN_EMBS = 7
K_NN_EMBS_DROPS_WITH_DISTANCE = True
USE_GAUSSIAN_DIST_WITH_RA2APED = True

RA2APED_MEAN_IDX = 0
RA2APED_STD_IDX = 1
RA2APED_MIN_IDX = 2
RA2APED_MAX_IDX = 3
RA2APED = {67: [-9.705440521240234,
  1.8214607238769531,
  -11.526901245117188,
  -7.883979797363281],
 59: [-6.036374568939209,
  2.7089477024209385,
  -9.996265411376953,
  -1.4329109191894531],
 60: [-4.948554992675781,
  2.3303184310400593,
  -10.302749633789062,
  -2.36041259765625],
 63: [-7.456778844197591,
  2.728812529683961,
  -11.161113739013672,
  -3.353179931640625],
 58: [-6.570741480047053,
  3.7120193620729083,
  -13.593929290771484,
  1.0713310241699219],
 61: [-5.56124013264974,
  1.955025831238078,
  -9.598270416259766,
  -3.1699256896972656],
 64: [-6.929258074079241,
  2.5240316603292974,
  -10.559303283691406,
  -3.3546142578125],
 57: [-3.332122224749941,
  1.8678924352711825,
  -7.6173553466796875,
  -0.13737106323242188],
 62: [-7.009473419189453,
  2.684997772473596,
  -10.315116882324219,
  -0.2279815673828125],
 55: [-4.069808096255896,
  4.010490269736009,
  -19.811756134033203,
  1.4511680603027344],
 56: [-4.285973492790671,
  3.42174272483469,
  -11.539604187011719,
  1.9654655456542969],
 54: [-3.2793317621404476,
  3.4479554278161677,
  -11.60806655883789,
  4.759010314941406],
 53: [-2.5906709877841445,
  3.474010152222265,
  -12.373516082763672,
  4.743236541748047],
 51: [-1.9036973262655323,
  4.004738942479048,
  -26.23637580871582,
  8.261405944824219],
 52: [-2.3334673006240636,
  3.352521028120202,
  -9.602531433105469,
  5.959072113037109],
 50: [-2.0087404251098633,
  3.4712012614120353,
  -14.799991607666016,
  6.231998443603516],
 48: [-1.6385718885674534,
  3.5993739121416297,
  -12.568031311035156,
  8.919933319091797],
 49: [-1.9714899682379388,
  3.5208942804220857,
  -12.56442642211914,
  5.6267547607421875],
 47: [-1.7900548974672954,
  3.578093552545197,
  -16.84859848022461,
  5.988414764404297],
 46: [-0.864400707307409,
  3.3495274681447933,
  -8.778961181640625,
  8.908443450927734],
 45: [-0.9387274768864997,
  3.872801136328194,
  -17.142539978027344,
  8.675254821777344],
 44: [-0.3421648262832571,
  3.6259940936593833,
  -9.148967742919922,
  9.050128936767578],
 37: [-0.2176951772158908,
  3.6620053230224667,
  -14.455244064331055,
  10.263179779052734],
 43: [-0.6163987037830783,
  3.4040286146024785,
  -11.908077239990234,
  8.518760681152344],
 42: [-0.6210196757177163,
  3.407996708951666,
  -9.084171295166016,
  8.984386444091797],
 39: [-0.4201058217572042,
  3.4420840192326794,
  -12.480327606201172,
  9.9415283203125],
 41: [-0.008910051981608073,
  3.291745735878087,
  -8.379894256591797,
  10.21963119506836],
 36: [0.10233390504035397,
  3.768019327383012,
  -14.560007095336914,
  7.558258056640625],
 40: [-0.37822549129245286,
  3.387220493856239,
  -12.380962371826172,
  8.108448028564453],
 38: [-0.3090576469676645,
  3.512990366008157,
  -15.691961288452148,
  8.148078918457031],
 35: [-0.3740841440054087,
  3.4081913661503185,
  -14.027814865112305,
  8.07907485961914],
 34: [0.15381867622180972,
  3.8685432902717998,
  -11.771440505981445,
  9.43313980102539],
 65: [-9.60733731587728,
  1.1806614435459473,
  -10.64004898071289,
  -7.106624603271484],
 33: [0.27107300556881325,
  3.705844660267562,
  -10.51688003540039,
  15.272884368896484],
 32: [0.7070169102061878,
  3.763509550359151,
  -12.881973266601562,
  12.890594482421875],
 31: [0.3346003947719451,
  3.7676618284814265,
  -9.049022674560547,
  11.00387954711914],
 28: [-0.08516272631558505,
  3.2756657662610364,
  -8.99229621887207,
  8.836936950683594],
 25: [-0.29366581447738,
  2.979438782064925,
  -5.840007781982422,
  16.255313873291016],
 30: [0.07533530162588897,
  3.569254897279265,
  -8.116556167602539,
  9.642879486083984],
 29: [-0.03143478301634271,
  3.70681576935861,
  -8.82918930053711,
  18.644424438476562],
 26: [0.0885802593427835,
  3.252095331377487,
  -8.361648559570312,
  12.04473876953125],
 27: [-0.16824573730871623,
  3.2235484483341255,
  -6.879489898681641,
  9.036163330078125],
 24: [0.32593141198158265,
  2.945163928332622,
  -7.021884918212891,
  11.472522735595703],
 23: [0.6827482091318263,
  3.1481429429063725,
  -5.965669631958008,
  28.614425659179688],
 22: [0.703333513543749,
  2.7022578732315874,
  -5.004674911499023,
  10.510208129882812],
 21: [0.6280150004795619,
  2.389972917106088,
  -4.905935287475586,
  12.998363494873047],
 20: [0.7800321001827889,
  2.10680251020921,
  -3.0974864959716797,
  11.472879409790039],
 18: [1.1590198826145481,
  2.12818135245352,
  -1.953989028930664,
  18.21963882446289],
 19: [1.0277738486395942,
  2.042722166242734,
  -2.9615631103515625,
  8.289424896240234],
 16: [1.76933839586046,
  1.7790746853726271,
  0.03658485412597656,
  10.502212524414062],
 17: [1.2676798157069995,
  1.7830668858729262,
  -0.9630126953125,
  10.63882064819336]}

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
                embs=None,
                num_references=1):
        
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
        self.num_references = num_references

        # additional attributes

        self.ages = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.data_set_metadata])
        
    def _find_ref_image(self, idx):
        # import pdb
        # pdb.set_trace()
        if DIFF_SELECT:
            diff_range = np.unique(np.concatenate((-np.arange(self.age_diff_learn_radius_lo,self.age_diff_learn_radius_hi + 1), np.arange(self.age_diff_learn_radius_lo,self.age_diff_learn_radius_hi + 1))))
            
            idxs = []
            while len(idxs) == 0:
                selected_diff = np.random.choice(diff_range, 1, replace=False)
                idxs = np.where(self.ages - self.ages[idx] == selected_diff)[0]
        else:
            # import pdb
            # pdb.set_trace()
            if USE_RA2APED:
                if self.ages[idx] in RA2APED.keys():
                    if USE_GAUSSIAN_DIST_WITH_RA2APED:
                        idxs = []
                        while len(idxs) == 0:
                            selected_diff = np.round(np.random.normal(loc=RA2APED[self.ages[idx]][RA2APED_MEAN_IDX], scale=RA2APED[self.ages[idx]][RA2APED_STD_IDX]))
                            if selected_diff < RA2APED[self.ages[idx]][RA2APED_MIN_IDX]:
                                selected_diff = RA2APED[self.ages[idx]][RA2APED_MIN_IDX]
                            elif selected_diff > RA2APED[self.ages[idx]][RA2APED_MAX_IDX]:
                                selected_diff = RA2APED[self.ages[idx]][RA2APED_MAX_IDX]
                            idxs = np.where(self.ages - self.ages[idx] == selected_diff)[0]
                    else:
                        idxs = np.where((RA2APED[self.ages[idx]][RA2APED_MIN_IDX] <= self.ages - self.ages[idx]) & (self.ages - self.ages[idx] <= RA2APED[self.ages[idx]][RA2APED_MAX_IDX]))[0]
                else:
                    idxs = np.where((self.age_diff_learn_radius_lo <= abs(self.ages - self.ages[idx])) & (abs(self.ages - self.ages[idx]) <= self.age_diff_learn_radius_hi))[0]
            else:
                idxs = np.where((self.age_diff_learn_radius_lo <= abs(self.ages - self.ages[idx])) & (abs(self.ages - self.ages[idx]) <= self.age_diff_learn_radius_hi))[0]
            

        if SELECT_WITH_EMBS:
            if USE_K_NN_EMBS:
                if K_NN_EMBS_DROPS_WITH_DISTANCE:
                    diff_abs = int(np.abs(selected_diff))
                    actual_knn = int(np.floor(K_NN_EMBS / (1.6**diff_abs)))
                    near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[idxs], axis=1))[0:actual_knn]
                else:
                    near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[idxs], axis=1))[0:K_NN_EMBS]
            else:
                near_ne_idxs_of_idxs = np.where(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[idxs], axis=1) <= np.percentile(np.linalg.norm(self.embs_rsh[idx] - self.embs_rsh[idxs], axis=1), EMBS_PERCENTILE))[0]
            
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

        if len(near_ne_in_range) < self.num_references:
            selected_idxs = np.random.choice(near_ne_in_range, self.num_references)
        else:
            selected_idxs = np.random.choice(near_ne_in_range, self.num_references, replace=False)
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
                embs_vld=None,
                num_references=1):
        
        
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

        self.num_references = num_references


        # additional attributes
        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        self.ages_ba_test = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batst_set_metadata])
        
    def _find_ref_image(self, idx):
        if DIFF_SELECT:
            diff_range = np.unique(np.concatenate((-np.arange(self.age_diff_learn_radius_lo,self.age_diff_learn_radius_hi + 1), np.arange(self.age_diff_learn_radius_lo,self.age_diff_learn_radius_hi + 1))))
            
            idxs = []
            while len(idxs) == 0:
                selected_diff = np.random.choice(diff_range, 1, replace=False)
                idxs = np.where(self.ages_ba_train - self.ages_ba_test[idx] == selected_diff)[0]
        else:
            if USE_RA2APED:
                if self.ages_ba_test[idx] in RA2APED.keys():
                    if USE_GAUSSIAN_DIST_WITH_RA2APED:
                        idxs = []
                        while len(idxs) == 0:
                            selected_diff = np.round(np.random.normal(loc=RA2APED[self.ages[idx]][RA2APED_MEAN_IDX], scale=RA2APED[self.ages[idx]][RA2APED_STD_IDX]))
                            if selected_diff < RA2APED[self.ages[idx]][RA2APED_MIN_IDX]:
                                selected_diff = RA2APED[self.ages[idx]][RA2APED_MIN_IDX]
                            elif selected_diff > RA2APED[self.ages[idx]][RA2APED_MAX_IDX]:
                                selected_diff = RA2APED[self.ages[idx]][RA2APED_MAX_IDX]
                            idxs = np.where(self.ages_ba_train - self.ages_ba_test[idx] == selected_diff)[0]
                    else:
                        idxs = np.where((RA2APED[self.ages_ba_test[idx]][RA2APED_MIN_IDX] <= self.ages_ba_train - self.ages_ba_test[idx]) & (self.ages_ba_train - self.ages_ba_test[idx] <= RA2APED[self.ages_ba_test[idx]][RA2APED_MAX_IDX]))[0]
                else:
                    idxs = np.where((self.age_diff_learn_radius_lo <= abs(self.ages_ba_train - self.ages_ba_test[idx])) & (abs(self.ages_ba_train - self.ages_ba_test[idx]) <= self.age_diff_learn_radius_hi))[0]
            else:
                idxs = np.where((self.age_diff_learn_radius_lo <= abs(self.ages_ba_train - self.ages_ba_test[idx])) & (abs(self.ages_ba_train - self.ages_ba_test[idx]) <= self.age_diff_learn_radius_hi))[0]
            


        if SELECT_WITH_EMBS:
            if USE_K_NN_EMBS:
                if K_NN_EMBS_DROPS_WITH_DISTANCE:
                    diff_abs = int(np.abs(selected_diff))
                    actual_knn = int(np.floor(K_NN_EMBS / (1.6**diff_abs)))
                    near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1))[0:actual_knn]
                else:
                    near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1))[0:K_NN_EMBS]
            else:
                near_ne_idxs_of_idxs = np.where(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1) <= np.percentile(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1), EMBS_PERCENTILE))[0]
            

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
        if len(near_ne_in_range) < self.num_references:
            selected_idxs = np.random.choice(near_ne_in_range, self.num_references)
        else:
            selected_idxs = np.random.choice(near_ne_in_range, self.num_references, replace=False)
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
                embs_vld=None,
                num_references=1):
        
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

        self.num_references = num_references

        # additional attributes

        self.ages_ba_train = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batrn_set_metadata])
        self.ages_ba_test = np.array([int(json.loads(metadata_for_image)['age']) for metadata_for_image in self.batst_set_metadata])
        
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
            if USE_K_NN_EMBS:
                if K_NN_EMBS_DROPS_WITH_DISTANCE:
                    diff_abs = int(np.abs(self.ages_ba_test[idx]-ref_age))
                    actual_knn = int(np.floor(K_NN_EMBS / (1.6**diff_abs)))
                    near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1))[0:actual_knn]
                else:
                    near_ne_idxs_of_idxs = np.argsort(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1))[0:K_NN_EMBS]    
            else:
                near_ne_idxs_of_idxs = np.where(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1) <= np.percentile(np.linalg.norm(self.embs_vld_rsh[idx] - self.embs_trn_rsh[idxs], axis=1), EMBS_PERCENTILE))[0]
            

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
        if len(near_ne_in_range) < self.num_references:
            selected_idxs = np.random.choice(near_ne_in_range, self.num_references)
        else:
            selected_idxs = np.random.choice(near_ne_in_range, self.num_references, replace=False)
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
            