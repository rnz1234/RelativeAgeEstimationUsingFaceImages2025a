import numpy as np
import json
from scipy.stats import gaussian_kde


KDE_BW_METHOD = 'silverman' #  This can be None, ‘scott’, ‘silverman’. None means also scott
print(f"Using KDE BW Method: {KDE_BW_METHOD}")

# return a measurement between 0 and 1. 1 == all lines are unique
def uniqueness(dataset_file_path):
    f = open(dataset_file_path, 'r')
    lines = f.readlines()
    lines_v = np.array([np.array(line.split('\n')[0][1:-1].split(), dtype=int) for line in lines])
    lines_vu = np.unique(lines_v, axis=0)

    return len(lines_vu) / len(lines)

# error analysis 
def get_statistics(dataset_metadata, dataset_indexes, im2age_map_batst):
    err_vec = []
    for i in range(len(dataset_metadata)):
        age_real = int(json.loads(dataset_metadata[i])['age'])
        age_pred = im2age_map_batst[str(dataset_indexes[i])]
        err_vec.append(age_real-age_pred)

    err_vec = np.array(err_vec)

    kde = gaussian_kde(dataset=err_vec, bw_method=KDE_BW_METHOD)

    return {
        "mean" : np.mean(err_vec),
        "std" : np.std(err_vec),
        "data" : err_vec,
        "kde": kde
    }


# error analysis 
def get_statistics_range(dataset_metadata, dataset_indexes, im2age_map_batst, age_range_min, age_range_max):
    err_vec = []
    count = 0
    for i in range(len(dataset_metadata)):
        age_real = int(json.loads(dataset_metadata[i])['age'])
        if (age_real >= age_range_min) and (age_real <= age_range_max):
            count += 1
            age_pred = im2age_map_batst[str(dataset_indexes[i])]
            err_vec.append(age_real-age_pred)

    err_vec = np.array(err_vec)

    kde = gaussian_kde(dataset=err_vec, bw_method=KDE_BW_METHOD)


    return {
        "mean" : np.mean(err_vec),
        "std" : np.std(err_vec),
        "data" : err_vec,
        "count" : count,
        "kde": kde
    }

# error analysis 
def get_statistics_range_idx(dataset_metadata, dataset_str_indexes, im2age_map_batst, age_range_min, age_range_max):
    err_vec = []
    count = 0
    for i in dataset_str_indexes:
        age_real = int(json.loads(dataset_metadata[int(i)])['age'])
        if (age_real >= age_range_min) and (age_real <= age_range_max):
            count += 1
            age_pred = im2age_map_batst[i]
            err_vec.append(age_real-age_pred)

    err_vec = np.array(err_vec)

    kde = gaussian_kde(dataset=err_vec, bw_method=KDE_BW_METHOD)
    
    return {
        "mean" : np.mean(err_vec),
        "std" : np.std(err_vec),
        "data" : err_vec,
        "count" : count,
        "kde": kde
    }