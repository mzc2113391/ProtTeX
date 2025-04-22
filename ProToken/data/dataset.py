"""create dataset for training and testing"""
import os
import sys
import pickle as pkl

import numpy as np
# use pool.map to parallelize the data loading
import multiprocessing
from functools import partial
import concurrent
import datetime
import warnings

protoken_input_feature_names = ["seq_mask", "decoder_seq_mask", "fake_aatype", "residue_index",
                                "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
                                "backbone_affine_tensor", "backbone_affine_tensor_label", "torsion_angles_sin_cos", "torsion_angles_mask", "atom14_atom_exists"]

protoken_dtype_dic = {"aatype": np.int32,
                      "fake_aatype": np.int32,
                      "seq_mask": np.bool_,
                      "decoder_seq_mask": np.bool_,
                      "residue_index": np.int32,
                      "template_all_atom_positions": np.float32,
                      "template_all_atom_masks": np.bool_,
                      "template_pseudo_beta": np.float32,
                      "backbone_affine_tensor": np.float32,
                      "backbone_affine_tensor_label": np.float32,
                      "torsion_angles_sin_cos": np.float32,
                      "torsion_angles_mask": np.bool_,
                      "atom14_atom_exists": np.float32,
                      "dist_gt_perms": np.float32,
                      "dist_mask_perms": np.int32,
                      "perms_padding_mask": np.bool_,}

# Define a function to read and deserialize a single file
def read_file(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
# Function to read files in parallel using ThreadPoolExecutor 
def read_files_in_parallel(file_paths, num_parallel_worker=32):
    # Using a with statement ensures threads are cleaned up promptly
    # time0 = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_worker) as executor:
        # Map the read_file function to each file path
        results = list(executor.map(read_file, file_paths))
    # time1 = datetime.datetime.now()
    # time_consumings = (time1 - time0).total_seconds()
    # file_size = sum([os.path.getsize(file_path) for file_path in file_paths])
    # speed_io_Gps = file_size/1024/1024/1024/time_consumings
    return results # , time_consumings, speed_io_Gps

def get_crop_idx(feature, crop_len):
    seq_len = np.sum(feature['seq_mask'])
    if seq_len <= crop_len:
        crop_start_idx = 0
        crop_end_idx = crop_len
    else:
        crop_start_idx = np.random.randint(0, seq_len - crop_len + 1)
        crop_end_idx = crop_start_idx + crop_len
    
    return crop_start_idx, crop_end_idx

def pad_to_len(arr, target_len, axis=0, pad_value=0):
    pad_len = target_len - arr.shape[axis]
    if pad_len > 0:
        pad_shape = list(arr.shape)
        pad_shape[axis] = pad_len
        arr = np.concatenate([arr, np.ones(pad_shape, dtype=arr.dtype) * pad_value], axis=axis)
    return arr

def random_crop_feature(feature, crop_start_idx, crop_end_idx, crop_len):
    # feature has been padded to a certain length
    new_feature = {}
    
    new_feature['aatype'] = feature['aatype'][:, crop_start_idx:crop_end_idx]
    new_feature['seq_mask'] = feature['seq_mask'][:, crop_start_idx:crop_end_idx]
    new_feature['fake_aatype'] = feature['fake_aatype'][:, crop_start_idx:crop_end_idx]
    new_feature['residue_index'] = feature['residue_index'][:, crop_start_idx:crop_end_idx]
    new_feature['template_all_atom_masks'] = feature['template_all_atom_masks'][:, crop_start_idx:crop_end_idx, :]
    new_feature['template_all_atom_positions'] = feature['template_all_atom_positions'][:, crop_start_idx:crop_end_idx, :, :]
    new_feature['template_pseudo_beta'] = feature['template_pseudo_beta'][:, crop_start_idx:crop_end_idx, :]
    new_feature['backbone_affine_tensor'] = feature['backbone_affine_tensor'][:, crop_start_idx:crop_end_idx, :]
    new_feature['backbone_affine_tensor_label'] = feature['backbone_affine_tensor_label'][:, crop_start_idx:crop_end_idx, :]
    new_feature['torsion_angles_sin_cos'] = feature['torsion_angles_sin_cos'][:, crop_start_idx:crop_end_idx, :]
    new_feature['torsion_angles_mask'] = feature['torsion_angles_mask'][:, crop_start_idx:crop_end_idx, :]

    new_feature['atom14_atom_exists'] = feature['atom14_atom_exists'][:, crop_start_idx:crop_end_idx, :]
    if 'dist_gt_perms' in feature.keys():
        new_feature['dist_gt_perms'] = feature['dist_gt_perms'][:, :, crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]
    if 'dist_mask_perms' in feature.keys():
        new_feature['dist_mask_perms'] = feature['dist_mask_perms'][:, :, crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]
    if 'ca_coords' in feature.keys():
        new_feature['ca_coords'] = feature['ca_coords'][:, crop_start_idx:crop_end_idx, :]
    new_feature['perms_padding_mask'] = feature['perms_padding_mask']
        
    if 'name' in feature:
        new_feature['name'] = feature['name']
    if 'TM-score' in feature:
        new_feature['TM-score'] = feature['TM-score']
        
    ret = {k: pad_to_len(v, crop_len, axis=1) for k, v in new_feature.items() if 'affine' not in k}
    #### different padding method for backbone affine tensor
    if crop_len > new_feature['backbone_affine_tensor'].shape[1]:
        ret.update(
            {'backbone_affine_tensor': 
                np.concatenate([new_feature['backbone_affine_tensor'], np.array([[1,0,0,0,0,0,0],] * (crop_len - new_feature['backbone_affine_tensor'].shape[1]), dtype=np.float32)[None, ...]], axis=1), 
            'backbone_affine_tensor_label':
                np.concatenate([new_feature['backbone_affine_tensor_label'], np.array([[1,0,0,0,0,0,0],] * (crop_len - new_feature['backbone_affine_tensor_label'].shape[1]), dtype=np.float32)[None, ...]], axis=1), 
            }
        )
    else:
        ret.update({'backbone_affine_tensor': new_feature['backbone_affine_tensor'], 
                    'backbone_affine_tensor_label': new_feature['backbone_affine_tensor_label']})
    return ret

def load_train_data_pickle(name_list, 
                           start_idx, 
                           end_idx,
                           num_parallel_worker=32,
                           random_crop=False,
                           crop_len=256,
                           tmscore_threshold=(0.90, 0.975), 
                           decoder_seq_mask_ratio=0.05,
                           adversarial=False,
                           n_sample_per_device=8):

    name_list_trunk = name_list[start_idx: end_idx]
    data_batch_load = read_files_in_parallel(name_list_trunk, num_parallel_worker = num_parallel_worker)

    def choose_a_feature(d, data_column=None):
        keys = ['gt_feature', 'recon_feature', 'af_feature']
        confidence = np.array([1.0, float(d['tmscore_recon']), float(d['tmscore_af'])])
        confidence = (confidence - tmscore_threshold[0]) / (tmscore_threshold[1] - tmscore_threshold[0])
        confidence = np.clip(confidence, 0.0, 1.0)
        confidence[0] = 1.0
        select_key = np.random.choice(keys, p=confidence / np.sum(confidence))
        d_ = d[select_key]
        d_ = d_[0] if isinstance(d_, tuple) else d_
        if data_column is None:
            return d_
        else:
            return d_[data_column]
    
    data_batch = [choose_a_feature(d) for d in data_batch_load]
    backbone_affine_tensor_label_batch = [choose_a_feature(d, 'backbone_affine_tensor') for d in data_batch_load]
    for label, d in zip(backbone_affine_tensor_label_batch, data_batch_load): 
        d.update({'backbone_affine_tensor_label': label})

    # crop features
    if random_crop:
        crop_indexes = [get_crop_idx(d, crop_len) for d in data_batch]
        if adversarial:
            crop_indexes = np.array(crop_indexes).reshape(-1, n_sample_per_device, 2)
            crop_indexes[:, n_sample_per_device//2:] = crop_indexes[:, :n_sample_per_device//2]
            crop_indexes = crop_indexes.reshape(-1, 2)
        data_batch = [random_crop_feature(d, crop_start_idx, crop_end_idx, crop_len)
            for d, (crop_start_idx, crop_end_idx) in zip(data_batch, crop_indexes)]
    
    # concate batch data
    batch_feature = {}
    for k in protoken_input_feature_names:
        if (k in data_batch[0].keys()):
            batch_feature[k] = np.concatenate([data[k] for data in data_batch], axis=0)
    
    # make decoder seq mask, N, C, middle 5%-10% residues
    def make_decoder_seq_mask(seq_len):
        seq_len_3 = int(round(seq_len / 3))
        seq_mask = np.ones(crop_len, dtype=np.bool_)
        seq_mask[seq_len:] = 0
        mask_len = int(round(seq_len_3 * decoder_seq_mask_ratio))
        
        mask_indexes = np.concatenate([np.random.randint(0, seq_len_3, mask_len),
                                       np.random.randint(seq_len_3, seq_len-seq_len_3, mask_len),
                                       np.random.randint(seq_len-seq_len_3, seq_len, mask_len)])
        
        seq_mask[mask_indexes] = 0
        return seq_mask
    
    batch_feature['decoder_seq_mask'] = np.array(
        [make_decoder_seq_mask(min(crop_len, int(data['raw_seq_len']))) for data in data_batch_load])
    
    # convert data dtype
    for k in protoken_input_feature_names: batch_feature[k] = batch_feature[k].astype(protoken_dtype_dic[k])
                
    return batch_feature