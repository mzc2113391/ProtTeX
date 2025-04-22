from common.config_load import Config
import jax 
import jax.numpy as jnp
from flax import traverse_util
import numpy as np
from functools import partial

def logger(f, logger_info, flush=False):
    f.write(logger_info + "\n")
    print(logger_info)
    if (flush):
        f.flush()

def split_multiple_rng_keys(rng_key, num_keys):
    rng_keys = jax.random.split(rng_key, num_keys + 1)
    return rng_keys[:-1], rng_keys[-1]

def make_rng_dict(rng_key, dict_keys, num_rngs_per_key=1, squeeze=True):
    rng_dict = {}
    squeeze_op = lambda x: jnp.squeeze(x) if squeeze else x
    for k in dict_keys:
        rng_dict[k], rng_key = split_multiple_rng_keys(rng_key, num_rngs_per_key)
        rng_dict[k] = squeeze_op(rng_dict[k])
    return rng_dict, rng_key

def loss_logger(f, loss_dict, prefix=""):
    for k, v in loss_dict.items():
        if isinstance(v, dict):
            logger(f, "{}{}:".format(prefix, k))
            loss_logger(f, v, prefix=prefix + "\t")
        else:
            logger(f, "{}{}: {:.4f}".format(prefix, k, v))

def set_dropout_rate_config(d, dropout_rate):
    if isinstance(d, Config):
        d = d.__dict__
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, Config):
            d[k] = set_dropout_rate_config(v, dropout_rate)
        else:
            d[k] = dropout_rate if "dropout" in k else v    
    return Config(d)

def periodic_decay_weight_schedule(step, period, decay_time_scale, min_weight, max_weight):
    step, period, decay_time_scale = float(step), float(period), float(decay_time_scale)
    period_factor = (1.0 + np.cos(2 * np.pi * step / period)) / 2.0
    decay_factor = np.exp(-step / decay_time_scale)
    
    weight = decay_factor * (max_weight - min_weight) * period_factor + min_weight 
    
    return weight

def decay_weight_schedule(step, decay_time_scale, min_weight, max_weight):
    step, decay_time_scale = float(step), float(decay_time_scale)
    decay_factor = np.exp(-step / decay_time_scale)
    
    weight = decay_factor * (max_weight - min_weight) + min_weight
    
    return weight

def orgnize_name_list(train_name_list_bin, bins, 
                      batch_size, num_batches, n_sample_per_device,
                      p_scaling=1.0, adversarial=False):
    bin_prob = \
        np.array([train_name_list_bin[b]['size'] for b in bins], dtype=np.float32) * p_scaling
    select_bin_idx = \
        np.random.choice(np.arange(len(bins)), 
                         size=(num_batches,), 
                         p = bin_prob/np.sum(bin_prob))
    select_bins = [bins[i] for i in select_bin_idx]
    
    file_ids = [
        np.random.randint(0, train_name_list_bin[b]['size'],
                          size=(batch_size // 2 if adversarial else batch_size,)) for b in select_bins
    ]
    orgnized_name_list = [] # reduce(lambda x, y: x+y, name_list)
    for ids, b in zip(file_ids, select_bins):
        batch_files = [train_name_list_bin[b]['name_list'][i] for i in ids]
        batch_files = [x if isinstance(x, str) else x[0] for x in batch_files]
        if adversarial:
            batch_files = np.array(batch_files).reshape(-1, n_sample_per_device//2)
            batch_files = np.concatenate([batch_files, batch_files], axis=-1).reshape(-1).tolist()
        orgnized_name_list.extend([str(x) for x in batch_files])

    # orgnized_name_list = [x if isinstance(x, str) else x[0] for x in orgnized_name_list]
    return orgnized_name_list, select_bins