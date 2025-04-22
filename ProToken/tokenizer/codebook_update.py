from common.config_load import Config
import jax 
import jax.numpy as jnp
from flax import traverse_util
import numpy as np
from functools import partial
from loss.utils import square_euclidean_distance

def reinitialize_dead_code_(code_count, codebook, rng_key):        
    code_count_sorted_idx = jnp.argsort(code_count)
    code_count_sorted_idx_reverse = code_count_sorted_idx[::-1]
    code_count_sorted = code_count[code_count_sorted_idx]
    dead_code_mask = code_count_sorted < 2
    
    codebook_update = jnp.where(
        jnp.expand_dims(dead_code_mask, -1),  
        codebook[code_count_sorted_idx_reverse],
        codebook[code_count_sorted_idx]
    )
    
    codebook_update = codebook_update + \
            jax.random.normal(rng_key, shape=codebook.shape) * 1e-2 * jnp.expand_dims(dead_code_mask, -1) 
    
    return codebook_update, code_count_sorted

def reinitialize_dead_code_split(code_count, 
                                 codebook, 
                                 input_embeddings,
                                 seq_mask,
                                 rng_key, 
                                 noise_rng_key, 
                                 noise_ratio=0.01,
                                 decay=0.99,
                                 l2norm=True):        
    code_sample_weight = code_count / (1e-6 + jnp.sum(code_count))
    ncode = codebook.shape[0]
    code_index_resampled = jax.random.choice(
        rng_key, a=jnp.arange(ncode), replace=True, 
        p=code_sample_weight, shape=(ncode,),
    )
    codebook_resampled = codebook[code_index_resampled]
    
    dead_code_mask = code_count < 2
    
    codebook_update = jnp.where(
        jnp.expand_dims(dead_code_mask, -1),  
        codebook_resampled, codebook
    )
    
    codebook_update = codebook_update + \
            jax.random.normal(noise_rng_key, shape=codebook.shape) * noise_ratio * jnp.expand_dims(dead_code_mask, -1) 
    
    return codebook_update, code_count

def reinitialize_dead_code_random(code_count, 
                                  codebook, 
                                  input_embeddings, # (B, Nres, dim_code)
                                  seq_mask, # (B, Nres, )
                                  rng_key, 
                                  noise_rng_key,
                                  noise_ratio=1e-4,
                                  decay=0.99,
                                  l2norm=True):            
    dead_code_mask = code_count < 1e-6
    ncode = codebook.shape[0]
    
    input_embeddings = input_embeddings.reshape(-1, input_embeddings.shape[-1])
    seq_mask = seq_mask.reshape(-1).astype(jnp.float32)
    n_input_embeddings = input_embeddings.shape[0]
    index_resampled = jax.random.choice(
        rng_key, a=jnp.arange(n_input_embeddings), replace=True, shape=(ncode,),
        p=seq_mask / jnp.sum(seq_mask)
    )
    codebook_resampled = input_embeddings[index_resampled]
    
    codebook_update = jnp.where(
        jnp.expand_dims(dead_code_mask, -1),  
        codebook_resampled, codebook
    )
    
    codebook_update = codebook_update + \
            jax.random.normal(noise_rng_key, shape=codebook.shape) * noise_ratio * jnp.expand_dims(dead_code_mask, -1) 
    
    return codebook_update, code_count

def codebook_update_cvq_random(embed_prob, 
                               codebook, 
                               input_embeddings, # (B, Nres, dim_code)
                               seq_mask, # (B, Nres, )
                               rng_key, 
                               noise_rng_key,
                               noise_ratio=1e-4,
                               decay=0.99,
                               l2norm=True):  
    ncode = codebook.shape[0]
    decay_ = jnp.exp(-(embed_prob * ncode * 100) / (1 - decay) -1e-3)

    input_embeddings = input_embeddings.reshape(-1, input_embeddings.shape[-1])
    seq_mask = seq_mask.reshape(-1).astype(jnp.float32)
    n_input_embeddings = input_embeddings.shape[0]
    index_resampled = jax.random.choice(
        rng_key, a=jnp.arange(n_input_embeddings), replace=True, shape=(ncode,),
        p=seq_mask / jnp.sum(seq_mask)
    )
    codebook_resampled = input_embeddings[index_resampled]
    codebook_update = codebook * (1 - decay_[..., None]) + codebook_resampled * decay_[..., None]
    
    return codebook_update, embed_prob

def codebook_update_cvq_closest(embed_prob, 
                                codebook, 
                                input_embeddings, # (B, Nres, dim_code)
                                seq_mask, # (B, Nres, )
                                rng_key, 
                                noise_rng_key,
                                noise_ratio=1e-4,
                                decay=0.99,
                                l2norm=True):  
    ncode = codebook.shape[0]
    decay_ = jnp.exp(-(embed_prob * ncode * 10) / (1 - decay) -1e-3)

    input_embeddings = input_embeddings.reshape(-1, input_embeddings.shape[-1])
    seq_mask = seq_mask.reshape(-1).astype(jnp.float32)
    distances = square_euclidean_distance(codebook[:, None, ...], input_embeddings[None, ...], normalized=l2norm) # (Ncode, Nemb)
    distances = distances + (1.0  - seq_mask[None, ...]) * 1e5
    
    codebook_resampled = input_embeddings[jnp.argmin(distances, axis=-1)]
    codebook_update = codebook * (1 - decay_[..., None]) + codebook_resampled * decay_[..., None]
    
    return codebook_update, embed_prob

class VQLURReinitialization:
    def __init__(self, 
                 vq_cfg, ## ['split', 'random', 'cvq']
                 decay=0.99,
                 re_initialization_step=20,
                 noise_ratio=0.01, 
                 pmap=True,
                 n_local_devices=8,
                 n_processes=1,
                 rank=0,
                 ):
        self.mode = vq_cfg.lur_mode 
        assert self.mode in ['split', 'random', 'cvq'], 'unsupported mode {}'.format(vq_cfg.lur_mode )
        self.re_initialization_step = re_initialization_step 
        self.noise_ratio = noise_ratio 
        
        self.decay = decay
        self.code_count_ema = 0.0 # 1.0 / vq_cfg.num_code
        
        self.pmap = pmap 
        self.fn = {'split': reinitialize_dead_code_split, 
                   'random': reinitialize_dead_code_random,
                   'cvq': codebook_update_cvq_closest}[self.mode]
        self.fn = partial(self.fn, noise_ratio=self.noise_ratio, decay=self.decay, l2norm=vq_cfg.l2_norm)
        self.synchronize_fn = lambda x: x
        if self.pmap:
            self.fn = jax.pmap(jax.jit(self.fn))
            pmap_sum = \
                jax.pmap(jax.jit(lambda x:jax.lax.psum(x, axis_name="i")), 
                         axis_name="i")
            self.synchronize_fn = \
                lambda x: pmap_sum(x * \
                                   (jnp.arange(n_local_devices) == np.random.randint(0, n_local_devices))[..., None, None].astype(x.dtype) * \
                                   jnp.array(rank == 0, dtype=x.dtype))

    def __call__(self, 
                 code_count, 
                 codebook, 
                 input_embeddings, # (B, Nres, dim_code)
                 seq_mask, # (B, Nres, )
                 rng_key, 
                 noise_rng_key,
                 update=True):
        
        self.code_count_ema = \
            self.decay * self.code_count_ema + code_count * (1 - self.decay)
            
        if update:
            codebook_update, code_count_ema = self.fn(self.code_count_ema, 
                                                      codebook, 
                                                      input_embeddings, # (B, Nres, dim_code)
                                                      seq_mask, # (B, Nres, )
                                                      rng_key, 
                                                      noise_rng_key)
            self.code_count_ema = code_count_ema
            
            codebook_update = self.synchronize_fn(codebook_update)
        else:
            codebook_update = codebook

        return codebook_update, self.code_count_ema
