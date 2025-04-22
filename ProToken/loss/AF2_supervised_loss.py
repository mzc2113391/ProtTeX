import jax 
import jax.numpy as jnp
from loss.utils import _l2_normalize, square_euclidean_distance

def supervised_single_mse_loss(single, true_single, seq_mask):
    # single, (N, C)
    # seq_mask, (N)

    return jnp.sum((single - true_single)**2 * seq_mask[:, None]) / \
           (jnp.sum(seq_mask) * true_single.shape[-1] + 1e-6)

def supervised_pair_mse_loss(pair, true_pair, seq_mask):
    # pair, (N, N, C)
    # seq_mask, (N)

    pair_mask = seq_mask[None, :] * seq_mask[:, None]
    return jnp.sum((pair - true_pair)**2 * pair_mask[:, :, None]) / \
           (jnp.sum(pair_mask) * true_pair.shape[-1] + 1e-6)
           
def supervised_single_cos_sim_loss(single, true_single, seq_mask, epsilon=1e-5):
    single = _l2_normalize(single, axis=-1, epsilon=epsilon)
    true_single = _l2_normalize(true_single, axis=-1, epsilon=epsilon)
    
    dist = square_euclidean_distance(single, true_single, axis=-1, normalized=True)
    
    return jnp.sum(dist * seq_mask) / \
              (jnp.sum(seq_mask) + 1e-6)
              
def supervised_pair_cos_sim_loss(pair, true_pair, seq_mask, epsilon=1e-5):
    pair = _l2_normalize(pair, axis=-1, epsilon=epsilon)
    true_pair = _l2_normalize(true_pair, axis=-1, epsilon=epsilon)
    
    pair_mask = seq_mask[None, :] * seq_mask[:, None]
    dist = square_euclidean_distance(pair, true_pair, axis=-1, normalized=True)
    
    return jnp.sum(dist * pair_mask) / \
              (jnp.sum(pair_mask) + 1e-6)