import jax
import jax.numpy as jnp 

def softmax_cross_entropy(logits, label, seq_mask):
    # logits: (Nres, Nbins)
    # label: (Nres, Nbins) (one hot)
    # seq_mask: (Nres) 
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(log_probs * label, axis=-1)
    
    return jnp.sum(loss * seq_mask) / (jnp.sum(seq_mask) + 1e-6)

def binary_cross_entropy(logits, label, seq_mask):
    # logits: (Nres)
    # label: (Nres)
    # seq_mask: (Nres) 
    
    loss = -label * jax.nn.log_sigmoid(logits) \
            - (1 - label) * jax.nn.log_sigmoid(-logits)
    
    return jnp.sum(loss * seq_mask) / (jnp.sum(seq_mask) + 1e-6)