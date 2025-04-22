"""Basic modules"""
import jax
import jax.numpy as jnp

from typing import Callable
from flax import linen as nn
from flax.training.common_utils import onehot

from common.config_load import Config
import ml_collections

class RelativePositionEmbedding(nn.Module):

    global_config: ml_collections.ConfigDict
    exact_distance: float
    num_buckets: int
    max_distance: float

    def setup(self):

        # basic precision setting
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

    def _relative_position_bucket(self, x, alpha=32,
                                  beta=64, gamma=128,):
        
        (x, alpha, beta, gamma) = jax.tree_util.tree_map(lambda t: jnp.asarray(t, dtype=self._dtype), (x, alpha, beta, gamma))

        scale = (beta - alpha) / jnp.log(gamma / alpha)
        x_abs = jnp.abs(x)
        gx = jnp.log((x_abs + 1e-5) / alpha) * scale + alpha ## Liyh: 1e-5?
        gx = jnp.minimum(beta, gx)
        gx = jnp.sign(x) * gx

        cond = jnp.greater(x_abs, alpha)
        ret = jax.lax.select(cond, gx, x)
        ret = jnp.clip(ret, -beta, beta)

        #### ret += beta
        ### Asymptotic symmetry
        ret += beta - 1
        cond = jnp.greater(ret, -0.5)
        ret = jax.lax.select(cond, ret, jnp.ones_like(ret) * (2 * beta - 1.))

        return jnp.asarray(ret, dtype=jnp.int32)

    def __call__(self, q_idx, k_idx):
        """Compute binned relative position encoding"""
        ### q_idx, k_idx: [..., Nres,]
        # [..., Nres, 1]
        context_position = jnp.expand_dims(q_idx, -1) ## Liyh: support batch dim? zhenyu: yes   
        # [..., Nres]
        memory_position = jnp.expand_dims(k_idx, -2)
        # [..., Nres, Nres]
        relative_position = memory_position - context_position
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        relpos_bucket = self._relative_position_bucket(
            relative_position,
            alpha=self.exact_distance,
            beta=self.num_buckets,
            gamma=self.max_distance,
        )
        relpos_onehot = onehot(relpos_bucket, 2*self.num_buckets)
        # [..., Nres, Nres], [..., Nres, Nres, 2*num_buckets]
        return relpos_bucket, relpos_onehot # int32, float32
    
class Softmax1(nn.Module):

    axis: int = -1

    def __call__(self, logits):
        zeros = jnp.zeros_like(logits)
        zeros, _ = jnp.split(zeros, (1,), axis=self.axis)
        logits_1 = jnp.concatenate([zeros, logits], self.axis)
        softmax = nn.softmax(logits_1, axis=self.axis)
        _, softmax1 = jnp.split(softmax, (1,), axis=self.axis)
        return softmax1
    
class ActFuncWrapper(nn.Module):
    """Wrapper for activation function"""

    act_fn: Callable

    @nn.compact
    def __call__(self, tensor):
        tensor_fp32 = jnp.asarray(tensor, jnp.float32)
        act = self.act_fn(tensor_fp32)
        act = jnp.asarray(act, tensor.dtype)
        return act

class Swish_beta(nn.Module):
    """Initialize Sodtmax"""
    beta: float = 1.0

    @nn.compact
    def __call__(self, x):
        y = x * nn.sigmoid(self.beta * x)
        return y

class TransMatMul(nn.Module):
    """Matrix multiplication"""
    transpose_b: bool = True

    @nn.compact
    def __call__(self, tensor, weights):
        ### tensor.ndim>=1; weights.ndim==2

        t_shape = tensor.shape
        tensor = self._reshape(tensor)
        output = jnp.matmul(tensor, weights.T) if self.transpose_b else jnp.matmul(tensor, weights)
        y = jnp.reshape(output, t_shape[:-1]+(-1,)) 
        return y
    
    def _reshape(self, tensor):
        t_shape = tensor.shape
        tensor = jnp.reshape(tensor, (-1, t_shape[-1]))
        return tensor

class RotaryEmbedding(nn.Module):

    dim: int
    custom_idx: bool = False
    t_max: int = 2048
    seq_dim: int = -2

    def setup(self):

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        self.inv_freq = jnp.asarray(inv_freq.repeat(2, 0), jnp.float32)
        self.t = jnp.arange(self.t_max, dtype=jnp.float32).reshape((1,-1))
    
    def __call__(self, q, k, pos_idx=0):
        # pos_idx: (...,len);

        if self.custom_idx:
            t = pos_idx
        else:
            t = self.t
        ###
        seq_len = q.shape[self.seq_dim]
        emb = t[:, :seq_len, None] * jnp.reshape(self.inv_freq, (1,1,-1)) # (B,Len,1)*(1,1,c)
        ###

        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return (self.apply_rotary_pos_emb(q, cos, sin),
                self.apply_rotary_pos_emb(k, cos, sin))
    
    def rotate_half(self, x):
        dim_x = x.shape
        x = jnp.reshape(x, dim_x[:-1] + (dim_x[-1]//2, 2))
        x1 = x[..., 0]
        x2 = x[..., 1]
        # O.Print()(x1.shape)
        return jnp.reshape(jnp.concatenate((-x2, x1), axis=-1), dim_x)
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        # x: (B,Q,c)
        return (x * cos) + (self.rotate_half(x) * sin)

def masked_layer_norm(layernorm, act, mask=None):
    """ Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.
    cf: DEBERTA@PyTorch https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/ops.py

    Args:
        layernorm: LayerNorm module or function
        act: [...,Cm]
        mask: [...]; The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`
        
    Example:

    """
    act_shape = act.shape
    if mask is not None:
        act = act * jnp.expand_dims(mask, -1)
    act = jnp.reshape(act, (-1, act_shape[-1])) # (B,c)
    # (...,c):
    act = layernorm(act)

    ''' # 避免value-dependent if:
    if mask is not None:
        mask = jnp.reshape(mask, (-1,1)) # (B,1) or (1,1)
        act = act * mask
    '''
    # (...,c):
    act = jnp.reshape(act, act_shape)
    
    return act

# @ZhangJ. 注意, epsilon可以设置成1e-8 (防止FP32的下溢出)；或者在此处临时使用FP64执行运算
def _l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / jnp.sqrt(
        jnp.maximum(jnp.sum(x**2, axis=axis, keepdims=True), epsilon))
    
def safe_l2_normalize(x, axis=-1, epsilon=1e-12):
    _dtype = x.dtype
    x = x.astype(jnp.float32)
    x = _l2_normalize(x, axis=axis, epsilon=epsilon)   
    return x.astype(_dtype)