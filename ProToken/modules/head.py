import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from modules.structure import FoldIteration
from common.geometry import quat_to_rot

from common.config_load import Config
# from config.global_setup import EnvironConfig
from .basic import ActFuncWrapper
import ml_collections


class DistogramHead(nn.Module):
    """Head to predict a distogram, support batch.

    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """
    global_config: ml_collections.ConfigDict
    cfg: Config
    pair_dim: int = 192
    intermediate_dim: int = 512

    def setup(self):
        
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.first_break = self.cfg.first_break
        self.last_break = self.cfg.last_break
        self.num_bins = self.cfg.num_bins
        self.act_fn = nn.silu

        self.half_logits = nn.Dense(features=self.intermediate_dim, # 192 -> 512
                                    kernel_init = nn.initializers.lecun_normal(), dtype=self._dtype, param_dtype=jnp.float32)
        self.output_logits = nn.Dense(self.num_bins, kernel_init=nn.initializers.zeros_init(), dtype=self._dtype, param_dtype=jnp.float32)# 512 -> 36
        
    def __call__(self, pair):
        """Builds DistogramHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'pair': pair representation, shape [N_res, N_res, c_z].

        Returns:
          Dictionary containing:
            * logits: logits for distogram, shape [N_res, N_res, N_bins].
            * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
        """
        # half_logits = self.half_logits(pair)
        half_embedding = self.act_fn(self.half_logits(pair))
        sym_embedding = half_embedding + jnp.swapaxes(half_embedding, -2, -3)
        logits = self.output_logits(sym_embedding)

        # logits = half_logits + jnp.swapaxes(half_logits, -2, -3)
        breaks = jnp.linspace(self.first_break, self.last_break, self.num_bins - 1, dtype=self._dtype)

        return logits, breaks
      
class PredictedLDDTHead(nn.Module):
  """Head to predict the per-residue LDDT to be used as a confidence measure."""
  
  global_config: ml_collections.ConfigDict
  cfg: Config
  
  def setup(self):
    
    self.bf16_flag = self.global_config.bf16_flag
    self.dropout_flag = self.global_config.use_dropout
    self.norm_small = self.global_config.norm_small
    self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
    
    self.init_fold_iteration = FoldIteration(self.global_config,
                                             self.cfg.fold_iteration, 
                                             self.cfg.fold_iteration.dropout, 
                                             stop_grad=True)
    
    self.input_layer_norm = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
    self.act_0 = nn.Dense(features=self.cfg.num_channel,
                          kernel_init = nn.initializers.lecun_normal(), dtype=self._dtype, param_dtype=jnp.float32)
    self.act_1 = nn.Dense(features=self.cfg.num_channel,
                          kernel_init = nn.initializers.lecun_normal(), dtype=self._dtype, param_dtype=jnp.float32)
    self.logits = nn.Dense(features=self.cfg.num_bins,
                           kernel_init = nn.initializers.zeros_init(), dtype=self._dtype, param_dtype=jnp.float32)
    self.relu = nn.relu
  
  def __call__(self, act, static_feat_2d, sequence_mask, affine, aatype):
    affine = jax.lax.stop_gradient(affine)
    quaternion, translation = affine[..., :4], (affine[..., 4], affine[..., 5], affine[..., 6])
    rotation = quat_to_rot(quaternion)
    res = self.init_fold_iteration(act=act, 
                                   static_feat_2d=static_feat_2d, 
                                   sequence_mask=jnp.expand_dims(sequence_mask, -1), 
                                   quaternion=quaternion, 
                                   rotation=rotation, 
                                   translation=translation, 
                                   initial_act=act, #### initial act is only used to predict side chain
                                   aatype=aatype, 
                                   contextual_mask=jnp.ones_like(sequence_mask)) #### fix affine tensor
    act = res[0]
    act = act.astype(self._dtype)
    act = self.input_layer_norm(act)
    act = self.act_0(act)
    act = self.relu(act)
    act = self.act_1(act)
    act = self.relu(act)
    logits = self.logits(act)
    return logits
  
def compute_plddt(logits):
  """Computes per-residue pLDDT from logits.
  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.
  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  _np, _softmax = jnp, jax.nn.softmax

  
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = _np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = _softmax(logits, axis=-1)
  predicted_lddt_ca = (probs * bin_centers[None, :]).sum(-1)
  return predicted_lddt_ca * 100

class InverseFoldingHead(nn.Module):
  
  global_config: ml_collections.ConfigDict
  cfg: Config
  
  def setup(self):

    self.bf16_flag = self.global_config.bf16_flag
    self.dropout_flag = self.global_config.use_dropout
    self.norm_small = self.global_config.norm_small
    self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
    
    self.input_layer_norm = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
    
    self.act_0 = nn.Dense(features=self.cfg.num_channel, kernel_init = nn.initializers.lecun_normal(), 
                          dtype=self._dtype, param_dtype=jnp.float32)
    self.act_1 = nn.Dense(features=self.cfg.num_channel, kernel_init = nn.initializers.lecun_normal(), 
                          dtype=self._dtype, param_dtype=jnp.float32)
    self.logits = nn.Dense(features=self.cfg.num_bins, kernel_init = nn.initializers.zeros_init(),
                           dtype=self._dtype, param_dtype=jnp.float32)
    
    self.relu = nn.relu
  
  def __call__(self, act):
    act = self.input_layer_norm(act)
    act = self.act_0(act)
    act = self.relu(act)
    act = self.act_1(act)
    act = self.relu(act)
    logits = self.logits(act)
    return logits