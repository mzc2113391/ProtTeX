"""encoder model"""
import jax
import numpy as np
import jax.numpy as jnp

from flax import linen as nn
from common.config_load import Config

from common.utils import dgram_from_positions
from modules.basic import RelativePositionEmbedding
from modules.head import InverseFoldingHead

from model.flash_evoformer import FlashEvoformerStack
from model.transformers import SelfResidualTransformer
import ml_collections

class BottleneckModel(nn.Module):
    
    global_config: ml_collections.ConfigDict
    cfg: Config
    num_layers: int = 6
    inverse_folding: bool = False
    
    def setup(self):
        
        self.postln_scale = self.cfg.common.postln_scale # 1.0
        self.single_channel = self.cfg.common.single_channel # 384
        self.pair_channel = self.cfg.common.pair_channel # 192
        
        self.single_update_transformer_stack_num = self.num_layers
        self.transformer_cfg = self.cfg.transformer
        
        # default dtype setting

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.pre_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype,
                                           param_dtype=jnp.float32)
        
        # rel_pos embedding setting
        self.rel_pos = RelativePositionEmbedding(global_config=self.global_config,
                                                 exact_distance=self.cfg.rel_pos.exact_distance, # 16
                                                 num_buckets=self.cfg.rel_pos.num_buckets, # 32
                                                 max_distance=self.cfg.rel_pos.max_distance,) # 64
        
        self.pair_activations = nn.Dense(features = self.pair_channel, # 64 -> 192
                                         kernel_init=nn.initializers.lecun_normal(),
                                         dtype=self._dtype, param_dtype=jnp.float32)
        
        # single residual update stack
        single_residual_update_stack = []
        post_ffn_operation_list = ('Dropout',)
        for i_ in range(self.single_update_transformer_stack_num):
            if i_ == self.single_update_transformer_stack_num -1:
                post_ffn_operation_list = ("ResidualLN", "Dropout")
            rt_block = SelfResidualTransformer(global_config=self.global_config,
                                               q_act_dim=self.single_channel,
                                               pair_act_dim=self.pair_channel,
                                               hidden_dim=self.transformer_cfg.hidden_dim,
                                               num_head=self.transformer_cfg.num_head,
                                               intermediate_dim=self.transformer_cfg.intermediate_dim,
                                               dropout_rate=self.transformer_cfg.dropout_rate,
                                               gating=self.transformer_cfg.gating,
                                               sink_attention=self.transformer_cfg.sink_attention,
                                               norm_method=self.transformer_cfg.norm_method,
                                               post_ffn_operation_list=post_ffn_operation_list,
                                               init_method=self.transformer_cfg.init_method,
                                               init_sigma=self.transformer_cfg.init_sigma,
                                               swish_beta=self.transformer_cfg.swish_beta,
                                               )
            single_residual_update_stack.append(rt_block)
        self.single_residual_update_stack = single_residual_update_stack
        
        # self.post_layer_norm = nn.LayerNorm(epsilon=NORM_SMALL, dtype=jnp.float32)
        if self.inverse_folding:
            self.module_inverse_folding = InverseFoldingHead(global_config=self.global_config,
                                                             cfg=self.cfg.inverse_folding)
        
    def __call__(self, single_act, seq_mask, residue_index):

        _, rel_pos = self.rel_pos(residue_index, residue_index) #[num_batch, Nres, Nres, 64]
        pair_act = self.pair_activations(rel_pos).astype(self._dtype) # [num_batch, Nres, Nres, 192]

        single_act = self.pre_layer_norm(single_act) # @ZhangJ. 入参single_act是L2_normed?

        # @ZhangJ. commented out:
        # vq_pair_act_left = self.vq_activations_left(single_act)
        # vq_pair_act_right = self.vq_activations_right(single_act)
        # vq_pair_act = jnp.expand_dims(vq_pair_act_left, axis=-2) + jnp.expand_dims(vq_pair_act_right, axis=-3) # [num_batch, Nres, Nres, 192]
        # pair_act += vq_pair_act
        
        mask_2d = jnp.expand_dims(seq_mask, -1) * jnp.expand_dims(seq_mask, -2)
        attention_masks = (seq_mask, seq_mask, mask_2d)
        acc_single_act = single_act
        for i in range(self.single_update_transformer_stack_num):
            single_act, acc_single_act = \
                self.single_residual_update_stack[i](act=single_act, # [num_batch, Nres, 384]
                                                     accumulated_act=acc_single_act, 
                                                     attention_masks=attention_masks,
                                                     pair_act=pair_act) 
        # Added by @ZhangJ.:
        single_act = self.postln_scale * single_act + acc_single_act
        
        # single_act = self.post_layer_norm(single_act) # no need
        ret = single_act 
        if self.inverse_folding:
            inverse_folding_logits = self.module_inverse_folding(single_act)
            ret = (ret, inverse_folding_logits)
        
        return ret
