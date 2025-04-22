"""Transformer variants"""

import jax
import jax.numpy as jnp

from flax import linen as nn
from modules.transformer_blocks import PreNonLinear, PostNonLinear, Transition, OuterProduct
from .transformers import SelfResidualTransformer
from dataclasses import field
from common.config_load import Config
import ml_collections

class SeqBlock(nn.Module):
    """SeqBlock"""

    global_config: ml_collections.ConfigDict
    seq_act_dim: int
    pair_act_dim: int
    hidden_dim: int
    num_head: int
    intermediate_dim: int
    attn_scale: float = 0.
    ffn_scale: float = 0.
    dropout_rate: float = 0.
    norm_method: str = "rmsnorm"
    gating: bool = False
    sink_attention: bool = False
    pre_attention_operation_list: tuple = ("PairEmbedding", "LN",) # = ["PairEmbedding", "LN"] # ["PairEmbedding", "PreLN", "AttDropout"]
    post_attention_operation_list: tuple = ("Dropout", "LN") # = ["Dropout", "LN"] # ["Dropout", "PostLN"]
    post_ffn_operation_list: tuple = ("Dropout",) # = ["Dropout"] # ["Dropout", "PostLN", "ResidualLN"]
    init_method: str = "AF2"
    init_sigma: float = 0.02
    swish_beta: float = 1.

    def setup(self):

        self.transformer = SelfResidualTransformer(global_config=self.global_config,
                                                   q_act_dim=self.seq_act_dim,
                                                   pair_act_dim=self.pair_act_dim,
                                                   hidden_dim=self.hidden_dim,
                                                   num_head=self.num_head,
                                                   intermediate_dim=self.intermediate_dim,
                                                   attn_scale=self.attn_scale,
                                                   ffn_scale=self.ffn_scale,
                                                   dropout_rate=self.dropout_rate,
                                                   norm_method=self.norm_method,
                                                   gating=self.gating,
                                                   sink_attention=self.sink_attention,
                                                   pre_attention_operation_list=self.pre_attention_operation_list, # ["PairEmbedding", "PreLN", "AttDropout"],
                                                   post_attention_operation_list=self.post_attention_operation_list, # ["Dropout", "PostLN"],
                                                   post_ffn_operation_list=self.post_ffn_operation_list, # ["Dropout", "PostLN", "ResidualLN"],
                                                   init_method=self.init_method,
                                                   init_sigma=self.init_sigma, ###
                                                   swish_beta=self.swish_beta,)

    def __call__(self, act, accumulated_act, attention_masks, pair_act=0., pos_index=None, distance_cutoff=None):
        '''construct'''
        ### Shapes:
        ### act&accumulated_act:(B,Q,c); attention_masks:[(B,Q),(B,K),(B,Q,K)]; pair_act:(B,Q,K,cz); pos_index:(B,Q)

        seq_act = act
        accumulated_seq_act = accumulated_act

        seq_act, accumulated_seq_act =\
            self.transformer(seq_act, accumulated_seq_act, attention_masks, pair_act=pair_act, distance_cutoff=distance_cutoff)
        
        return seq_act, accumulated_seq_act


class PairBlock(nn.Module):

    global_config: ml_collections.ConfigDict
    seq_act_dim: int
    pair_act_dim: int
    outerproduct_dim: int
    intermediate_dim: int
    outerproduct_scale: float = 0.
    ffn_scale: float = 0.
    pre_outerproduct_operation_list: tuple = ("LN",) # = ["LN"] # ["PreLN"; "Dropout"]
    post_outerproduct_operation_list: tuple = ("LN",) # = ["LN"] # ["PostLN", "Dropout", "ResiDual"]
    post_ffn_operation_list: tuple = ("Dropout",) # = ["Dropout"] # ["PostLN", "Dropout", "ResiDual"]
    dropout_rate: float = 0.
    norm_method: str = "rmsnorm" # ["layernorm", "rmsnorm"]
    init_method: str = "AF2"
    init_sigma: float = 0.02
    swish_beta: float = 1.

    def setup(self):
        
        self.pre_process = PreNonLinear(global_config=self.global_config,
                                        q_data_dim=self.seq_act_dim,
                                        pair_act_dim=self.pair_act_dim,
                                        operation_list=self.pre_outerproduct_operation_list,
                                        dropout_rate=self.dropout_rate,
                                        norm_method=self.norm_method,
                                        self_attention=True,)
        

        outerproduct_ = nn.checkpoint(OuterProduct) if self.global_config.remat_flag else OuterProduct
        self.outerproduct = outerproduct_(global_config=self.global_config,
                                          input_dim=self.seq_act_dim,
                                          output_dim=self.pair_act_dim,
                                          outerproduct_dim=self.outerproduct_dim,
                                          init_sigma=0.02,)

        self.post_outerproduct = PostNonLinear(global_config=self.global_config,
                                               o_data_dim=self.pair_act_dim,
                                               operation_list=self.post_outerproduct_operation_list,
                                               dropout_rate=self.dropout_rate,
                                               norm_method=self.norm_method,
                                               accumulated_scale=self.outerproduct_scale,)
                                            
        ffn_ = nn.checkpoint(Transition) if self.global_config.remat_flag else Transition
        self.ffn = ffn_(global_config=self.global_config,
                        input_dim=self.pair_act_dim,
                        intermediate_dim=self.intermediate_dim, ### 8//3*pair_act_dim,
                        output_dim=self.pair_act_dim,
                        init_method=self.init_method,
                        init_sigma=self.init_sigma,
                        swish_beta=self.swish_beta,)
        
        self.post_ffn = PostNonLinear(global_config=self.global_config,
                                      o_data_dim=self.pair_act_dim,
                                      operation_list=self.post_ffn_operation_list,
                                      dropout_rate=self.dropout_rate,
                                      norm_method=self.norm_method, # "rmsnorm",
                                      accumulated_scale=self.ffn_scale,)

        # if RECOMPUTE_FLAG:
        #     self.outerproduct.recompute()
        #     self.ffn.recompute()

    def __call__(self, seq_act, pair_act, accumulated_pair_act, attention_masks):
        ### Shapes:
        ### seq_act:(B,Q,c);
        ### pair_act&accumulated_pair_act:(B,Q,K,c); attention_masks:[(B,Q),(B,K),(B,Q,K)].
        act_shape = pair_act.shape
        flatten_shape = (act_shape[0], -1, act_shape[-1])

        seq_mask = attention_masks[0]
        pair_mask = attention_masks[-1]

        residual_act = jnp.reshape(pair_act, flatten_shape) # (B,Q*K,c)
        accumulated_act = jnp.reshape(accumulated_pair_act, flatten_shape) # (B,Q*K,c)
        pair_mask = jnp.reshape(pair_mask, (act_shape[0], -1)) # (B,Q*K)

        q_act = seq_act
        k_act = q_act
        v_act = q_act

        results_tuple = self.pre_process(q_act, k_act, v_act, attention_masks)
        seq_act = results_tuple[0]

        outerproduct_output = self.outerproduct(seq_act, seq_mask)
        outerproduct_output = jnp.reshape(outerproduct_output, flatten_shape) # (B,Q*K,c)

        residual_act, accumulated_act = self.post_outerproduct(residual_act, outerproduct_output, pair_mask, accumulated_act)
  
        ffn_output = self.ffn(residual_act)
        residual_act, accumulated_act = self.post_ffn(residual_act, ffn_output, pair_mask, accumulated_act)

        residual_act = jnp.reshape(residual_act, act_shape) # (B,Q,K,c)
        accumulated_act = jnp.reshape(accumulated_act, act_shape) # (B,Q,K,c)

        return residual_act, accumulated_act

class FlashEvoformerStack(nn.Module):
    r"""Batchwise Evoformer without Triangle Operations"""

    global_config: ml_collections.ConfigDict
    seq_act_dim: int
    pair_act_dim: int
    outerproduct_dim: int
    hidden_dim: int
    num_head: int
    intermediate_dim: int
    dropout_rate: float = 0.
    norm_method: str = "rmsnorm" # ["layernorm", "rmsnorm"]
    gating: bool = True
    sink_attention: bool = False
    init_method: str = "AF2"
    init_sigma: float = 0.02
    swish_beta: float = 1.
    post_ffn_operation_list: tuple = ("Dropout",) # ["LN", "Dropout", "ResidualLN"] ### Include "ResiDual" at the last layer.

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        
        self.prepare_pair_act = PostNonLinear(global_config=self.global_config,
                                              o_data_dim=self.pair_act_dim,
                                              operation_list=("LN",),
                                              dropout_rate=0., # 参考AF2, 写死
                                              accumulated_scale=0.,
                                              execute_residual=False, ### 仅做LN，不执行skip connection update
                                              norm_method="layernorm",)
        
        self.prepare_seq_act = PostNonLinear(global_config=self.global_config,
                                             o_data_dim=self.seq_act_dim,
                                             operation_list=("LN",),
                                             dropout_rate=0., # 参考AF2, 写死
                                             accumulated_scale=0.,
                                             execute_residual=False, ### 仅做LN，不执行skip connection update
                                             norm_method="layernorm",)
        
        self.seq_block = SeqBlock(global_config=self.global_config,
                                  seq_act_dim=self.seq_act_dim,
                                  pair_act_dim=self.pair_act_dim,
                                  hidden_dim=self.hidden_dim,
                                  num_head=self.num_head,
                                  intermediate_dim=self.intermediate_dim,
                                  # attn_scale=attn_scale,
                                  # ffn_scale=ffn_scale,
                                  dropout_rate=self.dropout_rate,
                                  norm_method=self.norm_method,
                                  gating=self.gating,
                                  sink_attention=self.sink_attention,
                                  # pre_attention_operation_list=pre_attention_operation_list,
                                  # post_attention_operation_list=post_attention_operation_list,
                                  post_ffn_operation_list=self.post_ffn_operation_list,
                                  init_method=self.init_method,
                                  init_sigma=self.init_sigma,
                                  swish_beta=self.swish_beta,)

        self.pair_block = PairBlock(global_config=self.global_config,
                                    seq_act_dim=self.seq_act_dim,
                                    pair_act_dim=self.pair_act_dim,
                                    outerproduct_dim=self.outerproduct_dim,
                                    intermediate_dim=self.intermediate_dim,
                                    # outerproduct_scale=outerproduct_scale,
                                    # ffn_scale=ffn_scale,
                                    dropout_rate=self.dropout_rate,
                                    norm_method=self.norm_method, # ["layernorm", "rmsnorm"]
                                    # pre_outerproduct_operation_list=pre_outerproduct_operation_list,
                                    # post_outerproduct_operation_list=post_outerproduct_operation_list,
                                    post_ffn_operation_list=self.post_ffn_operation_list,
                                    init_method=self.init_method,
                                    init_sigma=self.init_sigma,
                                    swish_beta=self.swish_beta,)
        

    def __call__(self, seq_act, pair_act, accumulated_seq_act, accumulated_pair_act, attention_masks, pos_index=None, distance_cutoff=None):

        ### 0. Prepare Cross-talking features:
        accumulated_seq_act = 0.* seq_act
        accumulated_pair_act = 0.* pair_act

        pair_act_shape = pair_act.shape
        flatten_shape = (pair_act_shape[0], -1, pair_act_shape[-1])

        pair_act_flatten = jnp.reshape(pair_act, flatten_shape) # (B,Q*K,c) # [Q,K,c]
        pair_act_residual = jnp.reshape(accumulated_pair_act, flatten_shape) # (B,Q*K,c) # [Q,K,c]
        pair_mask_flatten = jnp.reshape(attention_masks[-1], flatten_shape[:-1]) # (B,Q*K) # [Q,K]
        seq_mask = attention_masks[0] # (B,Q) # [Q,]
        
        # (B,Q*K,c): # [Q,K,c]
        pair_act_prepared, _pair_act_residual = self.prepare_pair_act(pair_act_flatten, 0.*pair_act_flatten, pair_mask_flatten, pair_act_residual)
        # pair_act_combined = self.postln_scale*pair_act_flatten + pair_act_residual # 模仿ResidualTransformer的final layer
        pair_act_prepared = jnp.reshape(pair_act_prepared, pair_act_shape) # (B,Q,K,c) # [Q,K,c]

        # (B,Q,c):
        seq_act_prepared, _seq_act_residual = self.prepare_seq_act(seq_act, 0.*seq_act, seq_mask, accumulated_seq_act) # [Q,c]
        # seq_act_combined = self.postln_scale*seq_act_postln + seq_act_residual # 模仿ResidualTransformer的final layer

        ### 1. Update Sequence Rep.:
        seq_act, accumulated_seq_act = self.seq_block(seq_act_prepared, accumulated_seq_act, attention_masks, pair_act=pair_act_prepared, pos_index=pos_index, distance_cutoff=distance_cutoff)

        ### 2. Update Pair Rep.:
        pair_act, accumulated_pair_act = self.pair_block(seq_act_prepared, pair_act_prepared, accumulated_pair_act, attention_masks)

        return seq_act, pair_act, accumulated_seq_act, accumulated_pair_act