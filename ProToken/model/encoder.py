"""encoder model"""
import jax
import numpy as np
import jax.numpy as jnp

from flax import linen as nn
from flax.training.common_utils import onehot
from common.config_load import Config

from common.utils import dgram_from_positions
from modules.basic import RelativePositionEmbedding

from model.flash_evoformer import FlashEvoformerStack
from model.transformers import SelfResidualTransformer
from modules.structure import StructureModule
from modules.templates import FlashSingleTemplateEmbedding
from modules.head import InverseFoldingHead
from modules.basic import ActFuncWrapper

import ml_collections


class Feature_Initializer(nn.Module):

    global_config: ml_collections.ConfigDict
    cfg: Config

    def setup(self):
        
        # basic precision setting
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.single_channel = self.cfg.common.single_channel # 256
        self.pair_channel = self.cfg.common.pair_channel # 128

        self.num_bins = self.cfg.distogram.num_bins # 36
        self.min_bin = self.cfg.distogram.first_break # 2.5
        self.max_bin = self.cfg.distogram.last_break # 20.0
        
        self.template_enabled = self.cfg.template.enabled # True

        self.prev_pos_linear = nn.Dense(features=self.pair_channel, # 36 -> 128
                                        kernel_init=nn.initializers.lecun_normal(),
                                        dtype=self._dtype, param_dtype=jnp.float32) # _dtype in, param->_dtype, _dtype out
        
        self.rel_pos = RelativePositionEmbedding(global_config=self.global_config,
                                                 exact_distance=self.cfg.rel_pos.exact_distance, # 16
                                                 num_buckets=self.cfg.rel_pos.num_buckets, # 32
                                                 max_distance=self.cfg.rel_pos.max_distance,) # 64

        self.pair_activations = nn.Dense(features=self.pair_channel, # rel_pos_feat_dim -> pair_channel: 64 -> 128
                                         kernel_init=nn.initializers.lecun_normal(), 
                                         dtype=self._dtype, param_dtype=jnp.float32)
        
        self.template_embedding = FlashSingleTemplateEmbedding(global_config=self.global_config,
                                                               num_channels=self.cfg.template.num_channel, # 128
                                                               num_block=self.cfg.template.num_block, # 1
                                                               init_sigma=self.cfg.template.init_sigma, # 0.02
                                                               init_method=self.cfg.template.init_method, # "AF2"
                                                               dropout_rate=self.cfg.template.dropout_rate, # 0.1
                                                               norm_method=self.cfg.template.norm_method,) # "rmsnorm"

        self.template_single_embedding = nn.Dense(self.single_channel, kernel_init=nn.initializers.lecun_normal(),
                                                  dtype=self._dtype, param_dtype=jnp.float32)  # template_feat_dim -> single_channe: 9 -> 256
        self.template_projection = nn.Dense(self.single_channel, kernel_init=nn.initializers.lecun_normal(),
                                            dtype=self._dtype, param_dtype=jnp.float32) # single_channel -> single_channel: 256 -> 256

    def __call__(self, seq_mask, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                 torsion_angles_sin_cos, torsion_angles_mask):
        
        """inputs of __call__ should be cast to self._dtype before put in the model"""
        # seq_mask: self._dtype, aatype: jnp.int32, residue_index: jnp.int32/self._dtype.
        # template_all_atom_masks: jnp.int32, template_all_atom_positions: jnp.float32, template_pseudo_beta: jnp.float32
        # torsion_angles_sin_cos: self._dtype, torsion_angles_mask: self._dtype
        # decoy_affine_tensor: jnp.float32(?)
        
        ### 将以下模块抽象成为feature_initializer:
        # '''
        template_pseudo_beta_mask = seq_mask # (self._dtype)
        mask_2d = jnp.expand_dims(seq_mask, -1) * jnp.expand_dims(seq_mask, -2) # [Nres, 1] * [1, Nres] -> [Nres, Nres] (self._dtype)
        num_res = residue_index.shape[0] # [Nres,]
        template_features = jnp.concatenate([jnp.reshape(torsion_angles_sin_cos, [num_res, 6]),torsion_angles_mask], 
                                            axis=-1, dtype=self._dtype) # [Nres, 9] (self._dtype)
        template_activations = self.template_single_embedding(template_features) # [Nres, 256] (self._dtype)
        template_activations = nn.relu(template_activations)
        template_activations = self.template_projection(template_activations) # [Nres, 256] (self._dtype)

        single_activations_init = template_activations # [Nres, 256] (self._dtype)

        _, rel_pos = self.rel_pos(residue_index, residue_index) # _: [Nres, Nres], rel_pos: [Nres, Nres, 64] (int32)
        rel_pos = jnp.asarray(rel_pos, self._dtype) # [Nres, Nres, 64] (self._dtype)
        
        pair_activations = self.pair_activations(rel_pos) # [Nres, Nres, 128] (self._dtype)

        pseudo_beta_dgram = dgram_from_positions(template_pseudo_beta, self.num_bins, self.min_bin, self.max_bin, self._dtype) # [Nres, Nres, 36] (self._dtype)
        
        pair_activations += self.prev_pos_linear(pseudo_beta_dgram) * jnp.expand_dims(mask_2d, -1) # [Nres, Nres, 128] (self._dtype)
        
        if self.template_enabled:
            template_pair_representation = self.template_embedding(pair_activations,
                                                                   template_all_atom_positions,
                                                                   template_all_atom_masks,
                                                                   template_pseudo_beta_mask,
                                                                   mask_2d,) # [Nres, Nres, 128] (self._dtype)
            pair_activations += template_pair_representation
        
            
        pair_activations_init = pair_activations

        return single_activations_init, pair_activations_init

class VQ_Encoder(nn.Module):

    global_config: ml_collections.ConfigDict
    cfg: Config
    # inverse_folding: bool = True

    def setup(self):
        
        # basic precision setting
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.seq_len = self.cfg.seq_len # 256
        self.esm_cfg = self.cfg.extended_structure_module # 8 layers

        self.postln_scale = self.cfg.common.postln_scale # 1.0
        self.single_channel = self.cfg.common.single_channel # 256
        self.pair_channel = self.cfg.common.pair_channel # 128
        
        #### distance cutoff 
        self.distance_cutoff_type = self.cfg.common.distance_cutoff_type 
        self.cutoff = self.cfg.common.cutoff
        self.distance_cutoff_fn = lambda x: 1.0 ### default: no cutoff 
        if self.distance_cutoff_type == "cosine":
            self.distance_cutoff_fn = lambda x: 0.5 * (1 + jnp.cos(x / self.cutoff * jnp.pi)) * (x < self.cutoff).astype(x.dtype)

        self.pair_update_evoformer_stack_num = self.cfg.pair_update_evoformer_stack_num # 4 for use, 1 for test
        self.single_update_transformer_stack_num = self.cfg.single_update_transformer_stack_num # 12 for use, 2 for test
        self.co_update_evoformer_stack_num = self.cfg.co_update_evoformer_stack_num # 8 for use, 1 for test
        
        self.evoformer_cfg = self.cfg.evoformer
        self.transformer_cfg = self.cfg.transformer

        self.esm_post_ln = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32)) # single_channel: 256
        
        self.feat_init = Feature_Initializer(global_config=self.global_config,
                                             cfg=self.cfg)
            
        pair_evoformer_update_stack = []
        post_ffn_operation_list = ('Dropout',)
        for i_ in range(self.pair_update_evoformer_stack_num):
            if i_ == self.pair_update_evoformer_stack_num - 1:
                post_ffn_operation_list = ('Dropout', 'LN')
            msa_block = FlashEvoformerStack(global_config=self.global_config,
                                            seq_act_dim=self.single_channel, # 256
                                            pair_act_dim=self.pair_channel, # 128
                                            outerproduct_dim=self.evoformer_cfg.outerproduct_dim, # 32
                                            hidden_dim=self.evoformer_cfg.hidden_dim, # 256 Zhenyu: Double Check here.
                                            num_head=self.evoformer_cfg.num_head, # 8
                                            dropout_rate=self.evoformer_cfg.dropout_rate, # 0.05
                                            gating=self.evoformer_cfg.gating, # True
                                            sink_attention=self.evoformer_cfg.sink_attention, # False
                                            norm_method=self.evoformer_cfg.norm_method, # rmsnorm
                                            intermediate_dim=self.evoformer_cfg.intermediate_dim, # 4
                                            post_ffn_operation_list=post_ffn_operation_list,
                                            init_method=self.evoformer_cfg.init_method, # AF2
                                            init_sigma=self.evoformer_cfg.init_sigma, # 0.02
                                            swish_beta=self.evoformer_cfg.swish_beta,) # 1.
            
            pair_evoformer_update_stack.append(msa_block)
        self.pair_evoformer_update_stack = pair_evoformer_update_stack

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

        # co evoformer update stack
        co_evoformer_update_stack = []
        post_ffn_operation_list = ('Dropout',)
        for i_ in range(self.co_update_evoformer_stack_num):
            if i_ == self.co_update_evoformer_stack_num -1:
                post_ffn_operation_list = ("Dropout", "LN")
            msa_block = FlashEvoformerStack(global_config=self.global_config,
                                            seq_act_dim=self.single_channel,
                                            pair_act_dim=self.pair_channel,
                                            outerproduct_dim=self.evoformer_cfg.outerproduct_dim,
                                            hidden_dim=self.evoformer_cfg.hidden_dim,
                                            num_head=self.evoformer_cfg.num_head,
                                            dropout_rate=self.evoformer_cfg.dropout_rate,
                                            gating=self.evoformer_cfg.gating,
                                            sink_attention=self.evoformer_cfg.sink_attention,
                                            norm_method=self.evoformer_cfg.norm_method,
                                            intermediate_dim=self.evoformer_cfg.intermediate_dim,
                                            post_ffn_operation_list=post_ffn_operation_list,
                                            init_method=self.evoformer_cfg.init_method,
                                            init_sigma=self.evoformer_cfg.init_sigma,
                                            swish_beta=self.evoformer_cfg.swish_beta,
                                            )
            co_evoformer_update_stack.append(msa_block)
        self.co_evoformer_update_stack = co_evoformer_update_stack

        self.extended_structure_module = StructureModule(self.global_config,
                                                         self.esm_cfg, ### 1 * 8 layer
                                                         self.seq_len, # 256
                                                         frozen_IPA=True,
                                                         share_weights=False,
                                                         stop_grad_ipa=False,
                                                         decoy_affine_init=True)
    
    # def __call__(seq_mask, aatype, decoy_affine_tensor,
    #              single_activations_init, pair_activations_init):
    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                 decoy_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask):
        
        """inputs of __call__ should be cast to self._dtype before put in the model"""
        # seq_mask: self._dtype, aatype: jnp.int32, residue_index: jnp.int32/self._dtype.
        # template_all_atom_masks: jnp.int32, template_all_atom_positions: jnp.float32, template_pseudo_beta: jnp.float32
        # torsion_angles_sin_cos: self._dtype, torsion_angles_mask: self._dtype
        # decoy_affine_tensor: jnp.float32(?)

        ### feature_initializer:
        single_activations_init, pair_activations_init = self.feat_init(seq_mask, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                 torsion_angles_sin_cos, torsion_angles_mask)
        
        mask_2d = jnp.expand_dims(seq_mask, -1) * jnp.expand_dims(seq_mask, -2) # [Nres, 1] * [1, Nres] -> [Nres, Nres] (self._dtype)
        
        ### get distance cutoff 
        ca_coords = template_all_atom_positions[..., 1, :].astype(jnp.float32) # [Nres, 3]
        distance_mtx = jnp.linalg.norm(ca_coords[..., None, :] - ca_coords[None, :, :], axis=-1) # [Nres, Nres]
        distance_cutoff = self.distance_cutoff_fn(distance_mtx) ### fp32
        distance_cutoff = jax.lax.stop_gradient(distance_cutoff) 

        attention_masks = (seq_mask, seq_mask, mask_2d)
        single_activations = single_activations_init
        pair_activations = pair_activations_init

        accumulated_single_act = single_activations
        accumulated_pair_act = pair_activations
        for i in range(self.pair_update_evoformer_stack_num):
            single_activations, pair_activations, accumulated_single_act, accumulated_pair_act \
                = self.pair_evoformer_update_stack[i](seq_act=single_activations, 
                                                      pair_act=pair_activations, 
                                                      accumulated_seq_act=accumulated_single_act, 
                                                      accumulated_pair_act=accumulated_pair_act, 
                                                      attention_masks=attention_masks,
                                                      distance_cutoff=distance_cutoff)
        

        single_act = single_activations_init
        acc_single_act = single_act
        pair_act = pair_activations
        for i in range(self.single_update_transformer_stack_num):
            single_act, acc_single_act = \
                self.single_residual_update_stack[i](act=single_act, # [num_batch, Nres, 256]
                                                     accumulated_act=acc_single_act, 
                                                     attention_masks=attention_masks, 
                                                     pair_act=pair_act,
                                                     distance_cutoff=distance_cutoff) # [num_batch, Nres, Nres, 128]

        # Added by @ZhangJ.:
        single_act = self.postln_scale * single_act + acc_single_act

        single_activations = single_act
        pair_activations = pair_activations
        accumulated_single_act = single_activations
        accumulated_pair_act = pair_activations
        for i in range(self.co_update_evoformer_stack_num):
            single_activations, pair_activations, accumulated_single_act, accumulated_pair_act \
                = self.co_evoformer_update_stack[i](seq_act=single_activations, 
                                                    pair_act=pair_activations, 
                                                    accumulated_seq_act=accumulated_single_act, 
                                                    accumulated_pair_act=accumulated_pair_act, 
                                                    attention_masks=attention_masks,
                                                    distance_cutoff=distance_cutoff)
                
        # pre_sm_single_act = self.pre_sm_single_update(single_activations) # 256 -> 384
                
        final_atom_positions, esm_single_act, atom14_pred_positions, final_affines, \
        angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj = \
            self.extended_structure_module(single_activations, # 256 # pre_sm_single_act 
                                           pair_activations,
                                           seq_mask,
                                           aatype,
                                           decoy_affine_tensor)
        
        final_single_activations = self.esm_post_ln(esm_single_act) # 256

        return final_single_activations, single_activations, pair_activations
    
class Local_VQ_Encoder(nn.Module):
    global_config: ml_collections.ConfigDict
    cfg: Config
    # inverse_folding: bool = True

    def setup(self):
        self.cfg.seq_len = self.cfg.common.max_n_neighbors
        
        self.batched_encoder = \
            nn.vmap(VQ_Encoder, 
                    in_axes=0,
                    variable_axes={'params': None},
                    split_rngs={'params': False, 'dropout': True})(self.global_config, self.cfg)
        self.max_n_neighbors = self.cfg.common.max_n_neighbors
        self.cutoff = self.cfg.common.cutoff
    
    def gather_neighbors(self, all_atom_positions, seq_mask, args):
        ca_coords = all_atom_positions[..., 1, :].astype(jnp.float32) # [Nres, 3] # take CA atoms
        distance_mtx = jnp.linalg.norm(ca_coords[..., None, :] - ca_coords[None, :, :], axis=-1) # [Nres, Nres]
        distance_mtx = distance_mtx + (1.0 - jnp.logical_and(seq_mask[None,...], seq_mask[..., None])).astype(distance_mtx.dtype) * 1e6 ### mask out self distance
        
        ### get neighbo_index 
        distance_mtx_neighbor = jnp.sort(distance_mtx, axis=-1)[:, :self.max_n_neighbors] # [Nres, Nneighbors]
        neighbor_index = jnp.argsort(distance_mtx, axis=-1)[:, :self.max_n_neighbors] # [Nres, Nneighbors]
        
        def _gather(x):
            nres = x.shape[0]
            x = jnp.expand_dims(x, axis=0) ### (Nres, ?) -> (1, Nres, ?)
            x = jnp.repeat(x, nres, axis=0) ### (1, Nres, ?) -> (Nres, Nres, ?)
            return jax.vmap(jnp.take, (0, 0, None))(x, neighbor_index, 0)
        
        seq_mask, args = jax.tree_map(_gather, (seq_mask, args))
        seq_mask = jnp.logical_and(seq_mask, distance_mtx_neighbor < self.cutoff)
        
        # jax.debug.breakpoint()
        
        return seq_mask, args
    
    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                 decoy_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask):
        
        seq_mask, (aatype, residue_index, template_all_atom_masks, 
                   template_all_atom_positions, template_pseudo_beta,
                   decoy_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask) = \
                self.gather_neighbors(template_all_atom_positions, seq_mask, 
                                      (aatype, residue_index, template_all_atom_masks, 
                                       template_all_atom_positions, template_pseudo_beta, 
                                       decoy_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask))
            
        final_single_activations, single_activations, pair_activations = \
            self.batched_encoder(seq_mask, aatype, residue_index, 
                                template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                                decoy_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
        
        return final_single_activations[:, 0, ...], None, None