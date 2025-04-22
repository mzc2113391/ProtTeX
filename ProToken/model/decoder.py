"""encoder model"""
import jax
import numpy as np
import jax.numpy as jnp

from flax import linen as nn
from common.config_load import Config

import common.residue_constants as residue_constants
from modules.basic import RelativePositionEmbedding

from model.flash_evoformer import FlashEvoformerStack
from modules.structure import StructureModule, FrameInitializer
from model.transformers import SelfResidualTransformer
from modules.head import DistogramHead
from modules.head import PredictedLDDTHead
from modules.transformer_blocks import OuterProduct
from modules.basic import ActFuncWrapper
from jax.experimental.host_callback import call

import ml_collections



class VQ_Decoder(nn.Module):

    global_config: ml_collections.ConfigDict
    cfg: Config # self.cfg = cfg
    pre_layer_norm: bool = True

    def setup(self):
        
        # config init
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.seq_len = self.cfg.seq_len

        self.postln_scale = self.cfg.common.postln_scale 
        self.single_channel = self.cfg.common.single_channel 
        self.pair_channel = self.cfg.common.pair_channel 

        self.pair_update_evoformer_stack_num = self.cfg.pair_update_evoformer_stack_num
        self.single_update_transformer_stack_num = self.cfg.single_update_transformer_stack_num
        self.co_update_evoformer_stack_num = self.cfg.co_update_evoformer_stack_num
        
        # layernorm to small numerical values 
        if self.pre_layer_norm:
            self.slv_layer_norm = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
        
        # default dense setting
        self.vq_activations_left = nn.Dense(features = self.pair_channel, # 256 -> 128
                                            kernel_init=nn.initializers.lecun_normal(),
                                            dtype=self._dtype, param_dtype=jnp.float32)
        self.vq_activations_right = nn.Dense(features = self.pair_channel, # 256 -> 128
                                            kernel_init=nn.initializers.lecun_normal(),
                                            dtype=self._dtype, param_dtype=jnp.float32)
        
        # rel_pos embedding setting
        self.rel_pos = RelativePositionEmbedding(global_config=self.global_config,
                                                 exact_distance=self.cfg.rel_pos.exact_distance, # 16
                                                 num_buckets=self.cfg.rel_pos.num_buckets, # 32
                                                 max_distance=self.cfg.rel_pos.max_distance,) # 64
        self.pair_activations = nn.Dense(features = self.pair_channel, # 64 -> 128
                                         kernel_init=nn.initializers.lecun_normal(),
                                         dtype=self._dtype, param_dtype=jnp.float32)
        
        # pair_update_evoformer_stack
        self.evoformer_cfg = self.cfg.evoformer
        pair_update_evoformer_stack = []
        post_ffn_operation_list = ("Dropout",)
        for i_ in range(self.pair_update_evoformer_stack_num):
            if i_ == self.pair_update_evoformer_stack_num -1: ### 最后一层需要ResidualLN:
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
                                            # pre_attention_operation_list=pre_attention_operation_list,
                                            # post_attention_operation_list=post_attention_operation_list,
                                            post_ffn_operation_list=post_ffn_operation_list,
                                            init_method=self.evoformer_cfg.init_method,
                                            init_sigma=self.evoformer_cfg.init_sigma,
                                            swish_beta=self.evoformer_cfg.swish_beta,)
            pair_update_evoformer_stack.append(msa_block)
        self.pair_update_evoformer_stack = pair_update_evoformer_stack
        
        # single_update_transformer_stack
        self.transformer_cfg = self.cfg.transformer
        single_update_transformer_stack = []
        post_ffn_operation_list = ("Dropout",)
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
                                               swish_beta=self.transformer_cfg.swish_beta,)
            single_update_transformer_stack.append(rt_block)
        self.single_update_transformer_stack = single_update_transformer_stack
        
        # co evoformer update stack
        co_update_evoformer_stack = []
        post_ffn_operation_list = ("Dropout",)
        for i_ in range(self.co_update_evoformer_stack_num):
            if i_ == self.co_update_evoformer_stack_num -1: ### 最后一层需要ResidualLN:
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
                                            # pre_attention_operation_list=pre_attention_operation_list,
                                            # post_attention_operation_list=post_attention_operation_list,
                                            post_ffn_operation_list=post_ffn_operation_list,
                                            init_method=self.evoformer_cfg.init_method,
                                            init_sigma=self.evoformer_cfg.init_sigma,
                                            swish_beta=self.evoformer_cfg.swish_beta,)
            co_update_evoformer_stack.append(msa_block)
        self.co_update_evoformer_stack = co_update_evoformer_stack
        
        # distogram head setting
        self.module_distogram = DistogramHead(self.global_config,
                                              self.cfg.distogram, self.pair_channel)
        
    def __call__(self, vq_act, seq_mask, residue_index):
        # vq_act: (B, Nres, C=384)
        # seq_mask: (B, Nres)
        # residue_index: (B, Nres)
        
        if self.pre_layer_norm:
            vq_act = self.slv_layer_norm(vq_act)
        
        _, rel_pos = self.rel_pos(residue_index, residue_index) #[num_batch, Nres, Nres, 64]
        pair_act = self.pair_activations(rel_pos) # [num_batch, Nres, Nres, 128]

        # @ZhangJ. ToDo: pair_act = outersum(left(vq), right(vq)) ? as AF2?
        # @ZhangJ. 这里有问题；参考AF2 SI Alg.#3
        # @Zhenyu. OK
        # vq_pair_act = self.vq_activations(jnp.expand_dims(vq_act, -2) + jnp.expand_dims(vq_act, -3)) # [num_batch, Nres, Nres, 128]
        # vq_pair_act_left = self.vq_activations_left(vq_act)
        # vq_pair_act_right = self.vq_activations_right(vq_act)
        # vq_pair_act = jnp.expand_dims(vq_pair_act_left, axis=-2) + jnp.expand_dims(vq_pair_act_right, axis=-3) # [num_batch, Nres, Nres, 128]

        # pair_act += vq_pair_act ### !! modified in PT1.1, do not add vq_act here
        # pair_act_op = self.outer_product(vq_act, seq_mask) # [num_batch, Nres, Nres, 128]
        # pair_act += pair_act_op
        # print(pair_act_op.shape)

        mask_2d = jnp.expand_dims(seq_mask, axis=-1) * jnp.expand_dims(seq_mask, axis=-2) # [num_batch, Nres, Nres]
        attention_masks = (seq_mask, seq_mask, mask_2d)
        
        ### updating pair_act
        single_activations = vq_act
        pair_activations = pair_act
        accumulated_single_act = single_activations
        accumulated_pair_act = pair_activations
        for i in range(self.pair_update_evoformer_stack_num):
            single_activations, pair_activations, accumulated_single_act, accumulated_pair_act \
                = self.pair_update_evoformer_stack[i](seq_act=single_activations, 
                                                      pair_act=pair_activations, 
                                                      accumulated_seq_act=accumulated_single_act, 
                                                      accumulated_pair_act=accumulated_pair_act, 
                                                      attention_masks=attention_masks)
        # # Added by @ZhangJ.:
        # pair_activations = self.postln_scale * pair_activations + accumulated_pair_act

        ### updating single_act
        single_act = vq_act
        accumulated_act = single_act
        for i in range(self.single_update_transformer_stack_num):
            single_act, accumulated_act = \
                self.single_update_transformer_stack[i](single_act, accumulated_act, attention_masks, pair_act=pair_activations)
        # Added by @ZhangJ.:
        single_act = self.postln_scale * single_act + accumulated_act

        ### 3. Evoformer: 最后执行第二个Evoformer同时更新pair & single
        single_act = single_act # (B,Nres,C=256)
        pair_act = pair_activations # (B, Nres, Nres, c=192)
        accumulated_single_act = single_act
        accumulated_pair_act = pair_act
        # single_act = self.single_activations(seq_act) #[num_batch, Nres, C=256]
        for it_evo in range(self.co_update_evoformer_stack_num):
            single_act, pair_act, accumulated_single_act, accumulated_pair_act= \
                self.co_update_evoformer_stack[it_evo](seq_act=single_act, # 256
                                                        pair_act=pair_act,  # 128
                                                        accumulated_seq_act=accumulated_single_act, 
                                                        accumulated_pair_act=accumulated_pair_act, 
                                                        attention_masks=attention_masks)
        
        # # Corrected by @ZhangJ.:
        # single_act = self.postln_scale * single_act + accumulated_single_act
        # pair_act = self.postln_scale * pair_act + accumulated_pair_act

        ### 4. DistogramHead
        dist_logits, dist_bin_edges = self.module_distogram(pair_act)

        return single_act, pair_act, dist_logits, dist_bin_edges
        

class Protein_Decoder(nn.Module):
    
    global_config: ml_collections.ConfigDict
    cfg: Config
    ipa_share_weights: bool = True
    
    def setup(self):
        
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32



        # default value setting
        self.seq_len = self.cfg.seq_len # 256
        self.esm_cfg = self.cfg.extended_structure_module
        self.fi_cfg = self.cfg.frame_initializer
        
        self.pre_sm_single_update = nn.Dense(self.esm_cfg.single_channel, kernel_init=nn.initializers.lecun_normal(),
                                             dtype=self._dtype, param_dtype=jnp.float32) # 256->384
        
        
        # structure module setting
        self.frame_initializer = FrameInitializer(global_config=self.global_config,
                                                  config=self.fi_cfg,
                                                  seq_length=self.seq_len, # 256
                                                  stop_grad_ipa=self.fi_cfg.stop_grad_ipa,
                                                  share_weights=False) # 2023.11.03
        
        self.structure_decoder_module_lonely = StructureModule(global_config=self.global_config,
                                                               config=self.esm_cfg, 
                                                               seq_length=self.seq_len, # 256
                                                               frozen_IPA=False,
                                                               share_weights=self.ipa_share_weights,
                                                               stop_grad_ipa=self.esm_cfg.stop_grad_ipa,
                                                               decoy_affine_init=True,
                                                               ret_single_pair=True) # 2023.11.02
        
        self.module_pLDDT = PredictedLDDTHead(global_config=self.global_config,
                                              cfg=self.cfg.predicted_lddt)

        # ### debug
        # self.mi_grad_single_activations = self.param(
        #     'mi_grad_single_activations', nn.initializers.zeros,
        #     (256, 384), jnp.float32
        # )
        
    
    def __call__(self, single_act, pair_act, seq_mask, aatype):
        """construct"""
        # single_act:(Nres, C=384); 
        # pair_act:(Nres, Nres, C=192); 
        # seq_mask:(Nres,);
        # aatype:(Nres,);
        
        ### 0. 准备features:
        aatype = jnp.ones_like(aatype).astype(jnp.int32) * 7 ### all Gly [Nres,]
        # mask_2d = jnp.expand_dims(seq_mask, axis=-1) * jnp.expand_dims(seq_mask, axis=-2) # [Nres, Nres]
        # attention_mask=(seq_mask,seq_mask,mask_2d)

        # Rewritten by @ZhangJ.:
        single_activations = single_act # [Nres, C=384]
        pair_activations = pair_act # [Nres, Nres, C=192]

        # fp_convert_list = [single_activations, pair_activations]
        # single_activations, pair_activations = jax.tree_util.tree_map(lambda x: jnp.asarray(x, self._dtype), fp_convert_list)
        
        pre_sm_single_act = self.pre_sm_single_update(single_activations) # 256 -> 384
        
        ### debug
        # pre_sm_single_act = pre_sm_single_act + self.mi_grad_single_activations
            
        affine_output_new, final_affines, act_initializer = self.frame_initializer(pre_sm_single_act, # single_activations,
                                                                                   pair_activations,
                                                                                   seq_mask,
                                                                                   aatype,)
        single_activations = act_initializer # 384
        
        # print("affine_output_new.shape:", affine_output_new.shape) # [NBATCH, NLAYER, NRES, 7]
        # print("final_affines.shape:", final_affines.shape) # [NBATCH, NRES, 7]
        
        # ### debug
        # single_activations = single_activations + self.mi_grad_single_activations

        final_atom_positions, single_activations, atom14_pred_positions, final_affines, \
        angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj, normed_single, normed_pair = \
            self.structure_decoder_module_lonely(single_activations,
                                                 pair_activations,
                                                 seq_mask,
                                                 aatype,
                                                 decoy_affine_tensor=final_affines)

        # final_atom_positions, single_activations, atom14_pred_positions, final_affines, \
        # angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj = \
        #     self.structure_decoder_module_share(single_activations,
        #                                          pair_activations,
        #                                          seq_mask,
        #                                          aatype,
        #                                          decoy_affine_tensor=None)
        
        # pLDDT
        # jax.tree_util.tree_map(lambda x: call(lambda y: print('single_activations', y.shape), x), single_activations)
        pLDDT_logits = self.module_pLDDT(single_activations, normed_pair, seq_mask, final_affines, aatype)
        # jax.tree_util.tree_map(lambda x: call(lambda y: print('pLDDT_logits', y.shape), x), pLDDT_logits)
        # output_list = [final_atom_positions, atom14_pred_positions, structure_traj]
        # final_atom_positions, atom14_pred_positions, structure_traj = jax.tree_util.tree_map(lambda x: jnp.asarray(x, self._dtype), output_list)
        return final_atom_positions, atom14_pred_positions, structure_traj, normed_single, normed_pair, pLDDT_logits