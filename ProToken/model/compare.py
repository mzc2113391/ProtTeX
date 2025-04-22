import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn
from jax import jit, vmap
import optax

import os
from typing import Callable
from flax.linen.initializers import lecun_normal, ones_init, zeros_init, he_uniform, constant, variance_scaling, truncated_normal, normal
from common.config_load import load_config, Config
from tokenizer.finite_scalar_quantization import FSQTokenizer
from loss.CA_distogram_loss import CA_DistogramLoss
from loss.fape_loss import backbone_loss_affine_with_weights
from loss.structure_violation_loss import structural_violation_loss, find_structural_violations_array
from loss.confidence_loss import IntegratedBCEpLDDTLoss, lddt
from loss.inverse_folding_loss import softmax_cross_entropy
from loss.AF2_supervised_loss import supervised_single_mse_loss, supervised_pair_mse_loss, supervised_single_cos_sim_loss, supervised_pair_cos_sim_loss
from loss.utils import _l2_normalize, square_euclidean_distance
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# from config.global_setup import EnvironConfig
# global_setup = EnvironConfig() ### Set Hyper-parameters here
# NORM_SMALL = global_setup.norm_small
# # MIXED_PRECISION_FLAG = global_setup.mixed_precision_flag
# BF16_FLAG = global_setup.bf16_flag
# SAFE_PRECISION_FLAG = global_setup.safe_precision_flag
# DROPOUT_FLAG = global_setup.use_dropout

# def l2_normalize(x, axis=-1, epsilon=1e-12, dtype=jnp.float32):
#     if SAFE_PRECISION_FLAG:
#         x = jnp.array(x, dtype=jnp.float32)
#         x = _l2_normalize(x, axis=axis, epsilon=epsilon)
#         x = jnp.array(x, dtype=dtype)
#     else:
#         x = _l2_normalize(x, axis=axis, epsilon=epsilon)
        
#     return x

def l2_normalize(x, axis=-1, epsilon=1e-12, dtype=jnp.float32):

    x = _l2_normalize(x, axis=axis, epsilon=epsilon)   
    return x

## @Liyh. Safe precision for l2_norm
def safe_l2_normalize(x, axis=-1, epsilon=1e-12, dtype=jnp.float32):
    _dtype = x.dtype
    x = x.astype(jnp.float32)
    x = _l2_normalize(x, axis=axis, epsilon=epsilon)   
    return x.astype(_dtype)

class PretrainEncoderWithLossCell(nn.Module):

    global_config: Config
    train_cfg: Config
    encoder: nn.Module
    vq_decoder: nn.Module
    protein_decoder: nn.Module
    
    def setup(self):
        self.distogram_loss_func = CA_DistogramLoss(self.train_cfg.distogram)
        self.confidence_loss_func = IntegratedBCEpLDDTLoss(self.train_cfg.confidence)
        
        ####### loss weights
        self.fape_loss_weight = self.train_cfg.fape.loss_weight
        self.fape_IPA_weight = jnp.array(self.train_cfg.fape.IPA_weight, dtype=jnp.float32)
        self.fape_IPA_weight = self.fape_IPA_weight / jnp.sum(self.fape_IPA_weight)
        self.violation_loss_weight = self.train_cfg.structural_violation.loss_weight
        self.distogram_w1 = self.train_cfg.distogram.w1
        self.distogram_w2 = self.train_cfg.distogram.w2
        self.distogram_w3 = self.train_cfg.distogram.w3
        self.distogram_loss_weight = self.train_cfg.distogram.weight
        self.confidence_loss_weight = self.train_cfg.confidence.loss_weight
        self.inverse_folding_loss_weight = self.train_cfg.inverse_folding.loss_weight
        self.AF2_supervised_single_mse_loss_weight = self.train_cfg.AF2_supervised_loss.single_mse_loss_weight
        self.AF2_supervised_single_cos_sim_loss_weight = self.train_cfg.AF2_supervised_loss.single_cos_sim_loss_weight
        self.AF2_supervised_pair_mse_loss_weight = self.train_cfg.AF2_supervised_loss.pair_mse_loss_weight
        self.AF2_supervised_pair_cos_sim_loss_weight = self.train_cfg.AF2_supervised_loss.pair_cos_sim_loss_weight

        self.seq_len_power = self.train_cfg.seq_len_power

        self.bf16_flag = self.global_config.bf16_flag
        self.safe_precision_flag = self.global_config.safe_precision_flag        
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self._safedtype = jnp.float32 if self.safe_precision_flag else self._dtype

    def __call__(self, seq_mask, true_aatype, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,
                 atom14_atom_exists, dist_gt_perms, dist_mask_perms, perms_padding_mask, AF2_normed_single, AF2_normed_pair, AF2_supervised_mask):
        
        ####### generate keys 
        fape_clamp_key = self.make_rng('fape_clamp_key')
        dmat_rng_key = self.make_rng('dmat_rng_key')
        
        if self.bf16_flag:
            bf16_process_list = [template_all_atom_positions, template_pseudo_beta,
                                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask]

            template_all_atom_positions, template_pseudo_beta, \
            backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask = jax.tree_util.tree_map(lambda x: jnp.bfloat16(x), bf16_process_list)
        
        ########### encoding
        single_act, inverse_folding_logits = self.encoder(seq_mask, aatype, residue_index,
                                  template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                                  backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
        
        ########### inverse folding loss
        inverse_folding_loss = 0.0
        if self.inverse_folding_loss_weight > 0.0:
            inverse_folding_logits = jnp.array(inverse_folding_logits, dtype=jnp.float32)
            true_aatype_onehot = jax.nn.one_hot(true_aatype, num_classes=20)
            inverse_folding_loss = softmax_cross_entropy(inverse_folding_logits, true_aatype_onehot, seq_mask)
            
        ########### l2 norm of single act 
        single_act = safe_l2_normalize(single_act, axis=-1, epsilon=self.global_config.norm_small, dtype=self._dtype) ## @Liyh: changed to safe l2_norm
                
        ########### vq decoder
        single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder(single_act, seq_mask, residue_index)
        
        ########### distogram loss
        dmat_loss, lddt_loss, contact_loss = 0.0, 0.0, 0.0
        if self.distogram_loss_weight > 0.0:
            dist_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask =\
                jax.tree_util.tree_map(jnp.float32, [dist_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask])
            
            dmat_loss, lddt_loss, contact_loss = self.distogram_loss_func(dist_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask, dmat_rng_key)

        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, normed_single, normed_pair, pLDDT_logits = self.protein_decoder(single_act_decode, pair_act_decode, seq_mask, aatype)

        ########### supervised loss
        single_mse_loss, single_cos_sim_loss, pair_mse_loss, pair_cos_sim_loss = 0.0, 0.0, 0.0, 0.0
        if self.AF2_supervised_single_mse_loss_weight > 0.0:
            normed_single, AF2_normed_single = jax.tree_util.tree_map(jnp.float32, [normed_single, AF2_normed_single])
            single_mse_loss = supervised_single_mse_loss(normed_single, AF2_normed_single, seq_mask)
            single_mse_loss = single_mse_loss * AF2_supervised_mask
            single_cos_sim_loss = supervised_single_cos_sim_loss(normed_single, AF2_normed_single, seq_mask)
            single_cos_sim_loss = single_cos_sim_loss * AF2_supervised_mask
        if self.AF2_supervised_pair_mse_loss_weight > 0.0:
            normed_pair, AF2_normed_pair = jax.tree_util.tree_map(jnp.float32, [normed_pair, AF2_normed_pair])
            pair_mse_loss = supervised_pair_mse_loss(normed_pair, AF2_normed_pair, seq_mask)
            pair_mse_loss = pair_mse_loss * AF2_supervised_mask
            pair_cos_sim_loss = supervised_pair_cos_sim_loss(normed_pair, AF2_normed_pair, seq_mask)
            pair_cos_sim_loss = pair_cos_sim_loss * AF2_supervised_mask
        
        ########### fape loss:
        final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor = \
            jax.tree_util.tree_map(jnp.float32, [final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor])
        no_clamp_mask = jax.random.bernoulli(fape_clamp_key, p=0.9, shape=(structure_traj.shape[0], seq_mask.shape[0]))

        fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
            gt_rigid_affine=backbone_affine_tensor, 
            gt_frames_mask=seq_mask, 
            gt_positions_mask=seq_mask,
            target_rigid_affine=structure_traj,
            config=self.train_cfg,
            no_clamp_mask=no_clamp_mask,
            pair_mask=seq_mask[None, :] * seq_mask[:, None],
            IPA_weights=self.fape_IPA_weight,
        )
        
        ########### structure violation loss 
        structure_violation_loss = 0.0
        if self.violation_loss_weight > 0.0:
            asym_id = jnp.zeros_like(seq_mask, dtype=jnp.int32)
            violation_result_dict = find_structural_violations_array(
                aatype=aatype,
                residue_index=residue_index,
                mask=atom14_atom_exists,
                pred_positions=final_atom14_positions,
                config=self.train_cfg,
                asym_id=asym_id,
            )
            structure_violation_loss = structural_violation_loss(seq_mask, violation_result_dict)
        
        structure_loss = self.fape_loss_weight * fape_loss + \
                         self.violation_loss_weight * structure_violation_loss

        distogram_loss = self.distogram_w1 * dmat_loss + \
                         self.distogram_w2 * contact_loss + \
                         self.distogram_w3 * lddt_loss
        # @ZhangJ.
        reconstruction_loss = structure_loss + \
                              self.distogram_loss_weight * distogram_loss
        
        # @ZhangJ.
        aux_loss = self.inverse_folding_loss_weight * inverse_folding_loss
                    
        AF2_supervised_loss = self.AF2_supervised_single_mse_loss_weight * single_mse_loss + \
                    self.AF2_supervised_single_cos_sim_loss_weight * single_cos_sim_loss + \
                    self.AF2_supervised_pair_mse_loss_weight * pair_mse_loss + \
                    self.AF2_supervised_pair_cos_sim_loss_weight * pair_cos_sim_loss
        
        ########### confidence loss
        confidence_loss = 0.0
        if self.confidence_loss_weight > 0.0:
            true_lddt = lddt(final_atom14_positions[None, :, 1, :], template_all_atom_positions[None, :, 1, :], seq_mask[None, :, None], per_residue=True)[0] * 100.0 # CA [0, 1] -> [0, 100], lddt is a batched function
            pLDDT_logits = jnp.array(pLDDT_logits, dtype=jnp.float32)
            confidence_loss = self.confidence_loss_func(jax.nn.softmax(pLDDT_logits, axis=-1), true_lddt, seq_mask)
        
        aux_loss += confidence_loss * self.confidence_loss_weight

        ########### seq length power
        seq_len_weight = jnp.power(jnp.sum(seq_mask), self.seq_len_power)
        
        loss = (reconstruction_loss + aux_loss)
        
        loss_dict = {
            "loss": loss,
            "inverse_folding_loss": inverse_folding_loss,
            "dmat_loss": dmat_loss,
            "contact_loss": contact_loss,
            "lddt_loss": lddt_loss,
            "fape_loss": fape_loss,
            "fape_last_IPA": fape_last_IPA,
            "fape_no_clamp_last_IPA": no_clamp_last_IPA,
            "structure_violation_loss": structure_violation_loss,
            "confidence_loss": confidence_loss,
            "AF2_supervision": {
                "AF2_supervised_loss": AF2_supervised_loss,
                "AF2_supervised_mask": AF2_supervised_mask,
                "single_mse_loss": single_mse_loss,
                "single_cos_sim_loss": single_cos_sim_loss,
                "pair_mse_loss": pair_mse_loss,
                "pair_cos_sim_loss": pair_cos_sim_loss,
            } 
        }
        
        return loss_dict, seq_len_weight
    

def pretrain_encoder_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    ##### AF2 supervised signal reweight
    AF2_supervision_dict = loss_dict.pop("AF2_supervision")
    AF2_supervised_mask = AF2_supervision_dict.pop("AF2_supervised_mask")
        
    seq_len_weight = seq_len_weight /(jnp.sum(seq_len_weight) + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    
    AF2_supervised_weight = seq_len_weight * AF2_supervised_mask
    AF2_supervised_weight = AF2_supervised_weight / (jnp.sum(AF2_supervised_weight) + 1e-6)
    AF2_supervision_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * AF2_supervised_weight), AF2_supervision_dict)
    
    loss_dict["single_mse_loss"] = AF2_supervision_dict["single_mse_loss"]
    loss_dict["single_cos_sim_loss"] = AF2_supervision_dict["single_cos_sim_loss"]
    loss_dict["pair_mse_loss"] = AF2_supervision_dict["pair_mse_loss"]
    loss_dict["pair_cos_sim_loss"] = AF2_supervision_dict["pair_cos_sim_loss"]
    
    # loss & aux loss valuies
    loss = loss_dict.pop("loss")
    loss = loss + AF2_supervision_dict.pop("AF2_supervised_loss")
    return loss, loss_dict


def pretrain_encoder_forward_per_device(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    ##### AF2 supervised signal reweight
    AF2_supervision_dict = loss_dict.pop("AF2_supervision")
    AF2_supervised_mask = AF2_supervision_dict.pop("AF2_supervised_mask")
    AF2_supervised_weight = seq_len_weight * AF2_supervised_mask
    ##### we need a psum here to aggregate the weights
    AF2_supervised_weight = AF2_supervised_weight / \
                        (jax.lax.psum(jnp.sum(AF2_supervised_weight), axis_name="i") + 1e-6)
    AF2_supervision_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * AF2_supervised_weight), AF2_supervision_dict)
    
    # weights_sum = jnp.sum(seq_len_weight) + 1e-6
    seq_len_weight = seq_len_weight / \
                        (jax.lax.psum(jnp.sum(seq_len_weight), axis_name="i") + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    
    loss_dict["single_mse_loss"] = AF2_supervision_dict["single_mse_loss"]
    loss_dict["single_cos_sim_loss"] = AF2_supervision_dict["single_cos_sim_loss"]
    loss_dict["pair_mse_loss"] = AF2_supervision_dict["pair_mse_loss"]
    loss_dict["pair_cos_sim_loss"] = AF2_supervision_dict["pair_cos_sim_loss"]
    
    loss = loss_dict.pop("loss")
    loss = loss + AF2_supervision_dict.pop("AF2_supervised_loss")
    
    return loss, loss_dict

#### abandon
# def pretrain_encoder_forward_psum(vmap_fn, params, batch_input, net_rng_key):
    
#     @jax.jit
#     def _forward_(params, batch_input, net_rng_key):
#         return vmap_fn(params, *batch_input, rngs=net_rng_key)
    
#     loss_dict, seq_len_weight = _forward_(params, batch_input, net_rng_key)
    
#     # pmap_reduce_func = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name="i")
#     weights_reduced = jax.lax.psum(jnp.sum(seq_len_weight), 'i')
#     loss_dict = jax.tree_util.tree_map(lambda x: jax.lax.psum(jnp.sum(x * seq_len_weight), 'i') / weights_reduced, loss_dict)
    
#     loss = loss_dict.pop("loss")
#     return loss, loss_dict


class PretrainFSQWithLossCell(nn.Module):

    global_config: Config
    train_cfg: Config
    bottleneck_encoder: nn.Module
    bottleneck_decoder: nn.Module
    encoder_fn: Callable ### params are wrapped with functools.partial
    vq_decoder_fn: Callable ### params are wrapped with functools.partial
    prot_decoder_fn: Callable ### params are wrapped with functools.partial
    fsq_tokenizer: Callable
    fsq_cfg: Config
    quantize: bool = True
    
    def setup(self):
        self.bf16_flag = self.global_config.bf16_flag
        self.safe_precision_flag = self.global_config.safe_precision_flag
        self.dropout_flag = self.global_config.use_dropout

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self._safedtype = jnp.float32 if self.safe_precision_flag else self._dtype
        
        self.drop_out_in = nn.Dropout(rate=self.train_cfg.dropout_rate, 
                                      deterministic=(not self.dropout_flag))
        self.project_in = nn.Dense(features=len(self.fsq_cfg.levels),
                                   # kernel_init=he_uniform())
                                   # kernel_init=lecun_normal())
                                   kernel_init=truncated_normal(),
                                   use_bias=False)
                                   # kernel_init=variance_scaling(0.1, "fan_in", "truncated_normal"))
        self.project_out = nn.Dense(features=self.fsq_cfg.dim_in,
                                    # kernel_init=he_uniform())
                                    # kernel_init=lecun_normal(),
                                    kernel_init=truncated_normal(),
                                    use_bias=False)
                                    # kernel_init=zeros_init())
                                    # kernel_init=constant(0.1))
        self.drop_out_out = nn.Dropout(rate=self.train_cfg.dropout_rate, 
                                      deterministic=(not self.dropout_flag))
        
        ####### loss weights
        self.fape_loss_weight = self.train_cfg.fape.loss_weight
        self.fape_IPA_weight = jnp.array(self.train_cfg.fape.IPA_weight, dtype=jnp.float32)
        self.fape_IPA_weight = self.fape_IPA_weight / jnp.sum(self.fape_IPA_weight)
        self.violation_loss_weight = self.train_cfg.structural_violation.loss_weight
        self.fsq_consistency_loss_weight = self.train_cfg.fsq_consistency.loss_weight
        self.l2_regularizer_loss_weight = self.train_cfg.l2_regularizer.loss_weight
        # self.l2_regularizer_threshold = self.train_cfg.l2_regularizer.threshold
        
        ####### for l2 regularizer: 
        self.fsq_act_lower_bound, self.fsq_act_upper_bound = self.fsq_tokenizer.get_act_bound() ##### fp32
        
        self.seq_len_power = self.train_cfg.seq_len_power
    
    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,
                 atom14_atom_exists):
        
        ####### generate keys 
        fape_clamp_key = self.make_rng('fape_clamp_key')
        
        ########### encoding
        single_act, inverse_folding_logits = self.encoder_fn(
                                  seq_mask, aatype, residue_index,
                                  template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                                  backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
        
        ########### l2 norm of single act 
        if self._dtype != jnp.float32:
            single_act = jnp.array(single_act, dtype=jnp.float32)
        single_act_normalized = _l2_normalize(single_act, axis=-1, epsilon=self.global_config.norm_small)
        
        ########### projection & quantization
        single_act = self.bottleneck_encoder(single_act_normalized, seq_mask, residue_index)
        #### debug
        # single_act_bak = single_act
        
        single_act = self.drop_out_in(single_act)
        single_act = self.project_in(single_act)
        l2_regularizer_loss = 0.0
        if self.l2_regularizer_loss_weight > 0.0:
            l2_regularizer_loss = \
            jnp.sum(
                seq_mask[..., None] * (
                    nn.relu(single_act - self.fsq_act_upper_bound[None, ...]) + \
                    nn.relu(self.fsq_act_lower_bound[None, ...] - single_act))
                ) / (jnp.sum(seq_mask) * single_act.shape[-1] + 1e-6)
        
        fsq_act = self.fsq_tokenizer(single_act, quantize=self.quantize)
        code_count = self.fsq_tokenizer.count_codes(fsq_act, seq_mask)
        #### debug
        # single_act_fsq = self.project_out(single_act)
        single_act_fsq = self.project_out(fsq_act)
        single_act_fsq = self.drop_out_out(single_act_fsq)
    
        #### debug 
        # single_act_fsq = self.bottleneck_decoder(single_act_bak, seq_mask)
        single_act_fsq = self.bottleneck_decoder(single_act_fsq, seq_mask, residue_index)
        single_act_fsq = safe_l2_normalize(single_act_fsq, axis=-1, epsilon=self.global_config.norm_small)
        
        ########## fsq consistency loss 
        fsq_consistency_loss = 0.0 
        if self.fsq_consistency_loss_weight > 0.0:
            fsq_consistency_loss = square_euclidean_distance(single_act_normalized, single_act_fsq, axis=-1, normalized=True)
            fsq_consistency_loss = jnp.sum(fsq_consistency_loss * seq_mask) / (jnp.sum(seq_mask) + 1e-6) 
        
        ########### vq decoder
        if self._dtype != jnp.float32:
            single_act_fsq = jnp.array(single_act_fsq, dtype=self._dtype)
        single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder_fn(single_act_fsq, seq_mask, residue_index)
        # #### debug 
        # single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder_fn(single_act_normalized, seq_mask, residue_index)
        
        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, normed_single, normed_pair, pLDDT_logits = self.prot_decoder_fn(single_act_decode, pair_act_decode, seq_mask, aatype)
        
        ########### fape loss:
        fape_loss, fape_last_IPA, no_clamp_last_IPA = 0.0, 0.0, 0.0
        if self.fape_loss_weight > 0.0:
            final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor = \
                jax.tree_util.tree_map(jnp.float32, [final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor])
            no_clamp_mask = jax.random.bernoulli(fape_clamp_key, p=0.9, shape=(structure_traj.shape[0], seq_mask.shape[0]))

            fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
                gt_rigid_affine=backbone_affine_tensor, 
                gt_frames_mask=seq_mask, 
                gt_positions_mask=seq_mask,
                target_rigid_affine=structure_traj,
                config=self.train_cfg,
                no_clamp_mask=no_clamp_mask,
                pair_mask=seq_mask[None, :] * seq_mask[:, None],
                IPA_weights=self.fape_IPA_weight,
            )
        
        ########### structure violation loss 
        structure_violation_loss = 0.0
        if self.violation_loss_weight > 0.0:
            final_atom14_positions = jnp.array(final_atom14_positions, dtype=jnp.float32)
            asym_id = jnp.zeros_like(seq_mask, dtype=jnp.int32)
            violation_result_dict = find_structural_violations_array(
                aatype=aatype,
                residue_index=residue_index,
                mask=atom14_atom_exists,
                pred_positions=final_atom14_positions,
                config=self.train_cfg,
                asym_id=asym_id,
            )
            structure_violation_loss = structural_violation_loss(seq_mask, violation_result_dict)
            
        ########### seq length power
        seq_len_weight = jnp.power(jnp.sum(seq_mask), self.seq_len_power)
            
        structure_loss = self.fape_loss_weight * fape_loss + \
                         self.violation_loss_weight * structure_violation_loss
        loss = structure_loss + self.fsq_consistency_loss_weight * fsq_consistency_loss\
                + self.l2_regularizer_loss_weight * l2_regularizer_loss
        
        loss_dict = {
            "loss": loss, 
            "fape_loss": fape_loss,
            "fape_last_IPA": fape_last_IPA,
            "fape_no_clamp_last_IPA": no_clamp_last_IPA,
            "structure_violation_loss": structure_violation_loss,
            "fsq_consistency_loss": fsq_consistency_loss,
            "fsq_l2_regularizer_loss": l2_regularizer_loss,
        }
        
        return loss_dict, code_count, seq_len_weight
    
    
def pretrain_fsq_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    weights_sum = jnp.sum(seq_len_weight) + 1e-6
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight)/weights_sum, loss_dict)
    
    # code count: [B, Ncodes]
    code_count = jnp.sum(code_count, axis=0)
    code_usage = jnp.sum(
        jnp.array(code_count > 2, dtype=jnp.float32)) / code_count.shape[-1]
    
    # loss & aux loss valuies
    loss = loss_dict.pop("loss")
    return loss, (loss_dict, code_usage)


def pretrain_fsq_forward_per_device(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    code_count = jnp.sum(code_count, axis=0) # (B, Ncodes) -> (Ncodes)
    seq_len_weight = seq_len_weight / (jax.lax.psum(jnp.sum(seq_len_weight), axis_name="i") + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    loss = loss_dict.pop("loss")
    
    return loss, (loss_dict, code_count)

class PretrainFSQVQDecoderWithLossCell(nn.Module):
    global_config: Config
    train_cfg: Config
    bottleneck_encoder: nn.Module
    bottleneck_decoder: nn.Module
    encoder_fn: Callable ### params are wrapped with functools.partial
    vq_decoder: nn.Module
    prot_decoder_fn: Callable ### params are wrapped with functools.partial
    fsq_tokenizer: Callable
    fsq_cfg: Config
    quantize: bool = True
    
    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.safe_precision_flag = self.global_config.safe_precision_flag
        self.dropout_flag = self.global_config.use_dropout

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self._safedtype = jnp.float32 if self.safe_precision_flag else self._dtype
        
        self.drop_out_in = nn.Dropout(rate=self.train_cfg.dropout_rate, 
                                      deterministic=(not self.dropout_flag))
        self.project_in = nn.Dense(features=len(self.fsq_cfg.levels),
                                   # kernel_init=he_uniform())
                                   # kernel_init=lecun_normal())
                                   kernel_init=truncated_normal(),
                                   use_bias=False)
                                   # kernel_init=variance_scaling(0.1, "fan_in", "truncated_normal"))
        self.project_out = nn.Dense(features=self.fsq_cfg.dim_in,
                                    # kernel_init=he_uniform())
                                    # kernel_init=lecun_normal(),
                                    kernel_init=truncated_normal(),
                                    use_bias=False)
                                    # kernel_init=zeros_init())
                                    # kernel_init=constant(0.1))
        self.drop_out_out = nn.Dropout(rate=self.train_cfg.dropout_rate, 
                                      deterministic=(not self.dropout_flag))
        
        ####### loss weights
        self.fape_loss_weight = self.train_cfg.fape.loss_weight
        self.fape_IPA_weight = jnp.array(self.train_cfg.fape.IPA_weight, dtype=jnp.float32)
        self.fape_IPA_weight = self.fape_IPA_weight / jnp.sum(self.fape_IPA_weight)
        self.violation_loss_weight = self.train_cfg.structural_violation.loss_weight
        self.fsq_consistency_loss_weight = self.train_cfg.fsq_consistency.loss_weight
        self.l2_regularizer_loss_weight = self.train_cfg.l2_regularizer.loss_weight
        # self.l2_regularizer_threshold = self.train_cfg.l2_regularizer.threshold
        
        ####### for l2 regularizer: 
        self.fsq_act_lower_bound, self.fsq_act_upper_bound = self.fsq_tokenizer.get_act_bound() ##### fp32
        
        self.seq_len_power = self.train_cfg.seq_len_power
    
    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,
                 atom14_atom_exists):
        
        ####### generate keys 
        fape_clamp_key = self.make_rng('fape_clamp_key')
        
        ########### encoding
        single_act, inverse_folding_logits = self.encoder_fn(
                                  seq_mask, aatype, residue_index,
                                  template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                                  backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
        
        ########### l2 norm of single act 
        if self._dtype != jnp.float32:
            single_act = jnp.array(single_act, dtype=jnp.float32)
        single_act_normalized = safe_l2_normalize(single_act, axis=-1, epsilon=self.global_config.norm_small)
        
        ########### projection & quantization
        single_act = self.bottleneck_encoder(single_act_normalized, seq_mask, residue_index)
        #### debug
        # single_act_bak = single_act
        
        single_act = self.drop_out_in(single_act)
        single_act = self.project_in(single_act)
        l2_regularizer_loss = 0.0
        if self.l2_regularizer_loss_weight > 0.0:
            l2_regularizer_loss = \
            jnp.sum(
                seq_mask[..., None] * (
                    nn.relu(single_act - self.fsq_act_upper_bound[None, ...]) + \
                    nn.relu(self.fsq_act_lower_bound[None, ...] - single_act))
                ) / (jnp.sum(seq_mask) * single_act.shape[-1] + 1e-6)
        
        fsq_act = self.fsq_tokenizer(single_act, quantize=self.quantize)
        code_count = self.fsq_tokenizer.count_codes(fsq_act, seq_mask)
        #### debug
        # single_act_fsq = self.project_out(single_act)
        single_act_fsq = self.project_out(fsq_act)
        single_act_fsq = self.drop_out_out(single_act_fsq)
    
        #### debug 
        # single_act_fsq = self.bottleneck_decoder(single_act_bak, seq_mask)
        single_act_fsq = self.bottleneck_decoder(single_act_fsq, seq_mask, residue_index)
        single_act_fsq = safe_l2_normalize(single_act_fsq, axis=-1, epsilon=self.global_config.norm_small)
        
        ########## fsq consistency loss 
        fsq_consistency_loss = 0.0 
        if self.fsq_consistency_loss_weight > 0.0:
            fsq_consistency_loss = square_euclidean_distance(single_act_normalized, single_act_fsq, axis=-1, normalized=True)
            fsq_consistency_loss = jnp.sum(fsq_consistency_loss * seq_mask) / (jnp.sum(seq_mask) + 1e-6) 
        
        ########### vq decoder
        if self._dtype != jnp.float32:
            single_act_fsq = jnp.array(single_act_fsq, dtype=self._dtype)
        single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder(single_act_fsq, seq_mask, residue_index)
        # #### debug 
        # single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder_fn(single_act_normalized, seq_mask, residue_index)
        
        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, normed_single, normed_pair, pLDDT_logits = self.prot_decoder_fn(single_act_decode, pair_act_decode, seq_mask, aatype)
        
        ########### fape loss:
        fape_loss, fape_last_IPA, no_clamp_last_IPA = 0.0, 0.0, 0.0
        if self.fape_loss_weight > 0.0:
            final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor = \
                jax.tree_util.tree_map(jnp.float32, [final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor])
            no_clamp_mask = jax.random.bernoulli(fape_clamp_key, p=0.9, shape=(structure_traj.shape[0], seq_mask.shape[0]))

            fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
                gt_rigid_affine=backbone_affine_tensor, 
                gt_frames_mask=seq_mask, 
                gt_positions_mask=seq_mask,
                target_rigid_affine=structure_traj,
                config=self.train_cfg,
                no_clamp_mask=no_clamp_mask,
                pair_mask=seq_mask[None, :] * seq_mask[:, None],
                IPA_weights=self.fape_IPA_weight,
            )
        
        ########### structure violation loss 
        structure_violation_loss = 0.0
        if self.violation_loss_weight > 0.0:
            final_atom14_positions = jnp.array(final_atom14_positions, dtype=jnp.float32)
            asym_id = jnp.zeros_like(seq_mask, dtype=jnp.int32)
            violation_result_dict = find_structural_violations_array(
                aatype=aatype,
                residue_index=residue_index,
                mask=atom14_atom_exists,
                pred_positions=final_atom14_positions,
                config=self.train_cfg,
                asym_id=asym_id,
            )
            structure_violation_loss = structural_violation_loss(seq_mask, violation_result_dict)
            
        ########### seq length power
        seq_len_weight = jnp.power(jnp.sum(seq_mask), self.seq_len_power)
            
        structure_loss = self.fape_loss_weight * fape_loss + \
                         self.violation_loss_weight * structure_violation_loss
        loss = structure_loss + self.fsq_consistency_loss_weight * fsq_consistency_loss\
                + self.l2_regularizer_loss_weight * l2_regularizer_loss
        
        loss_dict = {
            "loss": loss, 
            "fape_loss": fape_loss,
            "fape_last_IPA": fape_last_IPA,
            "fape_no_clamp_last_IPA": no_clamp_last_IPA,
            "structure_violation_loss": structure_violation_loss,
            "fsq_consistency_loss": fsq_consistency_loss,
            "fsq_l2_regularizer_loss": l2_regularizer_loss,
        }
        
        return loss_dict, code_count, seq_len_weight
    
def pretrain_fsq_vq_decoder_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    weights_sum = jnp.sum(seq_len_weight) + 1e-6
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight)/weights_sum, loss_dict)
    
    # code count: [B, Ncodes]
    code_count = jnp.sum(code_count, axis=0)
    code_usage = jnp.sum(
        jnp.array(code_count > 2, dtype=jnp.float32)) / code_count.shape[-1]
    
    # loss & aux loss valuies
    loss = loss_dict.pop("loss")
    return loss, (loss_dict, code_usage)

def pretrain_fsq_vq_decoder_forward_per_device(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    code_count = jnp.sum(code_count, axis=0) # (B, Ncodes) -> (Ncodes)
    seq_len_weight = seq_len_weight / (jax.lax.psum(jnp.sum(seq_len_weight), axis_name="i") + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    loss = loss_dict.pop("loss")
    
    return loss, (loss_dict, code_count)

###### unfinished class 
class InitializeVQWithLossCell(nn.Module):

    global_config: Config
    train_cfg: Config
    bottleneck_encoder: nn.Module ### with frozen parameters 
    bottleneck_decoder: nn.Module ### with frozen parameters
    encoder_fn: Callable ### params are wrapped with functools.partial
    vq_decoder_fn: Callable ### params are wrapped with functools.partial
    prot_decoder_fn: Callable ### params are wrapped with functools.partial
    fsq_project_indexes_fn: Callable ### params are wrapped with functools.partial
    vq_cfg: Config
    
    def setup(self):
        self.bf16_flag = self.global_config.bf16_flag
        self.safe_precision_flag = self.global_config.safe_precision_flag
        self.dropout_flag = self.global_config.use_dropout

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self._safedtype = jnp.float32 if self.safe_precision_flag else self._dtype
        
        self.project_in = nn.Dense(features=self.vq_cfg.dim_code,
                                   kernel_init=lecun_normal())
        self.project_out = nn.Dense(features=self.vq_cfg.dim_in,
                                    kernel_init=lecun_normal())
        self.codebook = self.param(
            "codebook",
            jax.nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="uniform"),
            (self.vq_cfg.num_code, self.vq_cfg.dim_code))
        
        self.seq_len_power = self.train_cfg.seq_len_power
    
    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,
                 atom14_atom_exists):
        
        ####### generate keys 
        fape_clamp_key = self.make_rng('fape_clamp_key')
        
        ########### encoding
        single_act, inverse_folding_logits = self.encoder_fn(
                                  seq_mask, aatype, residue_index,
                                  template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                                  backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
        
        ########### l2 norm of single act 
        if self._dtype != jnp.float32:
            single_act = jnp.array(single_act, dtype=jnp.float32)
        single_act_normalized = _l2_normalize(single_act, axis=-1, epsilon=self.global_config.norm_small)
        
        ########### bottleneck encoding
        single_act = self.bottleneck_encoder(single_act_normalized, seq_mask, residue_index)
        
        ##### get fsq indexes
        fsq_indexes = self.fsq_project_indexes_fn(single_act)
        
        ########### projection
        single_act = self.project_in(single_act)
        single_act_vq = safe_l2_normalize(single_act_vq, axis=-1, epsilon=self.global_config.norm_small)
        
        ########## infoNCE loss, to be done
        
        loss_dict = {
            "loss": 0.0, 
        }
        
        return loss_dict


class PretrainVQWithLossCell(nn.Module):
    global_config: Config
    train_cfg: Config
    bottleneck_encoder: nn.Module
    bottleneck_decoder: nn.Module
    encoder_fn: Callable ### params are wrapped with functools.partial
    vq_decoder_fn: Callable ### params are wrapped with functools.partial
    prot_decoder_fn: Callable ### params are wrapped with functools.partial
    vq_tokenizer: nn.Module
    vq_cfg: Config
    
    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.safe_precision_flag = self.global_config.safe_precision_flag
        self.dropout_flag = self.global_config.use_dropout

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self._safedtype = jnp.float32 if self.safe_precision_flag else self._dtype
        
        self.project_in = nn.Dense(features=self.vq_cfg.dim_code,
                                   kernel_init=lecun_normal())
        self.project_out = nn.Dense(features=self.vq_cfg.dim_in,
                                    kernel_init=lecun_normal())
        
        ####### loss weights
        self.fape_loss_weight = self.train_cfg.fape.loss_weight
        self.fape_IPA_weight = jnp.array(self.train_cfg.fape.IPA_weight, dtype=jnp.float32)
        self.fape_IPA_weight = self.fape_IPA_weight / jnp.sum(self.fape_IPA_weight)
        self.violation_loss_weight = self.train_cfg.structural_violation.loss_weight
        
        self.vq_consistency_loss_weight = self.train_cfg.vq.consistency_loss_weight
        self.vq_entropy_loss_weight = self.train_cfg.vq.entropy_loss_weight
        self.vq_e_latent_loss_weight = self.train_cfg.vq.e_latent_loss_weight
        self.vq_q_latent_loss_weight = self.train_cfg.vq.q_latent_loss_weight
        
        self.seq_len_power = self.train_cfg.seq_len_power
    
    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,
                 atom14_atom_exists):
        
        ####### generate keys 
        fape_clamp_key = self.make_rng('fape_clamp_key')
        
        ########### encoding
        single_act, inverse_folding_logits = self.encoder_fn(
                                  seq_mask, aatype, residue_index,
                                  template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                                  backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
        
        ########### l2 norm of single act 
        if self._dtype != jnp.float32:
            single_act = jnp.array(single_act, dtype=jnp.float32)
        single_act_normalized = _l2_normalize(single_act, axis=-1, epsilon=self.global_config.norm_small)
        
        ########### projection & quantization
        single_act = self.bottleneck_encoder(single_act_normalized, seq_mask)
        single_act = self.project_in(single_act)
        vq_act, vq_result_dict = self.vq_tokenizer(single_act, seq_mask)
        single_act_vq = self.project_out(vq_act)
        single_act_vq = self.bottleneck_decoder(single_act_vq, seq_mask)
        single_act_vq = safe_l2_normalize(single_act_vq, axis=-1, epsilon=self.global_config.norm_small)
        
        ########## vq consistency loss 
        vq_consistency_loss = 0.0 
        if self.vq_consistency_loss_weight > 0.0:
            vq_consistency_loss = square_euclidean_distance(single_act_normalized, single_act_vq, axis=-1, normalized=True)
            vq_consistency_loss = jnp.sum(vq_consistency_loss * seq_mask) / (jnp.sum(seq_mask) + 1e-6) 
            
        ######### vq aux loss 
        vq_entropy_loss = vq_result_dict["entropy_loss"]
        vq_e_latent_loss = vq_result_dict["e_latent_loss"]
        vq_q_latent_loss = vq_result_dict["q_latent_loss"]
        code_count = vq_result_dict["code_count"]
        
        ########### vq decoder
        if self._dtype != jnp.float32:
            single_act_vq = jnp.array(single_act_vq, dtype=self._dtype)
        single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder_fn(single_act_vq, seq_mask, residue_index)
        
        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, normed_single, normed_pair, pLDDT_logits = self.prot_decoder_fn(single_act_decode, pair_act_decode, seq_mask, aatype)
        
        ########### fape loss:
        final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor = \
            jax.tree_util.tree_map(jnp.float32, [final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor])
        no_clamp_mask = jax.random.bernoulli(fape_clamp_key, p=0.9, shape=(structure_traj.shape[0], seq_mask.shape[0]))

        fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
            gt_rigid_affine=backbone_affine_tensor, 
            gt_frames_mask=seq_mask, 
            gt_positions_mask=seq_mask,
            target_rigid_affine=structure_traj,
            config=self.train_cfg,
            no_clamp_mask=no_clamp_mask,
            pair_mask=seq_mask[None, :] * seq_mask[:, None],
            IPA_weights=self.fape_IPA_weight,
        )
        
        ########### structure violation loss 
        structure_violation_loss = 0.0
        if self.violation_loss_weight > 0.0:
            asym_id = jnp.zeros_like(seq_mask, dtype=jnp.int32)
            violation_result_dict = find_structural_violations_array(
                aatype=aatype,
                residue_index=residue_index,
                mask=atom14_atom_exists,
                pred_positions=final_atom14_positions,
                config=self.train_cfg,
                asym_id=asym_id,
            )
            structure_violation_loss = structural_violation_loss(seq_mask, violation_result_dict)
            
        ########### seq length power
        seq_len_weight = jnp.power(jnp.sum(seq_mask), self.seq_len_power)
            
        structure_loss = self.fape_loss_weight * fape_loss + \
                         self.violation_loss_weight * structure_violation_loss
        loss = structure_loss + self.vq_consistency_loss_weight * vq_consistency_loss + \
               self.vq_entropy_loss_weight * vq_entropy_loss + \
               self.vq_e_latent_loss_weight * vq_e_latent_loss + \
               self.vq_q_latent_loss_weight * vq_q_latent_loss
        
        loss_dict = {
            "loss": loss, 
            "fape_loss": fape_loss,
            "fape_last_IPA": fape_last_IPA,
            "fape_no_clamp_last_IPA": no_clamp_last_IPA,
            "structure_violation_loss": structure_violation_loss,
            "vq_consistency_loss": vq_consistency_loss,
            "vq_entropy_loss": vq_entropy_loss,
            "vq_e_latent_loss": vq_e_latent_loss,
            "vq_q_latent_loss": vq_q_latent_loss,
        }
        
        return loss_dict, code_count, seq_len_weight
    
    
def pretrain_vq_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    weights_sum = jnp.sum(seq_len_weight) + 1e-6
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight)/weights_sum, loss_dict)
    
    # code count: [B, Ncodes]
    code_count = jnp.sum(code_count, axis=0)
    code_usage = jnp.sum(
        jnp.array(code_count > 2, dtype=jnp.float32)) / code_count.shape[-1]
    
    # loss & aux loss valuies
    loss = loss_dict.pop("loss")
    return loss, (loss_dict, code_usage)


def pretrain_vq_forward_per_device(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    code_count = jnp.sum(code_count, axis=0) # (B, Ncodes) -> (Ncodes)
    seq_len_weight = seq_len_weight / (jax.lax.psum(jnp.sum(seq_len_weight), axis_name="i") + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    loss = loss_dict.pop("loss")
    
    return loss, (loss_dict, code_count)


class TrainEncoderFSQDecoderWithLossCell(nn.Module):

    global_config: Config
    train_cfg: Config
    encoder: nn.Module
    fsq_tokenizer: Callable
    fsq_cfg: Config
    vq_decoder: nn.Module
    protein_decoder: nn.Module
    quantize: bool = True
    
    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.safe_precision_flag = self.global_config.safe_precision_flag
        self.dropout_flag = self.global_config.use_dropout
        
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self._safedtype = jnp.float32 if self.safe_precision_flag else self._dtype
        
        self.distogram_loss_func = CA_DistogramLoss(self.train_cfg.distogram)
        self.confidence_loss_func = IntegratedBCEpLDDTLoss(self.train_cfg.confidence)
        
        ####### project_in & project_out
        # self.project_in = nn.Dense(features=len(self.fsq_cfg.levels), 
        #                            kernel_init=truncated_normal(), use_bias=False)
        # self.project_out = nn.Dense(features=self.fsq_cfg.dim_in, 
        #                             kernel_init=truncated_normal(), use_bias=False)
        ####### ms setup 
        self.project_in = nn.Dense(features=len(self.fsq_cfg.levels), 
                                   kernel_init=normal(),
                                   bias_init=normal())
        self.project_out = nn.Dense(features=self.fsq_cfg.dim_in, 
                                    kernel_init=normal(),
                                    bias_init=normal())
        
        ####### loss weights
        self.fape_loss_weight = self.train_cfg.fape.loss_weight
        self.fape_IPA_weight = jnp.array(self.train_cfg.fape.IPA_weight, dtype=jnp.float32)
        self.fape_IPA_weight = self.fape_IPA_weight / jnp.sum(self.fape_IPA_weight)
        self.violation_loss_weight = self.train_cfg.structural_violation.loss_weight
        self.distogram_w1 = self.train_cfg.distogram.w1
        self.distogram_w2 = self.train_cfg.distogram.w2
        self.distogram_w3 = self.train_cfg.distogram.w3
        self.distogram_loss_weight = self.train_cfg.distogram.weight
        self.confidence_loss_weight = self.train_cfg.confidence.loss_weight
        self.inverse_folding_loss_weight = self.train_cfg.inverse_folding.loss_weight
        
        self.l2_regularizer_loss_weight = self.train_cfg.l2_regularizer.loss_weight
        # self.l2_regularizer_threshold = self.train_cfg.l2_regularizer.threshold
        
        ####### for l2 regularizer: 
        self.fsq_act_lower_bound, self.fsq_act_upper_bound = self.fsq_tokenizer.get_act_bound() ##### fp32

        self.seq_len_power = self.train_cfg.seq_len_power

    def __call__(self, seq_mask, true_aatype, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,
                 atom14_atom_exists, dist_gt_perms, dist_mask_perms, perms_padding_mask):
        
        ####### generate keys 
        fape_clamp_key = self.make_rng('fape_clamp_key')
        dmat_rng_key = self.make_rng('dmat_rng_key')
        
        if self.bf16_flag:
            bf16_process_list = [template_all_atom_positions, template_pseudo_beta,
                                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask]

            template_all_atom_positions, template_pseudo_beta, \
            backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask = jax.tree_util.tree_map(lambda x: jnp.bfloat16(x), bf16_process_list)
        
        ########### encoding
        single_act, inverse_folding_logits = self.encoder(seq_mask, aatype, residue_index,
                                  template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                                  backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
        
        ########### inverse folding loss
        inverse_folding_loss = 0.0
        if self.inverse_folding_loss_weight > 0.0:
            inverse_folding_logits = jnp.array(inverse_folding_logits, dtype=jnp.float32)
            true_aatype_onehot = jax.nn.one_hot(true_aatype, num_classes=20)
            inverse_folding_loss = softmax_cross_entropy(inverse_folding_logits, true_aatype_onehot, seq_mask)
            
        ########### fsq tokenize
        single_act = self.project_in(single_act)
        fsq_act = self.fsq_tokenizer(single_act, quantize=self.quantize)
        code_count = self.fsq_tokenizer.count_codes(fsq_act, seq_mask)
        fsq_act = self.project_out(fsq_act)
        
        l2_regularizer_loss = 0.0
        if self.l2_regularizer_loss_weight > 0.0:
            l2_regularizer_loss = \
            jnp.sum(
                seq_mask[..., None] * (
                    nn.relu(single_act - self.fsq_act_upper_bound[None, ...]) + \
                    nn.relu(self.fsq_act_lower_bound[None, ...] - single_act))
                ) / (jnp.sum(seq_mask) * single_act.shape[-1] + 1e-6)
                
        ########### vq decoder
        single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder(fsq_act, seq_mask, residue_index)
        
        ########### distogram loss
        dmat_loss, lddt_loss, contact_loss = 0.0, 0.0, 0.0
        if self.distogram_loss_weight > 0.0:
            dist_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask =\
                jax.tree_util.tree_map(jnp.float32, [dist_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask])
            
            dmat_loss, lddt_loss, contact_loss = self.distogram_loss_func(dist_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask, dmat_rng_key)

        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, normed_single, normed_pair, pLDDT_logits = self.protein_decoder(single_act_decode, pair_act_decode, seq_mask, aatype)
        
        ########### fape loss:
        final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor = \
            jax.tree_util.tree_map(jnp.float32, [final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor])
        no_clamp_mask = jax.random.bernoulli(fape_clamp_key, p=0.9, shape=(structure_traj.shape[0], seq_mask.shape[0]))

        fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
            gt_rigid_affine=backbone_affine_tensor, 
            gt_frames_mask=seq_mask, 
            gt_positions_mask=seq_mask,
            target_rigid_affine=structure_traj,
            config=self.train_cfg,
            no_clamp_mask=no_clamp_mask,
            pair_mask=seq_mask[None, :] * seq_mask[:, None],
            IPA_weights=self.fape_IPA_weight,
        )
        
        ########### structure violation loss 
        structure_violation_loss = 0.0
        if self.violation_loss_weight > 0.0:
            asym_id = jnp.zeros_like(seq_mask, dtype=jnp.int32)
            violation_result_dict = find_structural_violations_array(
                aatype=aatype,
                residue_index=residue_index,
                mask=atom14_atom_exists,
                pred_positions=final_atom14_positions,
                config=self.train_cfg,
                asym_id=asym_id,
            )
            structure_violation_loss = structural_violation_loss(seq_mask, violation_result_dict)
        
        structure_loss = self.fape_loss_weight * fape_loss + \
                         self.violation_loss_weight * structure_violation_loss

        distogram_loss = self.distogram_w1 * dmat_loss + \
                         self.distogram_w2 * contact_loss + \
                         self.distogram_w3 * lddt_loss
        # @ZhangJ.
        reconstruction_loss = structure_loss + \
                              self.distogram_loss_weight * distogram_loss
        
        # @ZhangJ.
        aux_loss = self.inverse_folding_loss_weight * inverse_folding_loss + \
                   self.l2_regularizer_loss_weight * l2_regularizer_loss
        
        ########### confidence loss
        confidence_loss = 0.0
        if self.confidence_loss_weight > 0.0:
            true_lddt = lddt(final_atom14_positions[None, :, 1, :], template_all_atom_positions[None, :, 1, :], seq_mask[None, :, None], per_residue=True)[0] * 100.0 # CA [0, 1] -> [0, 100], lddt is a batched function
            pLDDT_logits = jnp.array(pLDDT_logits, dtype=jnp.float32)
            confidence_loss = self.confidence_loss_func(jax.nn.softmax(pLDDT_logits, axis=-1), true_lddt, seq_mask)
        
        aux_loss += confidence_loss * self.confidence_loss_weight

        ########### seq length power
        seq_len_weight = jnp.power(jnp.sum(seq_mask), self.seq_len_power)
        
        loss = (reconstruction_loss + aux_loss)
        
        loss_dict = {
            "loss": loss,
            "inverse_folding_loss": inverse_folding_loss,
            "dmat_loss": dmat_loss,
            "contact_loss": contact_loss,
            "lddt_loss": lddt_loss,
            "fape_loss": fape_loss,
            "fape_last_IPA": fape_last_IPA,
            "fape_no_clamp_last_IPA": no_clamp_last_IPA,
            "structure_violation_loss": structure_violation_loss,
            "confidence_loss": confidence_loss,
            "l2_regularizer_loss": l2_regularizer_loss
        }
        
        return loss_dict, code_count, seq_len_weight
    
def encoder_fsq_decoder_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
        
    seq_len_weight = seq_len_weight /(jnp.sum(seq_len_weight) + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    
    # code count: [B, Ncodes]
    code_count = jnp.sum(code_count, axis=0)
    code_usage = jnp.sum(
        jnp.array(code_count > 2, dtype=jnp.float32)) / code_count.shape[-1]
    
    # loss & aux loss valuies
    loss = loss_dict.pop("loss")
    return loss, (loss_dict, code_usage)


class JointTrainFSQWithLossCell(nn.Module):

    global_config: Config
    train_cfg: Config
    encoder: nn.Module
    bottleneck_encoder: nn.Module
    fsq_tokenizer: Callable
    fsq_cfg: Config
    vq_cfg: Config
    bottleneck_decoder: nn.Module
    vq_decoder: nn.Module
    protein_decoder: nn.Module
    if_AF2_supervised: bool = True
    quantize: bool = True
    
    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.safe_precision_flag = self.global_config.safe_precision_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self._safedtype = jnp.float32 if self.safe_precision_flag else self._dtype
        
        self.distogram_loss_func = CA_DistogramLoss(self.train_cfg.distogram)
        self.confidence_loss_func = IntegratedBCEpLDDTLoss(self.train_cfg.confidence)
        
        ####### layernorm or l2norm: 
        self.slv_norm_method = self.train_cfg.slv_norm_method
        assert self.slv_norm_method in ["layer_norm", "l2_norm"], \
                "unsupported norm method {}".format(self.slv_norm_method)
        if self.slv_norm_method == "layer_norm":
            #$ self.pre_fsq_slv_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32)
            self.post_fsq_slv_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32)
        
        ####### project_in & project_out
        self.project_in = nn.Dense(features=len(self.fsq_cfg.levels), 
                                   kernel_init=truncated_normal(), use_bias=False)
        self.project_out_1 = nn.Sequential([
            nn.Dense(features=self.vq_cfg.dim_code*4, use_bias=True, kernel_init=truncated_normal()),
            nn.silu,
            nn.Dense(features=self.vq_cfg.dim_code, use_bias=True, kernel_init=truncated_normal())
            ])
        # self.project_out_1 = nn.Dense(features=self.vq_cfg.dim_code, 
        #                               kernel_init=truncated_normal(),  use_bias=False)
        self.project_out_2 = nn.Dense(features=self.fsq_cfg.dim_in, 
                                      kernel_init=truncated_normal(),  use_bias=False)
        
        ####### loss weights (individual)
        self.pre_fsq_cfg = self.train_cfg.pre_fsq
        self.pre_fsq_rel_weight = self.pre_fsq_cfg.loss_weight
        self.pre_fsq_loss_weights = {
            "distogram_loss": self.pre_fsq_cfg.distogram.weight,
            "dmat_loss": self.pre_fsq_cfg.distogram.w1,
            "contact_loss": self.pre_fsq_cfg.distogram.w2,
            "lddt_loss": self.pre_fsq_cfg.distogram.w3,
            "fape_loss": self.pre_fsq_cfg.fape.loss_weight,
            "structure_violation_loss": self.pre_fsq_cfg.structural_violation.loss_weight,
            "confidence_loss": self.pre_fsq_cfg.confidence.loss_weight,
            "AF2_supervision": {
                "single_mse_loss": self.pre_fsq_cfg.AF2_supervised_loss.single_mse_loss_weight,
                "single_cos_sim_loss": self.pre_fsq_cfg.AF2_supervised_loss.single_cos_sim_loss_weight,
                "pair_mse_loss": self.pre_fsq_cfg.AF2_supervised_loss.pair_mse_loss_weight,
                "pair_cos_sim_loss": self.pre_fsq_cfg.AF2_supervised_loss.pair_cos_sim_loss_weight,
            } 
        }
        
        self.post_fsq_cfg = self.train_cfg.post_fsq
        self.post_fsq_loss_weights = {
            "distogram_loss": self.post_fsq_cfg.distogram.weight,
            "dmat_loss": self.post_fsq_cfg.distogram.w1,
            "contact_loss": self.post_fsq_cfg.distogram.w2,
            "lddt_loss": self.post_fsq_cfg.distogram.w3,
            "fape_loss": self.post_fsq_cfg.fape.loss_weight,
            "structure_violation_loss": self.post_fsq_cfg.structural_violation.loss_weight,
            "confidence_loss": self.post_fsq_cfg.confidence.loss_weight,
            "AF2_supervision": {
                "single_mse_loss": self.post_fsq_cfg.AF2_supervised_loss.single_mse_loss_weight,
                "single_cos_sim_loss": self.post_fsq_cfg.AF2_supervised_loss.single_cos_sim_loss_weight,
                "pair_mse_loss": self.post_fsq_cfg.AF2_supervised_loss.pair_mse_loss_weight,
                "pair_cos_sim_loss": self.post_fsq_cfg.AF2_supervised_loss.pair_cos_sim_loss_weight,
            } 
        }
        
        ######## loss weights (shared)
        self.inverse_folding_loss_weight = self.train_cfg.inverse_folding.loss_weight
        self.fape_IPA_weight = jnp.array(self.train_cfg.fape.IPA_weight, dtype=jnp.float32)
        self.fape_IPA_weight = self.fape_IPA_weight / jnp.sum(self.fape_IPA_weight)
        
        ######### fsq loss
        self.l2_regularizer_loss_weight = self.train_cfg.fsq.l2_regularizer_loss_weight
        self.fsq_consistency_loss_weight = self.train_cfg.fsq.consistency_loss_weight
        self.fsq_consistency_loss_prob = self.train_cfg.fsq.consistency_loss_prob
        ####### for l2 regularizer: 
        self.fsq_act_lower_bound, self.fsq_act_upper_bound = self.fsq_tokenizer.get_act_bound() ##### fp32

        self.seq_len_power = self.train_cfg.seq_len_power

        # self.test_norm = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32)

    def __call__(self, seq_mask, true_aatype, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,
                 atom14_atom_exists, dist_gt_perms, dist_mask_perms, perms_padding_mask, AF2_normed_single, AF2_normed_pair, AF2_supervised_mask):
        
        ####### generate keys 
        pre_fsq_fape_clamp_key = self.make_rng('pre_fsq_fape_clamp_key')
        pre_fsq_dmat_rng_key = self.make_rng('pre_fsq_dmat_rng_key')
        post_fsq_fape_clamp_key = self.make_rng('post_fsq_fape_clamp_key')
        post_fsq_dmat_rng_key = self.make_rng('post_fsq_dmat_rng_key')
        fsq_consistency_rng_key = self.make_rng('fsq_consistency_rng_key')
        
        if self.bf16_flag:
            bf16_process_list = [template_all_atom_positions, template_pseudo_beta,
                                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask]

            template_all_atom_positions, template_pseudo_beta, \
            backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask = jax.tree_util.tree_map(lambda x: jnp.bfloat16(x), bf16_process_list)
        
        ########### encoding
        single_act = self.encoder(seq_mask, aatype, residue_index,
                                  template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                                  backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
            
        ########### normalization
        if self.slv_norm_method == "layer_norm":
            # single_act_normalized = self.pre_fsq_slv_layer_norm(single_act)
            single_act_normalized = single_act
        elif self.slv_norm_method == "l2_norm":
            single_act_normalized = safe_l2_normalize(single_act, axis=-1, epsilon=self.norm_small, dtype=single_act.dtype)   
        
        ########### information bottleneck
        single_act, inverse_folding_logits = self.bottleneck_encoder(single_act, seq_mask, residue_index)
        ########### inverse folding loss
        inverse_folding_loss = 0.0
        if self.inverse_folding_loss_weight > 0.0:
            inverse_folding_logits = jnp.array(inverse_folding_logits, dtype=jnp.float32)
            true_aatype_onehot = jax.nn.one_hot(true_aatype, num_classes=20)
            inverse_folding_loss = softmax_cross_entropy(inverse_folding_logits, true_aatype_onehot, seq_mask)
            
        ########### fsq tokenize
        single_act = self.project_in(single_act)
        fsq_act = self.fsq_tokenizer(single_act, quantize=self.quantize)
        code_count = self.fsq_tokenizer.count_codes(fsq_act, seq_mask)
        single_act_fsq = self.project_out_1(fsq_act) ###### project out 1 换成 MLP 3->128->32 激活函数 silu, has_bias True，过完以后做 l2 norm
        # print("single_act_fsq: min max", jnp.min(jnp.abs(single_act_fsq)), jnp.max(jnp.abs(single_act_fsq)))
        # single_act_fsq = self.test_norm(single_act_fsq)
        single_act_fsq = safe_l2_normalize(single_act_fsq, axis=-1, 
                                           dtype=single_act_fsq.dtype, epsilon=self.norm_small)
        single_act_fsq = self.project_out_2(single_act_fsq) # has_bias = False
        single_act_fsq = self.bottleneck_decoder(single_act_fsq, seq_mask, residue_index)
        
        ########### normalization
        if self.slv_norm_method == "layer_norm":
            single_act_fsq_normalized = self.post_fsq_slv_layer_norm(single_act_fsq)
        elif self.slv_norm_method == "l2_norm":
            single_act_fsq_normalized = safe_l2_normalize(single_act_fsq, axis=-1, 
                                                          dtype=single_act_fsq.dtype, epsilon=self.norm_small)
        
        l2_regularizer_loss = 0.0
        if self.l2_regularizer_loss_weight > 0.0:
            l2_regularizer_loss = \
            jnp.sum(
                seq_mask[..., None] * (
                    nn.relu(single_act - self.fsq_act_upper_bound[None, ...]) + \
                    nn.relu(self.fsq_act_lower_bound[None, ...] - single_act))
                ) / (jnp.sum(seq_mask) * single_act.shape[-1] + 1e-6)
            
        ########### consistency loss
        fsq_consistency_loss = 0.0 
        if self.fsq_consistency_loss_weight > 0.0:
            fsq_consistency_loss = square_euclidean_distance(
                jax.lax.stop_gradient(single_act_normalized), 
                single_act_fsq_normalized, 
                axis=-1, 
                normalized=(self.slv_norm_method=="l2_norm")
            )
            ## random mask 
            fsq_consistency_loss_mask = jax.random.bernoulli(
                fsq_consistency_rng_key, 
                p=self.fsq_consistency_loss_prob, 
                shape=fsq_consistency_loss.shape
            )
            fsq_consistency_loss = jnp.sum(
                fsq_consistency_loss * seq_mask * fsq_consistency_loss_mask)\
                    / (jnp.sum(seq_mask * fsq_consistency_loss_mask) + 1e-6)
                    
        ######## prepare for dgram loss
        dist_gt_perms_fp32, dist_mask_perms_fp32, perms_padding_mask_fp32 =\
                jax.tree_util.tree_map(jnp.float32, [dist_gt_perms, dist_mask_perms, perms_padding_mask])
        ######## prepare for structure loss
        backbone_affine_tensor_fp32 = jnp.array(backbone_affine_tensor, dtype=jnp.float32)
        template_all_atom_positions_fp32 = jnp.array(template_all_atom_positions, dtype=jnp.float32)
        ####### prepare for AF2 supervised loss 
        AF2_normed_single_fp32, AF2_normed_pair_fp32, AF2_supervised_mask_fp32 = jax.tree_util.tree_map(jnp.float32, [AF2_normed_single, AF2_normed_pair, AF2_supervised_mask])
            
        def decoder_fn(single_act,
                       fape_clamp_key_, 
                       dmat_rng_key_,):
            single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = self.vq_decoder(single_act, seq_mask, residue_index)
            
            ########### distogram loss
            dist_logits_fp32 = jnp.array(dist_logits, dtype=jnp.float32)
            
            dmat_loss, lddt_loss, contact_loss = \
                self.distogram_loss_func(
                    dist_logits_fp32, 
                    dist_gt_perms_fp32, 
                    dist_mask_perms_fp32, 
                    perms_padding_mask_fp32, 
                    dmat_rng_key_
                )
            
            ########### protein decoder
            final_atom_positions, final_atom14_positions, structure_traj, normed_single, normed_pair, pLDDT_logits = self.protein_decoder(single_act_decode, pair_act_decode, seq_mask, aatype)
            
            ########### fape loss:
            final_atom14_positions_fp32, structure_traj_fp32 = \
                jax.tree_util.tree_map(jnp.float32, [final_atom14_positions, structure_traj])
            no_clamp_mask = jax.random.bernoulli(fape_clamp_key_, p=0.9, shape=(structure_traj.shape[0], seq_mask.shape[0]))

            fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
                gt_rigid_affine=backbone_affine_tensor_fp32, 
                gt_frames_mask=seq_mask, 
                gt_positions_mask=seq_mask,
                target_rigid_affine=structure_traj_fp32,
                config=self.train_cfg,
                no_clamp_mask=no_clamp_mask,
                pair_mask=seq_mask[None, :] * seq_mask[:, None],
                IPA_weights=self.fape_IPA_weight,
            )
            
            ########### structure violation loss 
            asym_id = jnp.zeros_like(seq_mask, dtype=jnp.int32)
            violation_result_dict = find_structural_violations_array(
                aatype=aatype,
                residue_index=residue_index,
                mask=atom14_atom_exists,
                pred_positions=final_atom14_positions_fp32,
                config=self.train_cfg,
                asym_id=asym_id,
            )
            structure_violation_loss = structural_violation_loss(seq_mask, violation_result_dict)
            
            ########### confidence loss
            true_lddt = lddt(
                final_atom14_positions_fp32[None, :, 1, :], 
                template_all_atom_positions_fp32[None, :, 1, :], 
                seq_mask[None, :, None], per_residue=True)[0] * 100.0 # CA [0, 1] -> [0, 100], lddt is a batched function
            pLDDT_logits = jnp.array(pLDDT_logits, dtype=jnp.float32)
            confidence_loss = self.confidence_loss_func(jax.nn.softmax(pLDDT_logits, axis=-1), true_lddt, seq_mask)

            ########### AF2 supervised loss
            single_mse_loss, single_cos_sim_loss, pair_mse_loss, pair_cos_sim_loss = 0.0, 0.0, 0.0, 0.0
            if self.if_AF2_supervised:
                ##### single repr loss
                normed_single_fp32 = jnp.array(normed_single, dtype=jnp.float32)
                single_mse_loss = supervised_single_mse_loss(normed_single_fp32, AF2_normed_single_fp32, seq_mask)
                single_mse_loss = single_mse_loss * AF2_supervised_mask_fp32
                single_cos_sim_loss = supervised_single_cos_sim_loss(normed_single_fp32, AF2_normed_single_fp32, seq_mask)
                single_cos_sim_loss = single_cos_sim_loss * AF2_supervised_mask_fp32
                
                ##### pair repr loss
                normed_pair_fp32 = jnp.array(normed_pair, dtype=jnp.float32)
                pair_mse_loss = supervised_pair_mse_loss(normed_pair_fp32, AF2_normed_pair_fp32, seq_mask)
                pair_mse_loss = pair_mse_loss * AF2_supervised_mask_fp32
                pair_cos_sim_loss = supervised_pair_cos_sim_loss(normed_pair_fp32, AF2_normed_pair_fp32, seq_mask)
                pair_cos_sim_loss = pair_cos_sim_loss * AF2_supervised_mask
                
            loss_dict = {
                "dmat_loss": dmat_loss,
                "contact_loss": contact_loss,
                "lddt_loss": lddt_loss,
                "fape_loss": fape_loss,
                "fape_last_IPA": fape_last_IPA,
                "fape_no_clamp_last_IPA": no_clamp_last_IPA,
                "structure_violation_loss": structure_violation_loss,
                "confidence_loss": confidence_loss,
            }
            AF2_supervision_dict = {
                "single_mse_loss": single_mse_loss,
                "single_cos_sim_loss": single_cos_sim_loss,
                "pair_mse_loss": pair_mse_loss,
                "pair_cos_sim_loss": pair_cos_sim_loss,
            }
            
            return loss_dict, AF2_supervision_dict
            
        loss_dict_pre_fsq_slv, loss_dict_pre_fsq_AF2_supervised = \
                                        decoder_fn(single_act_normalized,
                                                   pre_fsq_fape_clamp_key,
                                                   pre_fsq_dmat_rng_key)
        loss_dict_post_fsq_slv, loss_dict_post_fsq_AF2_supervised = \
                                        decoder_fn(single_act_fsq_normalized,
                                                   post_fsq_fape_clamp_key,
                                                   post_fsq_dmat_rng_key)
        
        ######## Aggregate structure loss
        structure_loss_keys = ["fape_loss", "structure_violation_loss"]
        structure_loss = jnp.sum(jnp.asarray(jax.tree_util.tree_map(
            lambda x: self.pre_fsq_rel_weight * self.pre_fsq_loss_weights[x] * loss_dict_pre_fsq_slv[x] + \
                      self.post_fsq_loss_weights[x] * loss_dict_post_fsq_slv[x], structure_loss_keys
        )))
        
        ######## Aggregate distogram loss
        distogram_loss_keys = ["dmat_loss", "contact_loss", "lddt_loss"]
        pre_fsq_distogram_loss = jnp.sum(jnp.asarray(jax.tree_util.tree_map(
            lambda x: self.pre_fsq_loss_weights[x] * loss_dict_pre_fsq_slv[x], distogram_loss_keys
        ))) * self.pre_fsq_rel_weight
        post_fsq_distogram_loss = jnp.sum(jnp.asarray(jax.tree_util.tree_map(
            lambda x: self.post_fsq_loss_weights[x] * loss_dict_post_fsq_slv[x], distogram_loss_keys
        )))
        
        reconstruction_loss = structure_loss + \
                    self.pre_fsq_loss_weights["distogram_loss"] * pre_fsq_distogram_loss + \
                    self.post_fsq_loss_weights["distogram_loss"] * post_fsq_distogram_loss 
        
        ######## Aggregate aux loss    
        aux_loss = self.inverse_folding_loss_weight * inverse_folding_loss + \
                self.l2_regularizer_loss_weight * l2_regularizer_loss + \
                self.fsq_consistency_loss_weight * fsq_consistency_loss + \
                self.pre_fsq_loss_weights["confidence_loss"] * loss_dict_pre_fsq_slv["confidence_loss"] * self.pre_fsq_rel_weight + \
                self.post_fsq_loss_weights["confidence_loss"] * loss_dict_post_fsq_slv["confidence_loss"]
                
        ####### Aggregate AF2 supervised loss
        AF2_supervised_loss_keys = ["single_mse_loss", "single_cos_sim_loss", "pair_mse_loss", "pair_cos_sim_loss"]
        AF2_supervised_loss = jnp.sum(jnp.asarray(jax.tree_util.tree_map(
            lambda x: self.pre_fsq_loss_weights["AF2_supervision"][x] * \
                      loss_dict_pre_fsq_AF2_supervised[x] * self.pre_fsq_rel_weight + \
                      self.post_fsq_loss_weights["AF2_supervision"][x] * \
                      loss_dict_post_fsq_AF2_supervised[x], AF2_supervised_loss_keys
        )))

        ########### seq length power
        seq_len_weight = jnp.power(jnp.sum(seq_mask), self.seq_len_power)
        
        loss = (reconstruction_loss + aux_loss)
        
        loss_dict = {
            "loss": loss,
            "inverse_folding_loss": inverse_folding_loss,
            "l2_regularizer_loss": l2_regularizer_loss,
            "fsq_consistency_loss": fsq_consistency_loss,
            "pre_fsq": loss_dict_pre_fsq_slv,
            "post_fsq": loss_dict_post_fsq_slv,
            "AF2_supervision": {
                "AF2_supervised_mask": AF2_supervised_mask_fp32,
                "AF2_supervised_loss": AF2_supervised_loss,
                "pre_fsq": loss_dict_pre_fsq_AF2_supervised,
                "post_fsq": loss_dict_post_fsq_AF2_supervised,
            }
        }
        
        return loss_dict, code_count, seq_len_weight

def joint_train_fsq_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    ##### AF2 supervised signal reweight
    AF2_supervision_dict = loss_dict.pop("AF2_supervision")
    AF2_supervised_mask = AF2_supervision_dict.pop("AF2_supervised_mask")
        
    seq_len_weight = seq_len_weight /(jnp.sum(seq_len_weight) + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    
    AF2_supervised_weight = seq_len_weight * AF2_supervised_mask
    AF2_supervised_weight = AF2_supervised_weight / (jnp.sum(AF2_supervised_weight) + 1e-6)
    AF2_supervision_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * AF2_supervised_weight), AF2_supervision_dict)
    
    for k in ["pre_fsq", "post_fsq"]:
        loss_dict[k]['single_mse_loss'] = AF2_supervision_dict[k]['single_mse_loss']
        loss_dict[k]['single_cos_sim_loss'] = AF2_supervision_dict[k]['single_cos_sim_loss']
        loss_dict[k]['pair_mse_loss'] = AF2_supervision_dict[k]['pair_mse_loss']
        loss_dict[k]['pair_cos_sim_loss'] = AF2_supervision_dict[k]['pair_cos_sim_loss']
        
    # code count: [B, Ncodes]
    code_count = jnp.sum(code_count, axis=0)
    code_usage = jnp.sum(
        jnp.array(code_count > 2, dtype=jnp.float32)) / code_count.shape[-1]
    
    # loss & aux loss valuies
    loss = loss_dict.pop("loss")
    loss = loss + AF2_supervision_dict.pop("AF2_supervised_loss")
    return loss, (loss_dict, code_usage)


def joint_train_fsq_forward_per_device(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, code_count, seq_len_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    ##### AF2 supervised signal reweight
    AF2_supervision_dict = loss_dict.pop("AF2_supervision")
    AF2_supervised_mask = AF2_supervision_dict.pop("AF2_supervised_mask")
    AF2_supervised_weight = seq_len_weight * AF2_supervised_mask
    ##### we need a psum here to aggregate the weights
    AF2_supervised_weight = AF2_supervised_weight / \
                        (jax.lax.psum(jnp.sum(AF2_supervised_weight), axis_name="i") + 1e-6)
    AF2_supervision_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * AF2_supervised_weight), AF2_supervision_dict)
    
    # weights_sum = jnp.sum(seq_len_weight) + 1e-6
    seq_len_weight = seq_len_weight / \
                        (jax.lax.psum(jnp.sum(seq_len_weight), axis_name="i") + 1e-6)
    loss_dict = jax.tree_util.tree_map(lambda x: jnp.sum(x * seq_len_weight), loss_dict)
    
    for k in ["pre_fsq", "post_fsq"]:
        loss_dict[k]['single_mse_loss'] = AF2_supervision_dict[k]['single_mse_loss']
        loss_dict[k]['single_cos_sim_loss'] = AF2_supervision_dict[k]['single_cos_sim_loss']
        loss_dict[k]['pair_mse_loss'] = AF2_supervision_dict[k]['pair_mse_loss']
        loss_dict[k]['pair_cos_sim_loss'] = AF2_supervision_dict[k]['pair_cos_sim_loss']
    
    loss = loss_dict.pop("loss")
    loss = loss + AF2_supervision_dict.pop("AF2_supervised_loss")
    
    # code count: [B, Ncodes]
    code_count = jnp.sum(code_count, axis=0)
    
    return loss, (loss_dict, code_count)