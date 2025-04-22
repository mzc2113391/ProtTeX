import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import jit, vmap

import os
from typing import Callable, Union
from flax.training import common_utils
from loss.fape_loss import backbone_loss_affine_with_weights
from loss.structure_violation_loss import structural_violation_loss, find_structural_violations_array
from modules.basic import safe_l2_normalize
from loss.utils import square_euclidean_distance

from data.utils import get_ppo_angles_sin_cos, make_label_mask_from_extra_mask, pseudo_beta_fn

from ml_collections import ConfigDict
from functools import partial

from data.dataset import protoken_input_feature_names

def post_prior_loss(logits, target_logits):
    target_prob = jax.nn.softmax(target_logits, axis=-1)
    loss = - jnp.sum(target_prob * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    
    return loss

class InferenceCell(nn.Module):
    global_config: ConfigDict
    train_cfg: ConfigDict
    encoder: Union[nn.Module, Callable]
    vq_tokenizer: Union[nn.Module, Callable]
    vq_decoder: Union[nn.Module, Callable]
    protein_decoder: Union[nn.Module, Callable]
    project_in: Union[nn.Module, Callable]
    project_out: Union[nn.Module, Callable]
    quantize: True

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self.seq_len_power = self.train_cfg.seq_len_power
    
    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,):
        
        ####### preprocess features
        if self.bf16_flag:
            bf16_process_list = [template_all_atom_positions, template_pseudo_beta,
                                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask]

            template_all_atom_positions, template_pseudo_beta, \
            backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask = jax.tree_util.tree_map(lambda x: jnp.bfloat16(x), bf16_process_list)
        
        ########### encoding
        # breakpoint()
        single_act, _, _ = self.encoder(seq_mask, aatype, residue_index,
                                        template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                                        backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
            
        ########### vq tokenize (residues are randomly quantized using gumbel & st)
        single_act_project_in = self.project_in(single_act)

        vq_act, quantize_results = \
            self.vq_tokenizer(single_act_project_in,
                              seq_mask,
                              quantize_type='st', entropy_tau=jax.lax.stop_gradient(0.07))
        vq_act = vq_act.astype(self._dtype)

        if not self.quantize:
            vq_act = quantize_results["raw"]

        vq_act_project_out = self.project_out(vq_act)
                
        ########### vq decoder
        single_act_decode, pair_act_decode, _, _ = \
            self.vq_decoder(vq_act_project_out, seq_mask, residue_index)
        
        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, _, _, _ = \
            self.protein_decoder(single_act_decode, pair_act_decode, seq_mask, aatype)
        
        aux_result = {
            "recon_pos": final_atom_positions,
            "code_indices": quantize_results["encoding_indices"],
        }

        return aux_result


class InferenceWithLossCell(nn.Module):
    global_config: ConfigDict
    train_cfg: ConfigDict
    encoder: Union[nn.Module, Callable]
    vq_tokenizer: Union[nn.Module, Callable]
    vq_decoder: Union[nn.Module, Callable]
    protein_decoder: Union[nn.Module, Callable]
    project_in: Union[nn.Module, Callable]
    project_out: Union[nn.Module, Callable]
    quantize: True

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        self.seq_len_power = self.train_cfg.seq_len_power

        ####### loss weights
        self.fape_loss_weight = self.train_cfg.fape.loss_weight
        self.fape_IPA_weight = jnp.array(self.train_cfg.fape.IPA_weight, dtype=jnp.float32)
        self.fape_IPA_weight = self.fape_IPA_weight / jnp.sum(self.fape_IPA_weight)
        self.violation_loss_weight = self.train_cfg.structural_violation.loss_weight
        self.vq_e_latent_loss_weight = self.train_cfg.vq.e_latent_loss_weight
        self.vq_q_latent_loss_weight = self.train_cfg.vq.q_latent_loss_weight
        self.vq_entropy_loss_weight = self.train_cfg.vq.entropy_loss_weight
        self.vq_gumbel_grad_ratio = self.train_cfg.vq.gumbel_grad_ratio
        self.mutual_information_post_1_loss_weight = self.train_cfg.mutual_information.post_1_loss_weight ### post_1_loss: with self-decoding structure, post_2_loss: with adversarial structure
        # self.mutual_information_label_smoothing = self.train_cfg.mutual_information.label_smoothing
        
        #### mutual information loss
        self.learnable_tau = self.param('learnable_tau',
                                        nn.initializers.ones,
                                        (),
                                        jnp.float32)
        self.tau = self.train_cfg.mutual_information.tau
        self.fix_tau = self.train_cfg.mutual_information.fix_tau
        self.tau_scaling_factor = self.train_cfg.mutual_information.tau_scaling_factor
        self.tau_upper_bound = self.train_cfg.mutual_information.tau_upper_bound
        self.tau_lower_bound = self.train_cfg.mutual_information.tau_lower_bound
        self.stop_code_grad_in_post_loss = self.train_cfg.mutual_information.stop_code_grad_in_post_loss

        #### uniformity loss 
        self.uniformity_loss_weight = self.train_cfg.uniformity.loss_weight
        self.uniformity_temperature = self.train_cfg.uniformity.temperature
        
        self.make_label_mask_from_extra_mask_fn = partial(make_label_mask_from_extra_mask, 
                                                          cutoff=self.train_cfg.neighbor.cutoff,
                                                          is_np=False)

        self.seq_len_power = self.train_cfg.seq_len_power
    
    def soft_clamp_tau(self):
        return self.tau_lower_bound + \
            (self.tau_upper_bound - self.tau_lower_bound) * jax.nn.sigmoid(self.learnable_tau)

    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, backbone_affine_tensor_label, 
                 torsion_angles_sin_cos, torsion_angles_mask):
        
        decoder_label_mask = seq_mask
        tau = 0.07
        ####### generate keys 
        fape_clamp_key = self.make_rng('fape_clamp_key')

        ####### preprocess features
        if self.bf16_flag:
            bf16_process_list = [template_all_atom_positions, template_pseudo_beta,
                                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask]

            template_all_atom_positions, template_pseudo_beta, \
            backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask = jax.tree_util.tree_map(lambda x: jnp.bfloat16(x), bf16_process_list)
        
        ########### encoding
        # breakpoint()
        single_act, _, _ = self.encoder(seq_mask, aatype, residue_index,
                                        template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                                        backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
            
        ########### vq tokenize (residues are randomly quantized using gumbel & st)
        single_act_project_in = self.project_in(single_act)
        single_act_project_in_l2_normed = self.vq_tokenizer.l2_normalize(single_act_project_in, seq_mask)

        vq_act, quantize_results = \
            self.vq_tokenizer(single_act_project_in,
                              seq_mask,
                              quantize_type='st', entropy_tau=jax.lax.stop_gradient(0.07))
        vq_act = vq_act.astype(self._dtype)

        code_count = quantize_results.pop("code_count")
        vq_entropy_loss = quantize_results.pop("entropy_loss") * jnp.sum(seq_mask, axis=-1)
        vq_entropy_loss = jnp.sum(vq_entropy_loss) / (jnp.sum(seq_mask, axis=-1) + 1e-6)
        
        if not self.quantize:
            vq_act = quantize_results["raw"]
        quantize_results["code_count"] = code_count
        quantize_results["entropy_loss"] = vq_entropy_loss
        vq_distances = quantize_results["distances"] if not self.stop_code_grad_in_post_loss else \
                        quantize_results["distances_sg"]
        vq_act_project_out = self.project_out(vq_act)
                
        ########### vq decoder
        single_act_decode, pair_act_decode, _, _ = \
            self.vq_decoder(vq_act_project_out, seq_mask, residue_index)
        
        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, _, _, _ = \
            self.protein_decoder(single_act_decode, pair_act_decode, seq_mask, aatype)
        
        aux_result = {
            "recon_pos": final_atom_positions,
            "code_indices": quantize_results["encoding_indices"],
            "code_count": quantize_results["code_count"],
        }

        ############## loss calculations
        ########### mutual_information_post_1_loss
        mutual_information_post_1_loss = 0.0
        if self.mutual_information_post_1_loss_weight > 0.0:
            ########### another round of encoding
            torsion_angles_sin_cos_decode = \
                get_ppo_angles_sin_cos(jax.lax.stop_gradient(final_atom_positions), 
                                       decoder_label_mask)
            template_pseudo_beta_decode = \
                pseudo_beta_fn(aatype, 
                               jax.lax.stop_gradient(final_atom_positions), all_atom_mask=None)
            backbone_affine_tensor_decode = jax.lax.stop_gradient(structure_traj[-1])
            
            single_act_decode, _, _ = \
                self.encoder(decoder_label_mask, aatype, residue_index,
                             template_all_atom_masks, jax.lax.stop_gradient(final_atom_positions), template_pseudo_beta_decode,
                             backbone_affine_tensor_decode, torsion_angles_sin_cos_decode, torsion_angles_mask)
            single_act_project_in_decode = self.project_in(single_act_decode)
            #### l2-norm (or not)
            single_act_project_in_decode = self.vq_tokenizer.l2_normalize(single_act_project_in_decode, decoder_label_mask)
            
            distance_logits = - self.vq_tokenizer.distance(single_act_project_in_decode, 
                                                           decoder_label_mask,
                                                           self.stop_code_grad_in_post_loss) / tau
            
            ### since we have stop gradient in final_atom_positions & structure traj, no need
            # ##### mask decoder label mask neighbors, to avoid nan in gradient of sin/cos
            # decoder_label_mask_shift_right = jnp.concatenate([decoder_label_mask[1:], jnp.ones(1, dtype=decoder_label_mask.dtype)], axis=-1)
            # decoder_label_mask_shift_left = jnp.concatenate([jnp.ones(1, dtype=decoder_label_mask.dtype), decoder_label_mask[:-1]], axis=-1)
            # decoder_label_mask = jnp.logical_and(decoder_label_mask, decoder_label_mask_shift_left)
            # decoder_label_mask = jnp.logical_and(decoder_label_mask, decoder_label_mask_shift_right)                    
            
            mutual_information_post_1_loss = post_prior_loss(
                logits = distance_logits, 
                target_logits = jax.lax.stop_gradient(-vq_distances / tau * self.tau_scaling_factor)
            )
            mutual_information_post_1_loss = \
                jnp.sum(mutual_information_post_1_loss * seq_mask, axis=-1) / (jnp.sum(seq_mask, axis=-1) + 1e-6)
        
        final_atom_positions, final_atom14_positions, structure_traj, backbone_affine_tensor_label = \
            jax.tree_util.tree_map(jnp.float32, [final_atom_positions, final_atom14_positions, 
                                                 structure_traj, backbone_affine_tensor_label])
        ########### fape loss:
        no_clamp_mask = jax.random.bernoulli(fape_clamp_key, p=0.9, 
                                             shape=(structure_traj.shape[0], seq_mask.shape[0]))
        
        structure_traj = structure_traj
        fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
            gt_rigid_affine=backbone_affine_tensor_label, 
            gt_frames_mask=decoder_label_mask, 
            gt_positions_mask=decoder_label_mask,
            target_rigid_affine=structure_traj,
            config=self.train_cfg,
            no_clamp_mask=no_clamp_mask,
            pair_mask=decoder_label_mask[None, :] * decoder_label_mask[:, None],
            IPA_weights=self.fape_IPA_weight,
        )
        
        ########### structure violation loss 
        # structure_violation_loss = 0.0
        # if self.violation_loss_weight > 0.0:
        #     asym_id = jnp.zeros_like(decoder_label_mask, dtype=jnp.int32)
        #     violation_result_dict = find_structural_violations_array(
        #         aatype=aatype,
        #         residue_index=residue_index,
        #         mask=atom14_atom_exists * decoder_label_mask[..., None],
        #         pred_positions=final_atom14_positions,
        #         config=self.train_cfg,
        #         asym_id=asym_id,
        #     )
        #     structure_violation_loss = structural_violation_loss(decoder_label_mask, violation_result_dict)
        
        structure_loss = self.fape_loss_weight * fape_loss # + \
                         # self.violation_loss_weight * structure_violation_loss

        reconstruction_loss = structure_loss

        ########### vq loss 
        vq_loss = 0.0 
        if self.quantize and self.vq_e_latent_loss_weight > 0.0:
            vq_loss += self.vq_e_latent_loss_weight * quantize_results["e_latent_loss"]
        if self.quantize and self.vq_q_latent_loss_weight > 0.0:
            vq_loss += self.vq_q_latent_loss_weight * quantize_results["q_latent_loss"]


        ########### uniformity loss
        uniformity_loss = 0.0
        if self.uniformity_loss_weight > 0.0:
            single_act_project_in_l2_normed = single_act_project_in_l2_normed.astype(jnp.float32)
            distance_matrix = square_euclidean_distance(
                jnp.expand_dims(single_act_project_in_l2_normed, -2), jnp.expand_dims(single_act_project_in_l2_normed, -3), axis=-1, normalized=self.vq_tokenizer.config.l2_norm,
            ) # (Ncode, Ncode)
            uniformity_gaussian_potential = jnp.exp(
                -self.uniformity_temperature * distance_matrix
            )
            seq_mask_2d = seq_mask[None, :].astype(jnp.float32) * seq_mask[:, None].astype(jnp.float32)
            uniformity_gaussian_potential = \
                jnp.sum(uniformity_gaussian_potential * seq_mask_2d) / (jnp.sum(seq_mask_2d) + 1e-6)
            uniformity_loss = jnp.log(
                jnp.clip(uniformity_gaussian_potential, 1e-8, jnp.inf)) * 0.5
            
        aux_loss = vq_loss + \
            self.mutual_information_post_1_loss_weight * mutual_information_post_1_loss + \
            self.vq_entropy_loss_weight * quantize_results['entropy_loss'] + \
            self.uniformity_loss_weight * uniformity_loss

        ########### seq length power
        seq_len_weight = jnp.power(jnp.sum(seq_mask), self.seq_len_power)
        loss = reconstruction_loss + aux_loss
        loss_dict = {
            "loss": loss,
            "fape_loss": fape_loss,
            "fape_last_IPA": fape_last_IPA,
            "fape_no_clamp_last_IPA": no_clamp_last_IPA,
            "vq_e_latent_loss": quantize_results["e_latent_loss"] if self.quantize else 0.0,
            "vq_q_latent_loss": quantize_results["q_latent_loss"] if self.quantize else 0.0,
            "vq_entropy_loss": quantize_results['entropy_loss'] if self.quantize else 0.0,
            "mutual_information_post_1_loss": mutual_information_post_1_loss,
            "embedding_uniformity_loss": uniformity_loss,
        }

        return loss_dict, aux_result, seq_len_weight


class DecoderCell(nn.Module):
    global_config: ConfigDict
    vq_tokenizer: Union[nn.Module, Callable]
    vq_decoder: Union[nn.Module, Callable]
    protein_decoder: Union[nn.Module, Callable]
    project_out: Union[nn.Module, Callable]
    quantize: True

    def setup(self):
        self.bf16_flag = self.global_config.bf16_flag
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
    
    def __call__(self, seq_mask, aatype, residue_index, code_indices,):
        
        ####### preprocess features
        vq_act = self.vq_tokenizer.decode_ids(code_indices)
        vq_act = vq_act.astype(self._dtype)

        vq_act_project_out = self.project_out(vq_act)
                
        ########### vq decoder
        single_act_decode, pair_act_decode, _, _ = \
            self.vq_decoder(vq_act_project_out, seq_mask, residue_index)
        
        ########### protein decoder
        final_atom_positions, final_atom14_positions, structure_traj, _, _, _ = \
            self.protein_decoder(single_act_decode, pair_act_decode, seq_mask, aatype)
        
        ########### results
        aux_result = {
            "recon_pos": final_atom_positions,
            "single_act_decode": single_act_decode,
            "code_indices": code_indices,
        }

        return aux_result


class EncoderCell(nn.Module):
    global_config: ConfigDict
    encoder: Union[nn.Module, Callable]
    vq_tokenizer: Union[nn.Module, Callable]
    project_in: Union[nn.Module, Callable]
    quantize: True

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

    def __call__(self, seq_mask, aatype, residue_index,
                 template_all_atom_masks, template_all_atom_positions, template_pseudo_beta, 
                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask,):
        
        ####### preprocess features
        if self.bf16_flag:
            bf16_process_list = [template_all_atom_positions, template_pseudo_beta,
                                 backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask]

            template_all_atom_positions, template_pseudo_beta, \
            backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask = jax.tree_util.tree_map(lambda x: jnp.bfloat16(x), bf16_process_list)
        
        ########### encoding
        # breakpoint()
        single_act, _, _ = self.encoder(seq_mask, aatype, residue_index,
                                        template_all_atom_masks, template_all_atom_positions, template_pseudo_beta,
                                        backbone_affine_tensor, torsion_angles_sin_cos, torsion_angles_mask)
            
        ########### vq tokenize (residues are randomly quantized using gumbel & st)
        single_act_project_in = self.project_in(single_act)

        vq_act, quantize_results = \
            self.vq_tokenizer(single_act_project_in,
                              seq_mask,
                              quantize_type='st', entropy_tau=jax.lax.stop_gradient(0.07))
        vq_act = vq_act.astype(self._dtype)

        if not self.quantize:
            vq_act = quantize_results["raw"]
        
        aux_result = {
            "code_indices": quantize_results["encoding_indices"],
        }
        return aux_result