import jax
import jax.numpy as jnp

from flax import linen as nn
from common.config_load import Config

import common.residue_constants as residue_constants
from common.geometry import initial_affine, quaternion_to_tensor, pre_compose, vecs_scale,\
     vecs_to_tensor, vecs_expand_dims, rots_expand_dims, quaternion_from_tensor, rots_stop_grad
from common.utils import torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos
from model.equivariant import InvariantPointAttention
from modules.basic import ActFuncWrapper, safe_l2_normalize
from common.config_load import Config

import ml_collections

class MultiRigidSidechain(nn.Module):
    """Class to make side chain atoms."""

    global_config: ml_collections.ConfigDict
    config: Config
    single_repr_dim: int

    def setup(self):

        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        
        # default value setting
        self._dtype = jnp.float32
        
        self.restype_atom14_to_rigid_group = jnp.array(residue_constants.restype_atom14_to_rigid_group, dtype=jnp.int32)
        self.restype_atom14_rigid_group_positions = jnp.array(residue_constants.restype_atom14_rigid_group_positions, dtype=self._dtype)
        self.restype_atom14_mask = jnp.array(residue_constants.restype_atom14_mask, dtype=self._dtype)
        self.restype_rigid_group_default_frame = jnp.array(residue_constants.restype_rigid_group_default_frame, dtype=self._dtype)
        
        # other setting
        self.relu = nn.relu
        
        # default dense setting
        self.input_projection = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(), 
                                         dtype=self._dtype, param_dtype=jnp.float32)
        self.input_projection_1 = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(), 
                                           dtype=self._dtype, param_dtype=jnp.float32)
        self.resblock1 = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(),
                                  dtype=self._dtype, param_dtype=jnp.float32)
        self.resblock2 = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.zeros_init(),
                                  dtype=self._dtype)
        self.resblock1_1 = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(), 
                                    dtype=self._dtype, param_dtype=jnp.float32)
        self.resblock2_1 = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.zeros_init(),
                                    dtype=self._dtype)
        self.unnormalized_angles = nn.Dense(features=14, kernel_init=nn.initializers.lecun_normal(), 
                                            dtype=self._dtype, param_dtype=jnp.float32)
        
        ### debug
        self.mi_grad_fuck_pred_positions_in_mu = self.param(
            'mi_grad_fuck_pred_positions_in_mu', nn.initializers.zeros,
            (256, 14), jnp.float32
        )
        self.mi_grad_fuck_all_frames_to_global = self.param(
            'mi_grad_fuck_all_frames_to_global', nn.initializers.zeros,
            (256,8), jnp.float32
        )

    def __call__(self, rotation, translation, act, initial_act, aatype):
        """Predict side chains using rotation and translation representations.

        Args:
          rotation: The rotation matrices.
          translation: A translation matrices.
          act: updated pair activations from structure module
          initial_act: initial act representations (input of structure module)
          aatype: Amino acid type representations int32

        Returns:
          angles, positions and new frames
        """

        ### Modified by @ZhangJ.
        act1 = self.input_projection(act) # remove relu
        init_act1 = self.input_projection_1(initial_act) # remove relu
        # Sum the activation list (equivalent to concat then Linear).
        act = act1 + init_act1
        # Mapping with some residual blocks.
        # resblock1
        old_act = act
        act = self.resblock1(self.relu(act))
        act = self.resblock2(self.relu(act))
        act += old_act
        # resblock2
        old_act = act
        act = self.resblock1_1(self.relu(act))
        act = self.resblock2_1(self.relu(act))
        act += old_act

        # Map activations to torsion angles. Shape: (num_res, 14).
        num_res = act.shape[0]
        
        unnormalized_angles = self.unnormalized_angles(self.relu(act))
        unnormalized_angles = jnp.reshape(unnormalized_angles, [num_res, 7, 2])
        angles = safe_l2_normalize(unnormalized_angles, axis=-1,
                                   epsilon=self.norm_small) ## @Liyh: changed to safe l2_norm


        backb_to_global = ((rotation[0], rotation[1], rotation[2],
                            rotation[3], rotation[4], rotation[5],
                            rotation[6], rotation[7], rotation[8]),
                           (translation[0], translation[1], translation[2]))
        all_frames_to_global = torsion_angles_to_frames(aatype, backb_to_global, angles,
                                                        self.restype_rigid_group_default_frame)
        all_frames_to_global = jax.tree_util.tree_map(lambda x: x+ self.mi_grad_fuck_all_frames_to_global, all_frames_to_global)
        pred_positions = frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global, ### debug   
                                                                       self.restype_atom14_to_rigid_group,
                                                                       self.restype_atom14_rigid_group_positions,
                                                                       self.restype_atom14_mask) ## need double check @zhenyu
        ### debug
        pred_positions = (pred_positions[0] + self.mi_grad_fuck_pred_positions_in_mu, pred_positions[1], pred_positions[2])
        atom_pos = pred_positions
        frames = all_frames_to_global
        res = (angles, unnormalized_angles, atom_pos, frames)
        return res

class FoldIteration(nn.Module):
    """A single iteration of the main structure module loop."""

    global_config: ml_collections.ConfigDict
    config: Config
    dropout_rate: float = 0.0
    stop_grad: bool = True

    def setup(self):

        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self.remat_flag = self.global_config.remat_flag
        
        # default value setting
        self._dtype = jnp.float32
        self.if_stop_grad = self.stop_grad # Added by @ZhangJ.

        # default layernorm setting
        self.attention_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32) # [384]
        self.transition_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32) # [384]

        # other setting
        self.relu = nn.relu
        self.zeros_like = jnp.zeros_like
        
        # default dense setting
        self.transition = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(), 
                                   dtype=self._dtype, param_dtype=jnp.float32)
        self.transition_1 = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(), 
                                     dtype=self._dtype, param_dtype=jnp.float32)
        self.transition_2 = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(), ### debug
                                     dtype=self._dtype, param_dtype=jnp.float32)
        self.affine_update = nn.Dense(features=6, kernel_init=nn.initializers.lecun_normal(), ### debug
                                      dtype=self._dtype, param_dtype=jnp.float32)
        
        self.drop_out = nn.Dropout(rate=self.dropout_rate, deterministic=(not self.dropout_flag))
        
        # special fuction setting
        attention_module_ = nn.checkpoint(InvariantPointAttention) if self.remat_flag else InvariantPointAttention
        self.attention_module = attention_module_(self.config.num_head, # 12
                                                  self.config.num_scalar_qk, # 16
                                                  self.config.num_scalar_v, # 16
                                                  self.config.num_point_v, # 8
                                                  self.config.num_point_qk, # 4
                                                  self.config.num_channel, # 384
                                                  self.config.pair_channel, # 128
                                                  sink_attention=self.config.sink_attention) # False
        # self.mu_side_chain = BatchMultiRigidSidechain(self.config.sidechain, single_repr_dim)
        self.mu_side_chain = MultiRigidSidechain(self.global_config, self.config.sidechain, self.config.num_channel)
        # if RECOMPUTE_FLAG:
        #     self.attention_module.recompute()
        
        ### debug
        self.mi_grad_fuck_act = self.param(
            'mi_grad_fuck_act', nn.initializers.zeros, 
            (256, self.config.num_channel), jnp.float32
        )

    def __call__(self, act, static_feat_2d, sequence_mask, quaternion, rotation, translation, initial_act, aatype, contextual_mask):
        """construct"""
        # act [Nres, 384]
        # static_feat_2d [Nres, Nres, 192]
        # sequence_mask [Nres, 1]
        # quaternion [Nres, 4]
        # rotation ([Nres,],) * 9
        # translation ([Nres,],) * 3
        # initial_act [Nres, 384]
        # aatype [Nres,]
        # contextual_mask [Nres,]
        
        # print('act:', act.shape)
        # print('static_feat_2d:', static_feat_2d.shape)
        # print('sequence_mask:', sequence_mask.shape)
        # print('rotation:', jax.tree_util.tree_map(jnp.shape, rotation))
        # print('translation:', jax.tree_util.tree_map(jnp.shape, translation))
        ### debug
        act = act + self.mi_grad_fuck_act
        
        attn = self.attention_module(act, static_feat_2d, sequence_mask, rotation, translation) # [Nres, 384]
        # print('attn:', attn.shape)
        act += attn #[Nres, 384]
        act = self.drop_out(act) # @Zhenyu. shut the dropout down currently # [Nres, 384]
        act = self.attention_layer_norm(act) # [Nres, 384]
        
        # Transition
        input_act = act # [Nres, 384]
        act = self.transition(act)
        act = self.relu(act)
        act = self.transition_1(act)
        act = self.relu(act)
        act = self.transition_2(act)

        act += input_act # [Nres, 384]
        act = self.drop_out(act)
        act = self.transition_layer_norm(act) # [Nres, 384]
        
        # debug
        # act = act + self.mi_grad_fuck_act

        # Affine update
        affine_update = self.affine_update(act) # [Nres, 384] -> [Nres, 6]   
        affine_update = (1 - jnp.expand_dims(contextual_mask, -1))*affine_update + jnp.expand_dims(contextual_mask, -1)*self.zeros_like(affine_update, dtype=self._dtype)
        # print('affine_update:', affine_update.shape)
        quaternion_update, rotation_update, translation_update = pre_compose(quaternion, rotation, translation, affine_update)
        quaternion = quaternion_update
        rotation = rotation_update
        translation = translation_update

        translation1 = vecs_scale(translation, 10.0)
        rotation1 = rotation
        
        angles_sin_cos, unnormalized_angles_sin_cos, atom_pos, frames = \
            self.mu_side_chain(rotation1, translation1, act, initial_act, aatype)
            
        affine_output = quaternion_to_tensor(quaternion, translation) # (NRES, 7)
        
        if self.if_stop_grad: # added by @ZhangJ. Modified by @zhenyu
            quaternion = jax.lax.stop_gradient(quaternion)
            rotation = jax.tree_util.tree_map(jax.lax.stop_gradient, rotation)
            # rotation = rots_stop_grad(rotation)
        res = (act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
               atom_pos, frames)
        return res

class StructureModule(nn.Module):
    """
    StructureModule as a network head. Set as Float32.
    IFP Version: Set self.frozen_IPA = True & self.config.if_stop_grad = False & contextual_mask = [1]*seq_length.
    Normal Version: Set self.frozen_IPA = False & self.config.if_stop_grad = True & contextual_mask = [0]*seq_length.
    """ 

    global_config: ml_collections.ConfigDict
    config: Config
    seq_length: int
    frozen_IPA: bool = False
    share_weights: bool = True
    stop_grad_ipa: bool = False
    decoy_affine_init: bool = False
    ret_single_pair: bool = False

    def setup(self):

        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.float32

        # single_repr_dim = self.config.single_channel
        # pair_dim = self.config.pair_channel
        self.num_layer = self.config.num_layer
        self.contextual_mask = jnp.array([0.] * self.seq_length, self._dtype)
        self.traj_w = jnp.array([1.] * 4 + [self.config.position_scale] * 3, self._dtype)

        # fold iteration init
        self.fold_iteration_stack = None
        self.fold_iteration = None
        if not self.share_weights:
            self.fold_iteration_stack = ()
            for nl in range(self.num_layer):
                self.fold_iteration_stack += (FoldIteration(global_config=self.global_config,
                                                            config=self.config,
                                                            dropout_rate=self.config.dropout,
                                                            stop_grad=self.stop_grad_ipa),)
        else:
            self.fold_iteration = FoldIteration(global_config=self.global_config,
                                                config=self.config, 
                                                dropout_rate=self.config.dropout,
                                                stop_grad=self.stop_grad_ipa)

        # default layernorm setting
        self.single_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32)
        self.pair_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32)
        
        # default dense setting
        self.initial_projection = nn.Dense(features=self.config.num_channel, kernel_init=nn.initializers.lecun_normal(), 
                                           dtype=self._dtype, param_dtype=jnp.float32)

        self.mi_grad_affine_output_new = self.param(
            'mi_grad_affine_output_new', nn.initializers.zeros, 
            (1, 256, 7), jnp.float32
        )
        
        ### debug
        self.mi_grad_fuck_act = self.param(
            'mi_grad_fuck_act', nn.initializers.zeros, 
            (256, self.config.num_channel), jnp.float32
        )


    ### Modified by @ZhangJ.
    def __call__(self, single, pair, seq_mask, aatype, decoy_affine_tensor=None):
        # single [Nres, 384]
        # pair [Nres, Nres, 192]
        # seq_mask [Nres,]
        # aatype [Nres,]
        # decoy_affine_tensor [Nres, 7]
        """construct"""

        is_bf16 = self.global_config.bf16_flag 
        if is_bf16:
            (single, pair) = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), (single, pair))
            decoy_affine_tensor = decoy_affine_tensor.astype(jnp.float32) if decoy_affine_tensor is not None else None

        ### debug 
        single = single + self.mi_grad_fuck_act
        
        jax.debug.print("{} {}\n\n", single.shape, jnp.linalg.norm(single, axis=-1))
        sequence_mask = jnp.expand_dims(seq_mask,-1) # [Nres, 1]
        # num_batch = seq_mask.shape[0] ## Liyh: need to check shape
        act = self.single_layer_norm(single) # [Nres, 384]
        
        ##### modified by @linxh, for supvervised loss
        if self.ret_single_pair:
            ret_single = act
        initial_act = act # [Nres, 384]
        act = self.initial_projection(act) # [Nres, 384] -> [Nres, 384]
        act_2d = self.pair_layer_norm(pair) # [Nres, Nres, 192] -> [Nres, Nres, 192]

        ##### modified by @linxh, for supvervised loss
        if self.ret_single_pair:
            ret_pair = act_2d
        
        quaternion, rotation, translation = initial_affine(self.seq_length)
        if self.decoy_affine_init:
            quaternion, rotation, translation = quaternion_from_tensor(decoy_affine_tensor)
            # @ZhangJ. Modified this:
            translation = vecs_scale(translation, 0.1) # Angstrom to nanometer conversion for translation_vecs
            
        quaternion, rotation, translation = jax.tree_util.tree_map(lambda x: self._dtype(x), (quaternion, rotation, translation))
        
        # fold iteration
        atom_pos, affine_output_new, angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, act_iter = \
            self.iteration_operation(act, act_2d, sequence_mask, quaternion, rotation, translation, initial_act, aatype)
        # atom_pos [3, 8 , 220, 14]
        atom14_pred_positions = vecs_to_tensor(atom_pos)[-1, :, :, :] # [seq_len, 14, 3]
        sidechain_atom_pos = atom_pos
        
        # @ZhangJ. 针对Gly 优化以下函数 (确定是否仅需要padding 14 to 37就可以):
        # atom37_pred_positions = batch_atom14_to_atom37(atom14_pred_positions,
        #                                                residx_atom37_to_atom14,
        #                                                atom37_atom_exists,
        #                                                self.indice0)
        
        # !double check this @czy
        # atom37: [N, CA, C, CB, O] ... atom14: [N, CA, C, O, CB ...]
        # atom14_pred_positions: [B, Nres, 14, 3], atom37_pred_positions: [B, Nres, 37, 3]
        
        # @ZhangJ. rewrote:
        atom14_pred_positions_split = jnp.split(atom14_pred_positions, (3,4,5), axis=-2)
        atom37_pred_positions = jnp.concatenate([atom14_pred_positions_split[0], 
                                                 atom14_pred_positions_split[2], 
                                                 atom14_pred_positions_split[1], 
                                                 atom14_pred_positions_split[3]], axis=-2)
        # atom37_pred_positions = ops.Concat(axis=-2)([atom14_pred_positions[..., :3, :], atom14_pred_positions[..., 4:5, :], atom14_pred_positions[..., 3:4, :]])
        atom37_pred_positions = jnp.pad(atom37_pred_positions, [[0, 0], [0, 23], [0, 0]])

        ##### debug
        affine_output_new = affine_output_new + self.mi_grad_affine_output_new
        
        # print('affine_output_new', affine_output_new.shape)
        structure_traj = affine_output_new * self.traj_w
        final_affines = affine_output_new[-1, :, :]
        final_atom_positions = atom37_pred_positions
        rp_structure_module = act_iter
        res = (final_atom_positions, rp_structure_module, atom14_pred_positions, final_affines, \
               angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj)
        
        ##### modified by @linxh, for supvervised loss
        if self.ret_single_pair:
            res = res + (ret_single, ret_pair)

        if is_bf16:
            res = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), res)

        return res
        
    def iteration_operation(self, act, act_2d, sequence_mask, quaternion, rotation, translation, initial_act,
                            aatype):
        affine_init = ()
        angles_sin_cos_init = ()
        um_angles_sin_cos_init = ()
        atom_pos_batch = ()
        frames_batch = ()
        
        if self.frozen_IPA:
            contextual_mask = jnp.squeeze(sequence_mask, -1) # [Nres,]
        else:
            contextual_mask = self.contextual_mask # [Nres,]

        for iter_ in range(1): #### debug
            if not (self.share_weights):
                act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
                atom_pos, frames = \
                    self.fold_iteration_stack[iter_](act, act_2d, sequence_mask, quaternion, rotation, translation, 
                                                     initial_act, aatype, contextual_mask)
            else:
                act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
                atom_pos, frames = \
                    self.fold_iteration(act, act_2d, sequence_mask, quaternion, rotation, translation, 
                                        initial_act, aatype, contextual_mask)
            affine_init = affine_init + (jnp.expand_dims(affine_output, 0),)
            angles_sin_cos_init = angles_sin_cos_init + (angles_sin_cos[None, ...],)
            um_angles_sin_cos_init = um_angles_sin_cos_init + (unnormalized_angles_sin_cos[None, ...],)
            atom_pos_batch += (jnp.concatenate(vecs_expand_dims(atom_pos, 0), axis=0)[:, None, ...],)
            frames_batch += (jnp.concatenate(rots_expand_dims(frames[0], 0) + vecs_expand_dims(frames[1], 0), axis=0)[:, None, ...],)

        affine_output_new = jnp.concatenate(affine_init, axis=0)
        angles_sin_cos_new = jnp.concatenate(angles_sin_cos_init, axis=0)
        um_angles_sin_cos_new = jnp.concatenate(um_angles_sin_cos_init, axis=0)
        frames_new = jnp.concatenate(frames_batch, axis=1)
        atom_pos_new = jnp.concatenate(atom_pos_batch, axis=1)
        
        res = (atom_pos_new, affine_output_new, angles_sin_cos_new, um_angles_sin_cos_new, frames_new, act)
        return res
    
    
class FrameInitializer(nn.Module):
    
    """
    Initialize the frame of the protein.
    """ 
    global_config: ml_collections.ConfigDict
    config: Config # self.fi_config
    seq_length: int #  256
    stop_grad_ipa: bool = False
    share_weights: bool = True
    
    def setup(self):

        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.float32
        
        single_repr_dim = self.config.single_channel # 384
        pair_dim = self.config.pair_channel # 192
        self.num_layer = self.config.num_layer # 1
        self.contextual_mask = jnp.array([0.] * self.seq_length) # [Nres,]

        # fold iteration init
        self.init_iteration = None
        self.init_iteration_stack = None
        if not self.share_weights:
            self.init_iteration_stack = ()
            for nl in range(self.num_layer):
                self.init_iteration_stack += (FoldIteration(global_config=self.global_config,
                                                            config=self.config, # self.fi_config
                                                            dropout_rate=self.config.dropout, # 0.0
                                                            stop_grad=self.stop_grad_ipa),) # False
        else:
            self.init_iteration = FoldIteration(global_config=self.global_config,
                                                config=self.config, 
                                                dropout_rate=self.config.dropout,
                                                stop_grad=self.stop_grad_ipa)

        # default layernorm setting
        self.single_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32) # [384]
        self.pair_layer_norm = nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32) # [192]
        
        # default dense setting
        self.initial_projection = nn.Dense(features=self.config.num_channel, # 384 -> 384
                                           kernel_init=nn.initializers.lecun_normal(), 
                                           dtype=self._dtype, 
                                           param_dtype=jnp.float32)

    
    def __call__(self, single, pair, seq_mask, aatype):
        """construct"""

        is_bf16 = self.global_config.bf16_flag
        if is_bf16:
            (single, pair) = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), (single, pair))

        sequence_mask = jnp.expand_dims(seq_mask,axis=-1) # [NRES, 1]
        
        act = self.single_layer_norm(single) # [NRES, 384]
        initial_act = act # [NRES, 384]
        act = self.initial_projection(act) # [NRES, 384] -> [NRES, 384]
        act_2d = self.pair_layer_norm(pair) # [NRES, NRES, 192] -> [NRES, NRES, 192]
        quaternion, rotation, translation = initial_affine(self.seq_length)
        quaternion, rotation, translation = jax.tree_util.tree_map(lambda x: self._dtype(x), (quaternion, rotation, translation))
   
        ## check quaternion, rotation, translation shape
        # print('quaternion', quaternion.shape)
        # print('rotation', jax.tree_util.tree_map(jnp.shape, rotation))
        # print('translation', jax.tree_util.tree_map(jnp.shape, translation))

        affine_init = ()
        for iter_ in range(self.num_layer): ### debug
            if not (self.share_weights):
                # print('iter_stack_loop1')
                act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
                atom_pos, frames = \
                    self.init_iteration_stack[iter_](act, act_2d, sequence_mask, quaternion, rotation, translation, 
                                                     initial_act, aatype, self.contextual_mask)
                # print('finished iter_stack_loop1')
            else:
                # print('iter_stack_loop2')
                act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
                atom_pos, frames = \
                    self.init_iteration(act, act_2d, sequence_mask, quaternion, rotation, translation, 
                                        initial_act, aatype, self.contextual_mask)
        

            affine_init = affine_init + (affine_output[None, ...],)
            
        affine_output_new = jnp.concatenate(affine_init, axis=1)
        final_affines = affine_output_new[-1, :, :]
        single_act = act

        if is_bf16:
            (affine_output_new, final_affines, single_act) = \
                jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), (affine_output_new, final_affines, single_act))
        
        return affine_output_new, final_affines, single_act