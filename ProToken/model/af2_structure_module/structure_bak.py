# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules and utilities for the structure module in the multimer system."""

import functools
import numbers
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from common import residue_constants
from model import geometry
from model import common_modules_bak
from model import all_atom_bak
from model.geometry import utils as geometry_utils
# import haiku as hk
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from common.config_load import Config

EPSILON = 1e-8
Float = Union[float, jnp.ndarray]

def squared_difference(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes Squared difference between two arrays."""
    return jnp.square(x - y)

def make_backbone_affine(
    positions: geometry.Vec3Array,
    mask: jnp.ndarray,
    aatype: jnp.ndarray,
    ) -> Tuple[geometry.Rigid3Array, jnp.ndarray]:
    """Make backbone Rigid3Array and mask."""
    del aatype
    a = residue_constants.atom_order['N']
    b = residue_constants.atom_order['CA']
    c = residue_constants.atom_order['C']

    rigid_mask = (mask[:, a] * mask[:, b] * mask[:, c]).astype(
        jnp.float32)

    rigid = all_atom_bak.make_transform_from_reference(
        a_xyz=positions[:, a], b_xyz=positions[:, b], c_xyz=positions[:, c])

    return rigid, rigid_mask


class QuatRigid(nn.Module):
    """Module for projecting Rigids via a quaternion."""
    
    """Module projecting a Rigid Object.

    For this Module the Rotation is parametrized as a quaternion,
    If 'full_quat' is True a 4 vector is produced for the rotation which is
    normalized and treated as a quaternion.
    When 'full_quat' is False a 3 vector is produced and the 1st component of
    the quaternion is set to 1.

    Args:
        global_config: Global Config, used to set certain properties of underlying
        Linear module, see common_modules.Linear for details.
        rigid_shape: Shape of Rigids relative to shape of activations, e.g. when
        activations have shape (n,) and this is (m,) output will be (n, m)
        full_quat: Whether to parametrize rotation using full quaternion.
        init: initializer to use, see common_modules.Linear for details
        name: Name to use for module.
    """
    rigid_shape: Union[int, Iterable[int]] = tuple()
    full_quat: bool = False
    initializer: str = 'zeros'
    name: str = 'quat_rigid'

    @nn.compact
    def __call__(self, activations: jnp.ndarray) -> geometry.Rigid3Array:
        """Executes Module.

        This returns a set of rigid with the same shape as activations, projecting
        the channel dimension, rigid_shape controls the trailing dimensions.
        For example when activations is shape (12, 5) and rigid_shape is (3, 2)
        then the shape of the output rigids will be (12, 3, 2).
        This also supports passing in an empty tuple for rigid shape, in that case
        the example would produce a rigid of shape (12,).

        Args:
            activations: Activations to use for projection, shape [..., num_channel]
        Returns:
            Rigid transformations with shape [...] + rigid_shape
        """
        
        ## initialize
        if isinstance(self.rigid_shape, int):
            rigid_shape = (self.rigid_shape,)
        else:
            rigid_shape = tuple(self.rigid_shape)
        
        ## calculate
        if self.full_quat:
            rigid_dim = 7
        else:
            rigid_dim = 6
        linear_dims = rigid_shape + (rigid_dim,)
        rigid_flat = common_modules_bak.Linear(
            linear_dims,
            initializer=self.initializer,
            precision=jax.lax.Precision.HIGHEST,
            name='rigid')(
                activations)
        rigid_flat = geometry_utils.unstack(rigid_flat)
        if self.full_quat:
            qw, qx, qy, qz = rigid_flat[:4]
            translation = rigid_flat[4:]
        else:
            qx, qy, qz = rigid_flat[:3]
            qw = jnp.ones_like(qx)
            translation = rigid_flat[3:]
        rotation = geometry.Rot3Array.from_quaternion(
            qw, qx, qy, qz, normalize=True)
        translation = geometry.Vec3Array(*translation)
        return geometry.Rigid3Array(rotation, translation)
    
    
class PointProjection(nn.Module):
    """Given input reprensentation and frame produces points in global frame."""
    
    """Constructs Linear Module.

    Args:
      num_points: number of points to project. Can be tuple when outputting
          multiple dimensions
      global_config: Global Config, passed through to underlying Linear
      return_local_points: Whether to return points in local frame as well.
      name: name of module, used for name scopes.
    """
    num_points: Union[Iterable[int], int]
    return_local_points: bool = False
    name: str = 'point_projection'
    
    @nn.compact
    def __call__(
        self, activations: jnp.ndarray, rigids: geometry.Rigid3Array
    ) -> Union[geometry.Vec3Array, Tuple[geometry.Vec3Array, geometry.Vec3Array]]:
        ### Initialize
        if isinstance(self.num_points, numbers.Integral):
            num_points = (self.num_points,)
        else:
            num_points = tuple(self.num_points)
            
        output_shape = num_points
        output_shape = output_shape[:-1] + (3 * output_shape[-1],)
        points_local = common_modules_bak.Linear(
            output_shape,
            precision=jax.lax.Precision.HIGHEST,
            name='point_projection')(
                activations)
        points_local = jnp.split(points_local, 3, axis=-1)
        points_local = geometry.Vec3Array(*points_local)
        rigids = rigids[(...,) + (None,) * len(output_shape)]
        points_global = rigids.apply_to_point(points_local)
        if self.return_local_points:
            return points_global, points_local
        else:
            return points_global
        

class InvariantPointAttention(nn.Module):
    """Invariant point attention module.

    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).

    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.
    """
    
    """Initialize.

    Args:
        config: iterative Fold Head Config
        global_config: Global Config of Model.
        dist_epsilon: Small value to avoid NaN in distance calculation.
        name: Sonnet name.
    """
    
    config: Config
    global_config: Config
    dist_epsilon: float = 1e-8
    name: str = 'invariant_point_attention'
    
    @nn.compact
    def __call__(self,
        inputs_1d: jnp.ndarray,
        inputs_2d: jnp.ndarray,
        mask: jnp.ndarray,
        rigid: geometry.Rigid3Array,
    ) -> jnp.ndarray:
        """Compute geometric aware attention.

        Given a set of query residues (defined by affines and associated scalar
        features), this function computes geometric aware attention between the
        query residues and target residues.

        The residues produce points in their local reference frame, which
        are converted into the global frame to get attention via euclidean distance.

        Equivalently the target residues produce points in their local frame to be
        used as attention values, which are converted into the query residues local
        frames.

        Args:
        inputs_1d: (N, C) 1D input embedding that is the basis for the
            scalar queries.
        inputs_2d: (N, M, C') 2D input embedding, used for biases values in the
            attention between query_inputs_1d and target_inputs_1d.
        mask: (N, 1) mask to indicate query_inputs_1d that participate in
            the attention.
        rigid: Rigid object describing the position and orientation of
            every element in query_inputs_1d.

        Returns:
        Transformation of the input embedding.
        """
        
        ### initialize
        _dist_epsilon = self.dist_epsilon
        _zero_initialize_last = self.global_config.zero_init
        
        num_head = self.config.num_head

        attn_logits = 0.

        num_point_qk = self.config.num_point_qk
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2
        point_weights = np.sqrt(1.0 / point_variance)

        # This is equivalent to jax.nn.softplus, but avoids a bug in the test...
        softplus = lambda x: jnp.logaddexp(x, jnp.zeros_like(x))
        raw_point_weights = self.param(
            'trainable_point_weights',
            nn.initializers.constant(np.log(np.exp(1.) - 1.)),
            [num_head],
            jnp.float32
            # softplus^{-1} (1)
        )

        # Trainable per-head weights for points.
        trainable_point_weights = softplus(raw_point_weights)
        point_weights *= trainable_point_weights
        q_point = PointProjection([num_head, num_point_qk],
                                name='q_point_projection')(inputs_1d,
                                                            rigid)

        k_point = PointProjection([num_head, num_point_qk],
                                name='k_point_projection')(inputs_1d,
                                                            rigid)

        dist2 = geometry.square_euclidean_distance(
            q_point[:, None, :, :], k_point[None, :, :, :], epsilon=0.)
        attn_qk_point = -0.5 * jnp.sum(point_weights[:, None] * dist2, axis=-1)
        attn_logits += attn_qk_point

        num_scalar_qk = self.config.num_scalar_qk
        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        scalar_weights = np.sqrt(1.0 / scalar_variance)
        q_scalar = common_modules_bak.Linear([num_head, num_scalar_qk],
                                        use_bias=False,
                                        name='q_scalar_projection')(
                                            inputs_1d)

        k_scalar = common_modules_bak.Linear([num_head, num_scalar_qk],
                                        use_bias=False,
                                        name='k_scalar_projection')(
                                            inputs_1d)
        q_scalar *= scalar_weights
        attn_logits += jnp.einsum('qhc,khc->qkh', q_scalar, k_scalar)

        attention_2d = common_modules_bak.Linear(
            num_head, name='attention_2d')(inputs_2d)
        attn_logits += attention_2d

        mask_2d = mask * jnp.swapaxes(mask, -1, -2)
        attn_logits -= 1e5 * (1. - mask_2d[..., None])

        attn_logits *= np.sqrt(1. / 3)     # Normalize by number of logit terms (3)
        attn = jax.nn.softmax(attn_logits, axis=-2)

        num_scalar_v = self.config.num_scalar_v

        v_scalar = common_modules_bak.Linear([num_head, num_scalar_v],
                                        use_bias=False,
                                        name='v_scalar_projection')(
                                            inputs_1d)

        # [num_query_residues, num_head, num_scalar_v]
        result_scalar = jnp.einsum('qkh, khc->qhc', attn, v_scalar)

        num_point_v = self.config.num_point_v
        v_point = PointProjection([num_head, num_point_v],
                                self.global_config,
                                name='v_point_projection')(inputs_1d,
                                                            rigid)
        
        result_point_global = jax.tree_util.tree_map(
            lambda x: jnp.sum(attn[..., None] * x, axis=-3), v_point[None])

        # Features used in the linear output projection. Should have the size
        # [num_query_residues, ?]
        output_features = []
        num_query_residues, _ = inputs_1d.shape

        flat_shape = [num_query_residues, -1]

        result_scalar = jnp.reshape(result_scalar, flat_shape)
        output_features.append(result_scalar)

        result_point_global = jax.tree_util.tree_map(lambda r: jnp.reshape(r, flat_shape),
                                        result_point_global)
        result_point_local = rigid[..., None].apply_inverse_to_point(
            result_point_global)
        output_features.extend(
            [result_point_local.x, result_point_local.y, result_point_local.z])

        point_norms = result_point_local.norm(_dist_epsilon)
        output_features.append(point_norms)

        # Dimensions: h = heads, i and j = residues,
        # c = inputs_2d channels
        # Contraction happens over the second residue dimension, similarly to how
        # the usual attention is performed.
        result_attention_over_2d = jnp.einsum('ijh, ijc->ihc', attn, inputs_2d)
        output_features.append(jnp.reshape(result_attention_over_2d, flat_shape))

        final_init = 'zeros' if _zero_initialize_last else 'linear'

        final_act = jnp.concatenate(output_features, axis=-1)

        return common_modules_bak.Linear(
            self.config.num_channel,
            initializer=final_init,
            name='output_projection')(final_act)
        
        
class FoldIteration(nn.Module):
    """A single iteration of iterative folding.

    First, each residue attends to all residues using InvariantPointAttention.
    Then, we apply transition layers to update the hidden representations.
    Finally, we use the hidden representations to produce an update to the
    affine of each residue.
    """
    config: Config
    global_config: Config
    name: str = 'fold_iteration'
    
    @nn.compact
    def __call__(
        self,
        activations: Mapping[str, Any],
        aatype: jnp.ndarray,
        sequence_mask: jnp.ndarray,
        update_rigid: bool,
        initial_act: jnp.ndarray,
        static_feat_2d: Optional[jnp.ndarray] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        c = self.config
        
        rigid = activations['rigid']
        act = activations['act']
        attention_module = InvariantPointAttention(
            self.config, self.global_config)
        # Attention
        act += attention_module(
            inputs_1d=act,
            inputs_2d=static_feat_2d,
            mask=sequence_mask,
            rigid=rigid)
        
        act = common_modules_bak.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        name='attention_layer_norm')(
            act)
        final_init = 'zeros' if self.global_config.zero_init else 'linear'
        
        # Transition
        input_act = act
        for i in range(c.num_layer_in_transition):
            init = 'relu' if i < c.num_layer_in_transition - 1 else final_init
            act = common_modules_bak.Linear(
                c.num_channel,
                initializer=init,
                name='transition')(
                    act)
            if i < c.num_layer_in_transition - 1:
                act = jax.nn.relu(act)
        act += input_act
        act = common_modules_bak.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name='transition_layer_norm')(act)
        if update_rigid:
            # Rigid update
            rigid_update = QuatRigid(
                self.global_config, init=final_init)(
                    act)
            rigid = rigid @ rigid_update
            
        sc = MultiRigidSidechain(c.sidechain, self.global_config)(
        rigid.scale_translation(c.position_scale), [act, initial_act], aatype)

        outputs = {'rigid': rigid, 'sc': sc}

        rotation = rigid.rotation #jax.tree_util.tree_map(jax.lax.stop_gradient, rigid.rotation)
        rigid = geometry.Rigid3Array(rotation, rigid.translation)

        new_activations = {
            'act': act,
            'rigid': rigid
        }
        return new_activations, outputs
    
    
def generate_monomer_rigids(representations: Mapping[str, jnp.ndarray],
                            batch: Mapping[str, jnp.ndarray],
                            config: Config,
                            global_config: Config,
                            # safe_key: prng.SafeKey
                            ) -> Dict[str, Any]:
    """Generate predicted Rigid's for a single chain.

    This is the main part of the iterative fold head - it iteratively applies
    folding to produce a set of predicted residue positions.

    Args:
        representations: Embeddings dictionary.
        batch: Batch dictionary.
        config: config for the iterative fold head.
        global_config: global config.
        safe_key: A prng.SafeKey object that wraps a PRNG key.

    Returns:
        A dictionary containing residue Rigid's and sidechain positions.
    """
    c = config
    sequence_mask = batch['seq_mask'][:, None]
    act = common_modules_bak.LayerNorm(
        axis=-1, create_scale=True, create_offset=True, name='single_layer_norm')(
            representations['single'])

    initial_act = act
    act = common_modules_bak.Linear(
        c.num_channel, name='initial_projection')(act)

    if "initial_atom_pos" in batch:
        atom = residue_constants.atom_order
        atom_pos = batch["initial_atom_pos"]
        if global_config.bfloat16: atom_pos = atom_pos.astype(jnp.float32)
        atom_pos = geometry.Vec3Array.from_array(atom_pos)
        rigid = all_atom_bak.make_transform_from_reference(
        a_xyz=atom_pos[:, atom["N"]],
        b_xyz=atom_pos[:, atom["CA"]],
        c_xyz=atom_pos[:, atom["C"]]).scale_translation(1/c.position_scale)

    else:
        # Sequence Mask has extra 1 at the end.
        rigid = geometry.Rigid3Array.identity(sequence_mask.shape[:-1])

    fold_iteration = FoldIteration(c, global_config, name='fold_iteration')

    assert len(batch['seq_mask'].shape) == 1

    activations = {
        'act':
            act,
        'rigid':
            rigid
    }

    act_2d = common_modules_bak.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        name='pair_layer_norm')(
            representations['pair'])

    def fold_iter(act, _):
        act, out = fold_iteration(
            act,
            initial_act=initial_act,
            static_feat_2d=act_2d,
            aatype=batch['aatype'],
            sequence_mask=sequence_mask,
            update_rigid=True
            )
        return act, out
    
    _s = [None] * c.num_layer

    # keys = jax.random.split(safe_key.get(), c.num_layer)
    activations, output = nn.scan(fold_iter)(activations, _s)
    output['act'] = activations['act']
    return output

class StructureModule(nn.Module):
    """StructureModule as a network head.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
    """
    config: Config
    global_config: Config
    
    @nn.compact
    def __call__(self,
                representations: Mapping[str, jnp.ndarray],
                batch: Mapping[str, Any],
                # safe_key: Optional[prng.SafeKey] = None,
               ) -> Dict[str, Any]:
        c = self.config
        ret = {}
        
        output = generate_monomer_rigids(
            representations=representations,
            batch=batch,
            config=self.config,
            global_config=self.global_config,
            #safe_key=safe_key
        )
        
        
        ret['traj'] = output['rigid'].scale_translation(c.position_scale).to_array()
        ret['sidechains'] = output['sc']
        ret['sidechains']['atom_pos'] = ret['sidechains']['atom_pos'].to_array()
        ret['sidechains']['frames'] = ret['sidechains']['frames'].to_array()
        if 'local_atom_pos' in ret['sidechains']:
            ret['sidechains']['local_atom_pos'] = ret['sidechains'][
                'local_atom_pos'].to_array()
            ret['sidechains']['local_frames'] = ret['sidechains'][
                'local_frames'].to_array()

        aatype = batch['aatype']
        seq_mask = batch['seq_mask']

        atom14_pred_mask = all_atom_bak.get_atom14_mask(
            aatype) * seq_mask[:, None]
        atom14_pred_positions = output['sc']['atom_pos'][-1]
        ret['final_atom14_positions'] = atom14_pred_positions  # (N, 14, 3)
        ret['final_atom14_mask'] = atom14_pred_mask  # (N, 14)

        atom37_mask = all_atom_bak.get_atom37_mask(aatype) * seq_mask[:, None]
        atom37_pred_positions = all_atom_bak.atom14_to_atom37(atom14_pred_positions, aatype)
        atom37_pred_positions *= atom37_mask[:, :, None]
        ret['final_atom_positions'] = atom37_pred_positions  # (N, 37, 3)
        ret['final_atom_mask'] = atom37_mask  # (N, 37)
        ret['final_rigids'] = ret['traj'][-1]

        ret['act'] = output['act']

        return ret

    
def l2_normalize(x: jnp.ndarray,
                 axis: int = -1,
                 epsilon: float = 1e-12
                 ) -> jnp.ndarray:
    return x / jnp.sqrt(
        jnp.maximum(jnp.sum(x**2, axis=axis, keepdims=True), epsilon))
    
class MultiRigidSidechain(nn.Module):
    """Class to make side chain atoms."""
    config: Config
    global_config: Config
    name: str = 'multi_rigid_sidechain'
    
    @nn.compact
    def __call__(self,
                 rigid: geometry.Rigid3Array,
                 representations_list: Iterable[jnp.ndarray],
                 aatype: jnp.ndarray
                ) -> Dict[str, Any]:
        """Predict sidechains using multi-rigid representations.

        Args:
        rigid: The Rigid's for each residue (translations in angstoms)
        representations_list: A list of activations to predict sidechains from.
        aatype: amino acid types.

        Returns:
        dict containing atom positions and frames (in angstrom)
        """
        
        act = [
        common_modules_bak.Linear(  # pylint: disable=g-complex-comprehension
            self.config.num_channel,
            name='input_projection')(jax.nn.relu(x))
        for x in representations_list]
        # Sum the activation list (equivalent to concat then Conv1D)
        act = sum(act)

        final_init = 'zeros' if self.global_config.zero_init else 'linear'

        # Mapping with some residual blocks.
        for _ in range(self.config.num_residual_block):
            old_act = act
            act = common_modules_bak.Linear(
                self.config.num_channel,
                initializer='relu',
                name='resblock1')(
                    jax.nn.relu(act))
            act = common_modules_bak.Linear(
                self.config.num_channel,
                initializer=final_init,
                name='resblock2')(
                    jax.nn.relu(act))
            act += old_act

        # Map activations to torsion angles.
        # [batch_size, num_res, 14]
        num_res = act.shape[0]
        unnormalized_angles = common_modules_bak.Linear(
            14, name='unnormalized_angles')(
                jax.nn.relu(act))
        unnormalized_angles = jnp.reshape(
            unnormalized_angles, [num_res, 7, 2])
        angles = l2_normalize(unnormalized_angles, axis=-1)

        outputs = {
            'angles_sin_cos': angles,  # jnp.ndarray (N, 7, 2)
            'unnormalized_angles_sin_cos':
                unnormalized_angles,  # jnp.ndarray (N, 7, 2)
        }

        # Map torsion angles to frames.
        # geometry.Rigid3Array with shape (N, 8)
        all_frames_to_global = all_atom_bak.torsion_angles_to_frames(
            aatype,
            rigid,
            angles)

        # Use frames and literature positions to create the final atom coordinates.
        # geometry.Vec3Array with shape (N, 14)
        pred_positions = all_atom_bak.frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global)

        outputs.update({
            'atom_pos': pred_positions,  # geometry.Vec3Array (N, 14)
            'frames': all_frames_to_global,  # geometry.Rigid3Array (N, 8)
        })
        return outputs
