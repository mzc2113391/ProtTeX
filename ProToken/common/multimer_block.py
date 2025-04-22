# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evoformer"""

import numpy as np
import jax.numpy as jnp

from flax import linen as nn
from flax.linen.initializers import lecun_normal, ones_init, zeros_init
from common.geometry import apply_to_point, invert_point, vecs_from_tensor, \
    vecs_dot_vecs, vecs_sub, vecs_cross_vecs, vecs_scale, \
    rots_expand_dims, vecs_expand_dims, invert_rigids, rigids_mul_vecs

def multimer_square_euclidean_distance(v1, v2, epsilon=1e-5):
    """multimer_square_euclidean_distance."""
    difference = vecs_sub(v1, v2)
    difference += epsilon
    distance = vecs_dot_vecs(difference, difference)
    # if epsilon:
    #     distance = jnp.maximum(distance, epsilon)
    return distance


def multimer_vecs_robust_norm(v, epsilon=1e-5):
    """multime computes norm of vectors 'v'."""
    v_l2_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + epsilon
    # if epsilon:
        # v_l2_norm = jnp.maximum(v_l2_norm, epsilon**2)
    return jnp.sqrt(v_l2_norm)


def multimer_vecs_robust_normalize(v, epsilon=1e-6):
    """multimer normalizes vectors 'v'."""
    norms = multimer_vecs_robust_norm(v, epsilon)
    return (v[0] / norms, v[1] / norms, v[2] / norms)


def multimer_rots_from_two_vecs(e0_unnormalized, e1_unnormalized):
    """multimer_rots_from_two_vecs."""
    e0 = multimer_vecs_robust_normalize(e0_unnormalized)
    c = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = vecs_sub(e1_unnormalized, vecs_scale(e0, c))
    e1 = multimer_vecs_robust_normalize(e1)
    e2 = vecs_cross_vecs(e0, e1)

    rots = (e0[0], e1[0], e2[0],
            e0[1], e1[1], e2[1],
            e0[2], e1[2], e2[2])
    return rots


def multimer_rigids_from_3_points(vec_a, vec_b, vec_c):
    """Create multimer Rigids from 3 points. """
    m = multimer_rots_from_two_vecs(
        e0_unnormalized=vecs_sub(vec_c, vec_b),
        e1_unnormalized=vecs_sub(vec_a, vec_b))
    rigid = (m, vec_b)
    return rigid


def multimer_rigids_get_unit_vector(point_a, point_b, point_c):
    """multimer_rigids_get_unit_vector."""
    rigid = multimer_rigids_from_3_points(vecs_from_tensor(point_a),
                                          vecs_from_tensor(point_b),
                                          vecs_from_tensor(point_c))
    rot, trans = rigid
    rotation = rots_expand_dims(rot, -1)
    translation = vecs_expand_dims(trans, -1)
    inv_rigid = invert_rigids((rotation, translation))
    rigid_vec = rigids_mul_vecs(inv_rigid, vecs_expand_dims(trans, -2))
    unit_vector = multimer_vecs_robust_normalize(rigid_vec)
    return unit_vector


def multimer_rigids_compute_dihedral_angle(a, b, c, d):
    """multimer_rigids_compute_dihedral_angle."""
    v1 = vecs_sub(a, b)
    v2 = vecs_sub(b, c)
    v3 = vecs_sub(d, c)

    c1 = vecs_cross_vecs(v1, v2)
    c2 = vecs_cross_vecs(v3, v2)
    c3 = vecs_cross_vecs(c2, c1)

    v2_mag = multimer_vecs_robust_norm(v2)
    return jnp.arctan2(vecs_dot_vecs(c3, v2), v2_mag * vecs_dot_vecs(c1, c2))


class MultimerInvariantPointAttention(nn.Module):
    """Invariant Point attention module."""

    num_head: int
    num_scalar_qk: int
    num_scalar_v: int
    num_point_v: int
    num_point_qk: int
    num_channel: int
    pair_dim: int

    def setup(self):

        self._dist_epsilon = 1e-8
        self.projection_num = self.num_head * self.num_scalar_v + self.num_head * self.num_point_v * 4 + \
                              self.num_head * self.pair_dim
        self.q_scalar = nn.Dense(features=self.num_head * self.num_scalar_qk,
                                 kernel_init=lecun_normal(), 
                                 has_bias=False)
        self.k_scalar = nn.Dense(features=self.num_head * self.num_scalar_qk,
                                 kernel_init=lecun_normal(), 
                                 has_bias=False)
        self.v_scalar = nn.Dense(features=self.num_head * self.num_scalar_v,
                                 kernel_init=lecun_normal(), 
                                 has_bias=False)
        self.q_point_local = nn.Dense(features=self.num_head * 3 * self.num_point_qk,
                                      kernel_init=lecun_normal())
        self.k_point_local = nn.Dense(features=self.num_head * 3 * self.num_point_qk,
                                      kernel_init=lecun_normal())
        self.v_point_local = nn.Dense(features=self.num_head * 3 * self.num_point_v,
                                      kernel_init=lecun_normal())

        self.trainable_point_weights = self.param(name="trainable_point_weights",
                                                  init_fn=ones_init(),
                                                  shape=(12,),
                                                  dtype=jnp.float32)
        self.attention_2d = nn.Dense(self.num_head, kernel_init=lecun_normal())
        self.output_projection = nn.Dense(self.num_channel, kernel_init=zeros_init())

        self.point_weights = jnp.sqrt(1.0 / (max(self.num_point_qk, 1) * 9. / 2))
        self.scalar_weights = jnp.sqrt(1.0 / (max(self.num_scalar_qk, 1) * 1.))

    def __call__(self, inputs_1d, inputs_2d, mask, rotation, translation):
        """Compute geometry-aware attention.

        Args:
          inputs_1d: (N, C) 1D input embedding that is the basis for the
            scalar queries.
          inputs_2d: (N, M, C') 2D input embedding, used for biases and values.
          mask: (N, 1) mask to indicate which elements of inputs_1d participate
            in the attention.
          rotation: describe the orientation of every element in inputs_1d
          translation: describe the position of every element in inputs_1d

        Returns:
          Transformation of the input embedding.
        """
        num_residues, _ = inputs_1d.shape

        num_head = self.num_head
        attn_logits = 0.
        num_point_qk = self.num_point_qk
        point_weights = self.point_weights

        trainable_point_weights = jnp.logaddexp(self.trainable_point_weights,
                                                jnp.zeros_like(self.trainable_point_weights))
        point_weights = point_weights * trainable_point_weights

        q_point_local = self.q_point_local(inputs_1d)
        q_point_local = jnp.reshape(q_point_local, (num_residues, num_head, num_point_qk * 3))
        q_point_local = jnp.split(q_point_local, 3, axis=-1)
        q_point_local = (jnp.squeeze(q_point_local[0]), jnp.squeeze(q_point_local[1]),
                         jnp.squeeze(q_point_local[2]))
        # Project query points into global frame.
        q_point_global = apply_to_point(rotation, translation, q_point_local, 2)
        q_point = [q_point_global[0][:, None, :, :], q_point_global[1][:, None, :, :], q_point_global[2][:, None, :, :]]

        k_point_local = self.k_point_local(inputs_1d)
        k_point_local = jnp.reshape(k_point_local, (num_residues, num_head, num_point_qk * 3))
        k_point_local = jnp.split(k_point_local, 3, axis=-1)
        k_point_local = (jnp.squeeze(k_point_local[0]), jnp.squeeze(k_point_local[1]),
                         jnp.squeeze(k_point_local[2]))

        # Project query points into global frame.
        k_point_global = apply_to_point(rotation, translation, k_point_local, 2)
        k_point = [k_point_global[0][None, :, :, :], k_point_global[1][None, :, :, :], k_point_global[2][None, :, :, :]]

        dist2 = multimer_square_euclidean_distance(q_point, k_point, epsilon=0.)

        attn_qk_point = -0.5 * jnp.sum(point_weights[:, None] * dist2, axis=-1)
        attn_logits += attn_qk_point

        num_scalar_qk = self.num_scalar_qk

        scalar_weights = self.scalar_weights
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = jnp.reshape(q_scalar, [num_residues, num_head, num_scalar_qk])

        k_scalar = self.k_scalar(inputs_1d)
        k_scalar = jnp.reshape(k_scalar, [num_residues, num_head, num_scalar_qk])

        q_scalar *= scalar_weights
        q = jnp.swapaxes(q_scalar, -2, -3)
        k = jnp.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        attn_qk_scalar = jnp.swapaxes(attn_qk_scalar, -2, -3)
        attn_qk_scalar = jnp.swapaxes(attn_qk_scalar, -2, -1)
        attn_logits += attn_qk_scalar

        attention_2d = self.attention_2d(inputs_2d)
        attn_logits += attention_2d

        mask_2d = mask * jnp.swapaxes(mask, -1, -2)
        attn_logits -= 1e5 * (1. - mask_2d[..., None])
        attn_logits *= jnp.sqrt(1. / 3)
        attn = nn.softmax(attn_logits, axis=-2)

        num_scalar_v = self.num_scalar_v
        v_scalar = self.v_scalar(inputs_1d)
        v_scalar = jnp.reshape(v_scalar, [num_residues, num_head, num_scalar_v])

        attn_tmp = jnp.swapaxes(attn, -1, -2)
        attn_tmp = jnp.swapaxes(attn_tmp, -2, -3)
        result_scalar = jnp.matmul(attn_tmp, jnp.swapaxes(v_scalar, -2, -3))
        result_scalar = jnp.swapaxes(result_scalar, -2, -3)

        num_point_v = self.num_point_v

        v_point_local = self.v_point_local(inputs_1d)
        v_point_local = jnp.reshape(v_point_local, (num_residues, num_head, num_point_v * 3))
        v_point_local = jnp.split(v_point_local, 3, axis=-1)
        v_point_local = (jnp.squeeze(v_point_local[0]), jnp.squeeze(v_point_local[1]),
                         jnp.squeeze(v_point_local[2]))
        # Project query points into global frame.
        v_point_global = apply_to_point(rotation, translation, v_point_local, 2)
        v_point = [v_point_global[0][None], v_point_global[1][None], v_point_global[2][None]]

        result_point_global = [jnp.sum(attn[..., None] * v_point[0], axis=-3),
                               jnp.sum(attn[..., None] * v_point[1], axis=-3),
                               jnp.sum(attn[..., None] * v_point[2], axis=-3)
                               ]

        num_query_residues, _ = inputs_1d.shape

        result_scalar = jnp.reshape(result_scalar, [num_query_residues, -1])

        output_feature1 = result_scalar

        result_point_global = [jnp.reshape(result_point_global[0], [num_query_residues, -1]),
                               jnp.reshape(result_point_global[1], [num_query_residues, -1]),
                               jnp.reshape(result_point_global[2], [num_query_residues, -1])]
        result_point_local = invert_point(result_point_global, rotation, translation, 1)
        output_feature20 = result_point_local[0]
        output_feature21 = result_point_local[1]
        output_feature22 = result_point_local[2]
        point_norms = multimer_vecs_robust_norm(result_point_local, self._dist_epsilon)
        output_feature3 = point_norms

        result_attention_over_2d = jnp.matmul(jnp.swapaxes(attn, 1, 2), inputs_2d)
        output_feature4 = jnp.reshape(result_attention_over_2d, [num_query_residues, -1])
        final_act = jnp.concatenate([output_feature1, output_feature20, output_feature21,
                                     output_feature22, output_feature3, output_feature4], axis=-1)
        final_result = self.output_projection(final_act)
        return final_result
