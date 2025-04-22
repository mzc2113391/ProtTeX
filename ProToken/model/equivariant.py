"""Equivariant"""
import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.linen.initializers import lecun_normal, zeros_init
from common.geometry import apply_to_point, invert_point
from modules.basic import Softmax1,ActFuncWrapper

class InvariantPointAttention(nn.Module):
    r"""
    Invariant Point attention module.
    This module is used to update the sequence representation ,which is the first input--inputs_1d,
    adding location information to the sequence representation.

    The attention consists of three parts, namely, q, k, v obtained by the sequence representation,
    q'k'v' obtained by the interaction between the sequence representation and the rigid body group,
    and b , which is th bias, obtained from the pair representation (the second inputs -- inputs_2d).

    .. math::
        a_{ij} = Softmax(w_l(c_1{q_i}^Tk_j+b{ij}-c_2\sum {\left \| T_i\circ q'_i-T_j\circ k'_j \right \| ^{2 } }))

    where i and j represent the ith and jth amino acids in the sequence, respectively,
    and T is the rotation and translation in the input.

    `Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.

    Args:
        num_head (int):         The number of the heads.
        num_scalar_qk (int):    The number of the scalar query/key.
        num_scalar_v (int):     The number of the scalar value.
        num_point_v (int):      The number of the point value.
        num_point_qk (int):     The number of the point query/key.
        num_channel (int):      The number of the channel.
        pair_dim (int):         The last dimension length of pair.

    Inputs:
        - **inputs_1d** (Tensor) - The first row of msa representation which is the output of evoformer module,
          also called the sequence representation, shape :math:`[N_{res}, num\_channel]`.
        - **inputs_2d** (Tensor) - The pair representation which is the output of evoformer module,
          shape :math:`[N_{res}, N_{res}, pair\_dim]`.
        - **mask** (Tensor) - A mask that determines which elements of inputs_1d are involved in the
          attention calculation, shape :math:`[N_{res}, 1]`
        - **rotation** (tuple) - A rotation term in a rigid body group T(r,t),
          A tuple of length 9, The shape of each elements in the tuple is :math:`[N_{res}]`.
        - **translation** (tuple) - A translation term in a rigid body group T(r,t),
          A tuple of length 3, The shape of each elements in the tuple is :math:`[N_{res}]`.

    Outputs:
        Tensor, the update of inputs_1d, shape :math:`[N_{res}, num\_channel]`.

    Supported Platforms
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import InvariantPointAttention
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> import mindspore.context as context
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> model = InvariantPointAttention(num_head=12, num_scalar_qk=16, num_scalar_v=16,
        ...                                 num_point_v=8, num_point_qk=4,
        ...                                 num_channel=384, pair_dim=128)
        >>> inputs_1d = Tensor(np.ones((256, 384)), mstype.float32)
        >>> inputs_2d = Tensor(np.ones((256, 256, 128)), mstype.float32)
        >>> mask = Tensor(np.ones((256, 1)), mstype.float32)
        >>> rotation = tuple([Tensor(np.ones(256), mstype.float16) for _ in range(9)])
        >>> translation = tuple([Tensor(np.ones(256), mstype.float16) for _ in range(3)])
        >>> attn_out = model(inputs_1d, inputs_2d, mask, rotation, translation)
        >>> print(attn_out.shape)
        (256, 384)
    """

    num_head: int
    num_scalar_qk: int
    num_scalar_v: int
    num_point_v: int
    num_point_qk: int
    num_channel: int
    pair_dim: int
    sink_attention: bool = False

    def setup(self):
      
        self._dtype = jnp.float32

        self._dist_epsilon = 1e-8
        self.projection_num = self.num_head * self.num_scalar_v + self.num_head * self.num_point_v * 4 + \
                              self.num_head * self.pair_dim

        self.q_scalar = nn.Dense(features=self.num_head * self.num_scalar_qk, kernel_init=lecun_normal(),
                                 dtype=self._dtype, param_dtype=jnp.float32)
        self.kv_scalar = nn.Dense(features=self.num_head * (self.num_scalar_qk + self.num_scalar_v), kernel_init=lecun_normal(),
                                  dtype=self._dtype, param_dtype=jnp.float32)
        self.q_point_local = nn.Dense(features=self.num_head * 3 * self.num_point_qk, kernel_init=lecun_normal(),
                                      dtype=self._dtype, param_dtype=jnp.float32)
        self.kv_point_local = nn.Dense(features=self.num_head * 3 * (self.num_point_qk + self.num_point_v), kernel_init=lecun_normal(),
                                       dtype=self._dtype, param_dtype=jnp.float32)
        
        self.soft_max = nn.softmax # modified by @linxh
        if self.sink_attention:
            self.soft_max = Softmax1(-1)
        self.soft_plus = nn.softplus

        self.trainable_point_weights = self.param("trainable_point_weights", nn.initializers.ones_init(), (self.num_head,), jnp.float32)
        self.attention_2d = nn.Dense(self.num_head, kernel_init=lecun_normal(), dtype=self._dtype, param_dtype=jnp.float32)
        self.output_projection = nn.Dense(self.num_channel, kernel_init=zeros_init(), dtype=self._dtype, param_dtype=jnp.float32)
        
        self.scalar_weights = jnp.sqrt(1.0 / (3 * 16)).astype(self._dtype)
        self.point_weights = jnp.sqrt(1.0 / (3 * 18)).astype(self._dtype)
        self.attention_2d_weights = jnp.sqrt(1.0 / 3).astype(self._dtype)

    def __call__(self, inputs_1d, inputs_2d, mask, rotation, translation):
        '''construct'''
        num_residues, _ = inputs_1d.shape

        # Improve readability by removing a large number of 'self's.
        num_head = self.num_head
        num_scalar_qk = self.num_scalar_qk
        num_point_qk = self.num_point_qk
        num_scalar_v = self.num_scalar_v
        num_point_v = self.num_point_v

        # Construct scalar queries of shape:
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = jnp.reshape(q_scalar, [num_residues, num_head, num_scalar_qk])

        # Construct scalar keys/values of shape:
        kv_scalar = self.kv_scalar(inputs_1d)
        kv_scalar = jnp.reshape(kv_scalar, [num_residues, num_head, num_scalar_v + num_scalar_qk])
        k_scalar, v_scalar = jnp.split(kv_scalar, [num_scalar_qk], axis=-1)

        # Construct query points of shape:
        # First construct query points in local frame.
        q_point_local = self.q_point_local(inputs_1d)

        q_point_local = jnp.split(q_point_local, 3, axis=-1)
        q_point_local = (jnp.squeeze(q_point_local[0]), jnp.squeeze(q_point_local[1]),
                         jnp.squeeze(q_point_local[2]))
        # Project query points into global frame.
        q_point_global = apply_to_point(rotation, translation, q_point_local, 1)

        # Reshape query point for later use.
        q_point0 = jnp.reshape(q_point_global[0], (num_residues, num_head, num_point_qk))
        q_point1 = jnp.reshape(q_point_global[1], (num_residues, num_head, num_point_qk))
        q_point2 = jnp.reshape(q_point_global[2], (num_residues, num_head, num_point_qk))

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        kv_point_local = self.kv_point_local(inputs_1d)

        kv_point_local = jnp.split(kv_point_local, 3, axis=-1)
        kv_point_local = (jnp.squeeze(kv_point_local[0]), jnp.squeeze(kv_point_local[1]),
                          jnp.squeeze(kv_point_local[2]))
        # Project key and value points into global frame.
        kv_point_global = apply_to_point(rotation, translation, kv_point_local, 1)

        kv_point_global0 = jnp.reshape(kv_point_global[0], (num_residues, num_head, (num_point_qk + num_point_v)))
        kv_point_global1 = jnp.reshape(kv_point_global[1], (num_residues, num_head, (num_point_qk + num_point_v)))
        kv_point_global2 = jnp.reshape(kv_point_global[2], (num_residues, num_head, (num_point_qk + num_point_v)))

        # Split key and value points.
        k_point0, v_point0 = jnp.split(kv_point_global0, [num_point_qk], axis=-1)
        k_point1, v_point1 = jnp.split(kv_point_global1, [num_point_qk], axis=-1)
        k_point2, v_point2 = jnp.split(kv_point_global2, [num_point_qk], axis=-1)

        trainable_point_weights = self.soft_plus(self.trainable_point_weights.astype(self._dtype))
        point_weights = self.point_weights * jnp.expand_dims(trainable_point_weights, axis=1)

        v_point = [jnp.swapaxes(v_point0, -2, -3), jnp.swapaxes(v_point1, -2, -3), jnp.swapaxes(v_point2, -2, -3)]
        q_point = [jnp.swapaxes(q_point0, -2, -3), jnp.swapaxes(q_point1, -2, -3), jnp.swapaxes(q_point2, -2, -3)]
        k_point = [jnp.swapaxes(k_point0, -2, -3), jnp.swapaxes(k_point1, -2, -3), jnp.swapaxes(k_point2, -2, -3)]

        dist2 = jnp.square(jnp.expand_dims(q_point[0], 2) - jnp.expand_dims(k_point[0], 1)) + \
                jnp.square(jnp.expand_dims(q_point[1], 2) - jnp.expand_dims(k_point[1], 1)) + \
                jnp.square(jnp.expand_dims(q_point[2], 2) - jnp.expand_dims(k_point[2], 1))

        attn_qk_point = -0.5 * jnp.sum(jnp.expand_dims(jnp.expand_dims(point_weights, 1), 1) * dist2, axis=-1)

        v = jnp.swapaxes(v_scalar, -2, -3)
        q = jnp.swapaxes(self.scalar_weights * q_scalar, -2, -3)
        k = jnp.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = self.attention_2d(inputs_2d)
        attention_2d = jnp.transpose(attention_2d, [2, 0, 1])
        attention_2d = self.attention_2d_weights * attention_2d

        attn_logits += attention_2d

        mask_2d = mask * jnp.swapaxes(mask, -1, -2)
        attn_logits -= 50 * (1. - mask_2d)

        attn = self.soft_max(attn_logits)

        result_scalar = jnp.matmul(attn, v)

        result_point_global = [jnp.swapaxes(jnp.sum(attn[:, :, :, None] * v_point[0][:, None, :, :], axis=-2), -2, -3),
                               jnp.swapaxes(jnp.sum(attn[:, :, :, None] * v_point[1][:, None, :, :], axis=-2), -2, -3),
                               jnp.swapaxes(jnp.sum(attn[:, :, :, None] * v_point[2][:, None, :, :], axis=-2), -2, -3)]

        result_point_global = [jnp.reshape(result_point_global[0], [num_residues, num_head * num_point_v]),
                               jnp.reshape(result_point_global[1], [num_residues, num_head * num_point_v]),
                               jnp.reshape(result_point_global[2], [num_residues, num_head * num_point_v])]
        result_scalar = jnp.swapaxes(result_scalar, -2, -3)

        result_scalar = jnp.reshape(result_scalar, [num_residues, num_head * num_scalar_v])

        result_point_local = invert_point(result_point_global, rotation, translation, 1)

        output_feature1 = result_scalar
        output_feature20 = result_point_local[0]
        output_feature21 = result_point_local[1]
        output_feature22 = result_point_local[2]

        output_feature3 = jnp.sqrt(self._dist_epsilon +
                                   jnp.square(result_point_local[0]) +
                                   jnp.square(result_point_local[1]) +
                                   jnp.square(result_point_local[2]))

        result_attention_over_2d = jnp.matmul(jnp.swapaxes(attn, 0, 1), inputs_2d)
        num_out = num_head * result_attention_over_2d.shape[-1]
        output_feature4 = jnp.reshape(result_attention_over_2d, [num_residues, num_out])

        final_act = jnp.concatenate([output_feature1, output_feature20, output_feature21,
                                     output_feature22, output_feature3, output_feature4], axis=-1)
        final_result = self.output_projection(final_act)
        return final_result