# ============================================================================
# Copyright 2023 CPL Research Group
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
"""attention modules"""
import jax
import numpy as np
import jax.numpy as jnp

from common.config_load import load_config, Config
from flax import linen as nn
from flax.linen.initializers import glorot_uniform, zeros_init, ones_init, normal, lecun_normal
from .basic import Softmax1, ActFuncWrapper, Swish_beta, TransMatMul, RotaryEmbedding
from .basic import masked_layer_norm
import ml_collections



class Attention(nn.Module):
    r"""
    This is an implementation of multihead attention in the paper `Attention is all you need
    <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length,
    and the key with key length and the target length, the attention will be performed as
    the following.

    .. math::

        Attention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

    if query, key and value tensor is same, then it will be modified version of self
    attention.

    Args:
        num_head(int):     The number of the heads.
        hidden_dim(int):   The hidden size of the input.
        gating(bool):       Indicator of if the attention is gated.
        q_data_dim(int):    The last dimension length of the query tensor.
        m_data_dim(int):    The last dimension length of the key and value tensor.
        output_dim(int):    The last dimension length of the output tensor.
        batch_size(int):    The batch size of parameters in attention, used in while
                            control flow. Default: ``None``.

    Inputs:
        - **q_data** (Tensor) - The query tensor with shape :math:`(batch\_size,
          query\_seq_length, q\_data_dim)` with query_seq_length the query sequence length.
        - **m_data** (Tensor) - The key/value tensor with shape :math:`(batch\_size,
          value\_seq_length, m\_data_dim)` with value_seq_length the value sequence length.
        - **attention_mask** (Tensor) - The mask for attention matrix with shape
          :math:`(batch\_size, num\_head, query\_seq_length, value\_seq_length)`.
        - **index** (Tensor) - The index of while loop, only used in case of while
          control flow. Default: ``None``.
        - **nonbatched_bias** (Tensor) - Non-batched bias for the attention matrix with
          shape :math:`(num\_heads, query\_seq_length, value\_seq_length)`. Default: ``None``.

    Outputs:
        Tensor, output tensor of the Attention layer with shape :math:`(batch\_size, query\_seq_length, hidden\_size)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import Attention
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = Attention(num_head=4, hidden_dim=64, gating=True, q_data_dim=64,
        ...                   m_data_dim=64, output_dim=64)
        >>> q_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
        >>> m_data = Tensor(np.ones((32, 256, 64)), mstype.float32)
        >>> attention_mask = Tensor(np.ones((32, 4, 128, 256)), mstype.float32)
        >>> attn_out= model(q_data, m_data, attention_mask)
        >>> print(attn_out.shape)
        (32, 128, 64)
    """

    global_config: ml_collections.ConfigDict
    num_head: int
    hidden_dim: int
    q_data_dim: int
    m_data_dim: int
    output_dim: int
    gating: bool = True
    sink_attention: bool = False
    rope: bool = False
    seq_dim: int = -2

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.dim_per_head = self.hidden_dim // self.num_head
        
        self.apply_rope = None
        if self.rope:
            self.apply_rope = RotaryEmbedding(dim=self.dim_per_head,
                                              custom_idx=False, # True,
                                              seq_dim=self.seq_dim,)
        
        self.wmatmul = TransMatMul(transpose_b=True)

        self.linear_q = nn.Dense(features=self.num_head * self.dim_per_head, kernel_init=nn.initializers.glorot_uniform(),
                                 use_bias=False, dtype=self._dtype, param_dtype=jnp.float32)
        self.linear_k = nn.Dense(features=self.num_head * self.dim_per_head, kernel_init=nn.initializers.glorot_uniform(),
                                 use_bias=False, dtype=self._dtype, param_dtype=jnp.float32)
        self.linear_v = nn.Dense(features=self.num_head * self.dim_per_head, kernel_init=nn.initializers.glorot_uniform(),
                                 use_bias=False, dtype=self._dtype, param_dtype=jnp.float32)

        self.softmax = nn.softmax # modified by @linxh, zero sink
        if self.sink_attention is True:
            self.softmax = Softmax1(-1)
        self.sigmoid = nn.sigmoid
        self.softmax = ActFuncWrapper(self.softmax)
        self.sigmoid = ActFuncWrapper(self.sigmoid)
        
        self.zero = jnp.array([0.], self._dtype)

        self._init_parameter()

    def __call__(self, q_data, k_data, v_data, pair_bias_fp32=0., pos_index=None):
        '''construct'''
        ### Shapes:
        ### q_data:(B,Q,c1); k&v_data:(B,K,c2); pair_bias_fp32:(B,h/1,Q,K)
        ### pos_index:(B,Q)

        linear_output_weight = self.linear_output_weight        

        linear_gating_weight = self.linear_gating_weight if self.gating else self.zero
        gating_bias = self.gating_bias if self.gating else self.zero
        o_bias = self.o_bias if self.gating else self.zero

        dim_q, dim_a = q_data.shape
        dim_k, dim_c = k_data.shape ### k_data.shape == v_data.shape
        dim_h = self.num_head

        q_data = jnp.reshape(q_data, (-1, dim_a))
        k_data = jnp.reshape(k_data, (-1, dim_c))
        v_data = jnp.reshape(v_data, (-1, dim_c))

        q = self.linear_q(q_data) * self.dim_per_head ** (-0.5)
        k = self.linear_k(k_data)
        v = self.linear_v(v_data)

        q = jnp.reshape(q, (dim_q, dim_h, -1))
        k = jnp.reshape(k, (dim_k, dim_h, -1))
        v = jnp.reshape(v, (dim_k, dim_h, -1))

        tmp_q = jnp.transpose(q, (1, 0, 2))
        tmp_k = jnp.transpose(k, (1, 0, 2))

        if self.apply_rope is not None: ### @JeffSHF
            tmp_q_flatten = jnp.reshape(tmp_q, (-1,) + tmp_q.shape[1:]) # (B*h,Q,c)
            tmp_k_flatten = jnp.reshape(tmp_k, (-1,) + tmp_k.shape[1:]) # (B*h,K,c)
            q_, k_ = self.apply_rope(q=tmp_q_flatten, k=tmp_k_flatten, 
                                     pos_idx=pos_index) # (B*h,Q,c), (B*h,K,c) # double check its correctness
            tmp_q = jnp.reshape(q_, tmp_q.shape)
            tmp_k = jnp.reshape(k_, tmp_q.shape)

        # logits = jnp.matmul(tmp_q, tmp_k)
        logits = jnp.einsum("hqc, hkc -> hqk", tmp_q, tmp_k) # (h,Q,K) # (BF16/FP32)
        
        ### In case of under/overflow: @ZhangJ.
        weights = self.softmax(logits+pair_bias_fp32).astype(self._dtype) 
        
        tmp_v = jnp.transpose(v, (1, 2, 0))
        weighted_avg = jnp.transpose(jnp.einsum("hqk, hck -> hqc", weights, tmp_v), (1, 0, 2))
        # weighted_avg = jnp.transpose(jnp.matmul(weights, tmp_v.T), (1, 0, 2))

        if self.gating:
            gating_bias = jnp.expand_dims(gating_bias, 0) # (1,h,c)
            gate_values = jnp.add(jnp.reshape(self.wmatmul(q_data, linear_gating_weight), (dim_q, dim_h, -1)), gating_bias).astype(self._dtype)
            # gate_values = self.linear_gating(q_data)
            gate_values = self.sigmoid(gate_values)
            weighted_avg = jnp.reshape(weighted_avg * gate_values, (dim_q, -1))
            o_bias = jnp.expand_dims(o_bias, 0).astype(self._dtype)

        weighted_avg = jnp.reshape(weighted_avg, (dim_q, -1))
        output = jnp.add(jnp.reshape(self.wmatmul(weighted_avg, linear_output_weight.astype(self._dtype)), (dim_q, -1)), o_bias).astype(self._dtype)
        
        return output # (Q,c)

    def _init_parameter(self):
        '''init parameter'''
        # self.linear_q_weight = self.param("linear_q_weight", glorot_uniform(dtype=jnp.float32), 
        #                                   [self.num_head * self.dim_per_head, self.q_data_dim], jnp.float32)
        # self.linear_k_weight = self.param("linear_k_weight", glorot_uniform(dtype=jnp.float32), 
        #                                   [self.num_head * self.dim_per_head, self.m_data_dim], jnp.float32)
        # self.linear_v_weight = self.param("linear_v_weight", glorot_uniform(dtype=jnp.float32), 
        #                                   [self.num_head * self.dim_per_head, self.m_data_dim], jnp.float32)
        self.linear_output_weight = self.param("linear_output_weight", zeros_init(), 
                                               [self.output_dim, self.num_head * self.dim_per_head], jnp.float32)
        
        if self.gating:
            self.linear_gating_weight = self.param("linear_gating_weight", zeros_init(),
                                                   [self.num_head * self.dim_per_head, self.q_data_dim], jnp.float32)
            self.gating_bias = self.param("gating_b", ones_init(), [self.num_head, self.dim_per_head], jnp.float32)
            self.o_bias = self.param("o_bias", zeros_init(), [self.output_dim,], jnp.float32)
            # self.linear_gating = nn.Dense(features=self.num_head * self.dim_per_head, kernel_init=nn.initializers.zeros_init(),
            #                               use_bias=True, bias_init=nn.initializers.ones_init(),
            #                               dtype=self._dtype, param_dtype=jnp.float32)


# class HyperAttention(nn.Module):
#     r"""
#     This is an implementation of multihead attention in the paper `Attention is all you need
#     <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length,
#     and the key with key length and the target length, the attention will be performed as
#     the following.

#     .. math::

#         Attention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

#     where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

#     if query, key and value tensor is same, then it will be modified version of self
#     attention.

#     Args:
#         num_head(int):     The number of the heads.
#         hidden_dim(int):   The hidden size of the input.
#         gating(bool):       Indicator of if the attention is gated.
#         q_data_dim(int):    The last dimension length of the query tensor.
#         m_data_dim(int):    The last dimension length of the key and value tensor.
#         output_dim(int):    The last dimension length of the output tensor.
#         batch_size(int):    The batch size of parameters in attention, used in while
#                             control flow. Default: ``None``.

#     Inputs:
#         - **q_data** (Tensor) - The query tensor with shape :math:`(batch\_size,
#           query\_seq_length, q\_data_dim)` with query_seq_length the query sequence length.
#         - **m_data** (Tensor) - The key/value tensor with shape :math:`(batch\_size,
#           value\_seq_length, m\_data_dim)` with value_seq_length the value sequence length.
#         - **attention_mask** (Tensor) - The mask for attention matrix with shape
#           :math:`(batch\_size, num\_head, query\_seq_length, value\_seq_length)`.
#         - **index** (Tensor) - The index of while loop, only used in case of while
#           control flow. Default: ``None``.
#         - **nonbatched_bias** (Tensor) - Non-batched bias for the attention matrix with
#           shape :math:`(num\_heads, query\_seq_length, value\_seq_length)`. Default: ``None``.

#     Outputs:
#         Tensor, output tensor of the Attention layer with shape :math:`(batch\_size, query\_seq_length, hidden\_size)`.

#     Supported Platforms:
#         ``Ascend`` ``GPU``

#     Examples:
#         >>> import numpy as np
#         >>> from mindsponge.cell import Attention
#         >>> from mindspore import dtype as mstype
#         >>> from mindspore import Tensor
#         >>> model = Attention(num_head=4, hidden_dim=64, gating=True, q_data_dim=64,
#         ...                   m_data_dim=64, output_dim=64)
#         >>> q_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
#         >>> m_data = Tensor(np.ones((32, 256, 64)), mstype.float32)
#         >>> attention_mask = Tensor(np.ones((32, 4, 128, 256)), mstype.float32)
#         >>> attn_out= model(q_data, m_data, attention_mask)
#         >>> print(attn_out.shape)
#         (32, 128, 64)
#     """

#     def __init__(self, num_head, hidden_dim, q_data_dim, m_data_dim, output_dim,
#                  sink_attention=True,
#                  gating=False,
#                  rope=False,
#                  seq_dim=-2,
#                  lora_rank=0,
#                  ):
#         super(HyperAttention, self).__init__()
#         self.q_data_dim = q_data_dim
#         self.m_data_dim = m_data_dim
#         self.output_dim = output_dim
#         self.num_head = num_head
#         self.gating = gating
#         self.hidden_dim = hidden_dim
#         self.dim_per_head = self.hidden_dim // self.num_head
        
#         self.apply_rope = None
#         if rope:
#             self.apply_rope = RotaryEmbedding(
#                 dim=self.dim_per_head,
#                 custom_idx=False, # True,
#                 seq_dim=seq_dim,
#                 )
        
#         self.hyper = False
#         if lora_rank > 0:
#             self.hyper = True
#             self.lora_left = LoRA(rank=lora_rank) # @ZhangJ.
#             self.lora_right = LoRA(rank=lora_rank)
        
#         self.matmul = O.MatMul(transpose_b=True)
#         self.batch_matmul_trans_b = O.BatchMatMul(transpose_b=True)
#         self.wmatmul = TransMatMul(transpose_b=True)

#         self.softmax = nn.Softmax(-1) # modified by @linxh, zero sink
#         if sink_attention is True:
#             self.softmax = Softmax1(-1)
        
#         self.sigmoid = nn.Sigmoid()
#         self.cast = O.Cast()
        
#         self._dtype = mnp.float32
#         self.convert_fp = ConvertFP("FP32")
#         if MIXED_PRECISION_FLAG: ### add recursive flags
#             self._dtype = mnp.float16
#             self.convert_fp = ConvertFP("FP16")
#             self.softmax = ActFuncWrapper(self.softmax)
#             self.sigmoid = ActFuncWrapper(self.sigmoid)
        
#         self.zero = Tensor([0.], self._dtype)

#         self._init_parameter()
#         # O.Print()(self.linear_v_weights.requires_grad)

#     def __call__(self, q_data, k_data, v_data, pair_bias_fp32, pair_act_geom, pos_index=None):
#         '''construct'''
#         ### Shapes:
#         ### q_data:(B,Q,c1); k&v_data:(B,K,c2); pair_bias_fp32:(B,h/1,Q,K)
#         ### pos_index:(B,Q)

#         linear_q_weight = self.linear_q_weight
#         linear_k_weight = self.linear_k_weight
#         linear_v_weight = self.linear_v_weight
#         linear_output_weight = self.linear_output_weight
        
#         o_bias = self.zero
#         linear_gating_weight = self.zero
#         gating_bias = self.zero
#         if self.gating:
#             linear_gating_weight = self.linear_gating_weight
#             gating_bias = self.gating_bias
#             o_bias = self.o_bias

#         fp_convert_list = [linear_q_weight, linear_k_weight, linear_v_weight, linear_output_weight,\
#                            linear_gating_weight, gating_bias, o_bias]
#         fp_convert_list += [q_data, k_data, v_data]

#         linear_q_weight, linear_k_weight, linear_v_weight, linear_output_weight,\
#             linear_gating_weight, gating_bias, o_bias,\
#             q_data, k_data, v_data = self.convert_fp(fp_convert_list)

#         dim_b, dim_q, dim_a = q_data.shape
#         _, dim_k, dim_c = k_data.shape ### k_data.shape == v_data.shape
#         dim_h = self.num_head

#         q_data = jnp.reshape(q_data, (-1, dim_a))
#         k_data = jnp.reshape(k_data, (-1, dim_c))
#         v_data = jnp.reshape(v_data, (-1, dim_c))

#         q_full = self.wmatmul(q_data, linear_q_weight) * self.dim_per_head ** (-0.5) # (B*Q,c)
#         k_full = self.wmatmul(k_data, linear_k_weight) # (B*K,c)
#         v_full = self.wmatmul(v_data, linear_v_weight) # (B*V,c)

#         q = jnp.reshape(q_full, (dim_b, dim_q, dim_h, -1))
#         k = jnp.reshape(k_full, (dim_b, dim_k, dim_h, -1))
#         v = jnp.reshape(v_full, (dim_b, dim_k, dim_h, -1))

#         tmp_q = O.Transpose()(q, (0, 2, 1, 3))
#         tmp_k = O.Transpose()(k, (0, 2, 1, 3))

#         if self.apply_rope is not None: ### @JeffSHF
#             tmp_q_flatten = jnp.reshape(tmp_q, (-1,) + tmp_q.shape[2:]) # (B*h,Q,c)
#             tmp_k_flatten = jnp.reshape(tmp_k, (-1,) + tmp_k.shape[2:]) # (B*h,K,c)
#             q_, k_ = self.apply_rope(tmp_q_flatten, tmp_k_flatten, pos_idx=pos_index) # (B*h,Q,c), (B*h,K,c)
#             tmp_q = jnp.reshape(q_, tmp_q.shape)
#             tmp_k = jnp.reshape(k_, tmp_q.shape)

#         logits = self.batch_matmul_trans_b(tmp_q, tmp_k) # (B*h,Q,c)
#         logits_flat_shape = logits.shape

#         ### Hyper-attention:
#         if self.hyper:
#             lora_mat_left = self.lora_left(pair_act_geom) # (B,Q,K,r*c)->(B*Q,r,c)
#             lora_mat_right = self.lora_right(pair_act_geom) # (B,Q,K,r*c)->(B*K,r,c)
#             # lora_batch_shape = (-1,) + lora_mat_left.shape[-2:]
#             # lora_mat_left = jnp.reshape(lora_mat_left, lora_batch_shape) # (B*Q,r,c)
#             # lora_mat_right = jnp.reshape(lora_mat_right, lora_batch_shape) # (B*K,r,c)

#             logits_left = self.batch_matmul_trans_b(jnp.expand_dims(q_full, -2), lora_mat_left) # (B*Q,1,c)@(B*Q,r,c)->(B*Q,1,r)
#             logits_left = jnp.reshape(logits_left, (dim_b, dim_q, -1)) # (B,Q,r)
#             logits_right = self.batch_matmul_trans_b(jnp.expand_dims(k_full, -2), lora_mat_right) # (B*K,1,c)@(B*K,r,c)->(B*K,1,r)
#             logits_right = jnp.reshape(logits_right, (dim_b, dim_k, -1)) # (B,K,r)
#             logits_hyper = self.batch_matmul_trans_b(logits_left, logits_right) # (B,Q,r)@(B,K,r)->(B,Q,K)
#             logits = jnp.reshape(logits, (dim_b,dim_h,dim_q,-1)) + jnp.expand_dims(logits_hyper, 1) # (B,h,Q,K) + (B,1,Q,K)
#             logits = jnp.reshape(logits, logits_flat_shape) # (B*h,Q,K)
        
#         ### In case of under/overflow: @ZhangJ.
#         logits = self.cast(logits, mnp.float32)
#         logits = O.Add()(logits, pair_bias_fp32)

#         weights = self.softmax(logits)
#         weights = self.cast(weights, self._dtype)

#         tmp_v = O.Transpose()(v, (0, 2, 3, 1))
#         weighted_avg = O.Transpose()(self.batch_matmul_trans_b(weights, tmp_v), (0, 2, 1, 3))

#         if self.gating:
#             gating_bias = jnp.expand_dims(jnp.expand_dims(gating_bias, 0), 0)
#             gate_values = O.Add()(jnp.reshape(self.wmatmul(q_data, linear_gating_weight),
#                                               (dim_b, dim_q, dim_h, -1)),
#                                   gating_bias)
#             gate_values = self.sigmoid(gate_values)
#             weighted_avg = jnp.reshape(weighted_avg * gate_values, (dim_b * dim_q, -1))
#             o_bias = jnp.expand_dims(o_bias, 0)

#         weighted_avg = jnp.reshape(weighted_avg, (dim_b * dim_q, -1))
#         output = O.Add()(jnp.reshape(self.wmatmul(weighted_avg, linear_output_weight),
#                                      (dim_b, dim_q, -1)), o_bias)
#         return output

#     def _init_parameter(self):
#         '''init parameter'''
#         self.linear_q_weight = Parameter(Tensor(
#             glorot_uniform(self.num_head * self.q_data_dim, self.dim_per_head * self.q_data_dim,
#                             [self.num_head * self.dim_per_head, self.q_data_dim]),
#             mnp.float32))
#         self.linear_k_weight = Parameter(Tensor(
#             glorot_uniform(self.num_head * self.m_data_dim, self.dim_per_head * self.m_data_dim,
#                             [self.num_head * self.dim_per_head, self.m_data_dim]),
#             mnp.float32))
#         self.linear_v_weight = Parameter(Tensor(
#             glorot_uniform(self.num_head * self.m_data_dim, self.dim_per_head * self.m_data_dim,
#                             [self.num_head * self.dim_per_head, self.m_data_dim]),
#             mnp.float32))
#         self.linear_output_weight = Parameter(
#             Tensor(np.zeros([self.output_dim, self.num_head * self.dim_per_head]),
#                     mnp.float32))
        
#         if self.gating:
#             self.linear_gating_weight = Parameter(
#                 Tensor(np.zeros([self.num_head * self.dim_per_head, self.q_data_dim]),
#                         mnp.float32))
#             self.gating_bias = Parameter(Tensor(np.ones((self.num_head, self.dim_per_head)),
#                                                     mnp.float32),
#                                             name="gating_b")
#             self.o_bias = Parameter(Tensor(np.zeros([self.output_dim]), mnp.float32))


class PreNonLinear(nn.Module):

    global_config: ml_collections.ConfigDict
    q_data_dim: int
    m_data_dim: int = 1
    pair_act_dim: int = 1
    num_head: int = 1
    operation_list: tuple = ("PairEmbedding",) # ["PairEmbedding", "AttDropout", "LN"]
    norm_method: str = "layernorm" ### ["layernorm", "rmsnorm"]
    dropout_rate: float = 0.
    self_attention: bool = True

    def setup(self):
        # *** PreAttention ***
        # 1. dropout -> masks
        # 2. mask -> pair_bias
        # 3. RelPos -> pair_bias (dense)
        # 4. preLN -> Q (& K)
        # 5. Sinusoidal or BERT-like APE

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32


        supported_operations = ("PairEmbedding", "AttDropout", "LN")
        if len(self.operation_list) > 0:
            for item in self.operation_list:
                assert item in supported_operations
                
        self._m_data_dim = self.q_data_dim if self.self_attention else self.m_data_dim
        
        self.pair_embedding = True if "PairEmbedding" in self.operation_list else False
        self.attention_dropout = True if "AttDropout" in self.operation_list and self.dropout_rate > 1e-3 else False
        self.ln = True if "LN" in self.operation_list else False

        self.wmatmul = TransMatMul(transpose_b=True)
        self.zero = jnp.array([0.], self._dtype)
        self.one = jnp.array([1.], self._dtype)
        
        self._init_parameter()

    def __call__(self, q_data, k_data=None, v_data=None, attention_masks=(None, None, None), pair_act=0.):
        ### Shapes:
        ### q/k/v_data:(B,Q/K,c)
        ### pair_act:(B,Q,K,cz)
        ### pos_index:(B,Q); for sinusoidal or learnable BERT-like PE.
        ### attention_masks: [(B,Q), (B,K), (B,Q,K)]

        attention_mask_q, attention_mask_k, attention_mask_2d = attention_masks

        if self.self_attention:
            k_data = q_data
            v_data = q_data
        
        pair_act_bias = self.zero
        if self.pair_embedding:
            feat_2d_weight = self.feat_2d_weight
            pair_act_shape = pair_act.shape
            # (Q,K,cz):
            pair_act_bias += masked_layer_norm(self.norm_p, pair_act, mask=attention_mask_2d)
            # (Q*K,C):
            pair_act_bias = jnp.reshape(pair_act_bias, (-1, pair_act_shape[-1]))
            # (Q*K,C)@(h,C).T -> (Q*K,h) -> (Q,K,h) -> (h,Q,K):
            pair_act_bias = jnp.transpose(jnp.reshape(self.wmatmul(pair_act_bias, feat_2d_weight), pair_act_shape[:-1]+(self.num_head,)), (2, 0, 1)) # FP32

        if self.attention_dropout:
            attention_mask_2d = self.dropout(attention_mask_2d) # haven't set broadcast dims, maybe no need for this

        pair_bias_fp32 = self.zero.astype(jnp.float32)
        if attention_mask_2d is not None:
            # (Q,K)
            attention_mask_2d_fp32 = jnp.asarray(attention_mask_2d, jnp.float32)
            pair_bias_fp32 += (attention_mask_2d_fp32 - 1.) * 1e5 # (Q,K); Padding
            pair_bias_fp32 = jnp.expand_dims(pair_bias_fp32, 0) # (1/h,Q,K)

        pair_bias_fp32 += jnp.asarray(pair_act_bias, jnp.float32) # (h,Q,K)
        
        q_act = q_data
        k_act = k_data
        v_act = v_data
        if self.ln:  
            q_act = masked_layer_norm(self.norm_s, q_act, mask=attention_mask_q)
            if self.self_attention:
                k_act = q_act
            else:
                k_act = masked_layer_norm(self.norm_s, k_act, mask=attention_mask_k)
            v_act = k_act
        
        return q_act, k_act, v_act, pair_bias_fp32
    
    def _init_parameter(self):
    #     '''init parameter'''    
        
        if self.ln:
            if self.norm_method == "layernorm":
                self.norm_s = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
            elif self.norm_method == "rmsnorm":
                self.norm_s = ActFuncWrapper(nn.RMSNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
        if self.pair_embedding:
            self.feat_2d_weight = self.param("feat_2d_weight", normal(stddev=1/np.sqrt(self.pair_act_dim)),
                                             (self.num_head, self.pair_act_dim), jnp.float32)
            if self.norm_method == "layernorm":
                self.norm_p = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
            elif self.norm_method == "rmsnorm":
                self.norm_p = ActFuncWrapper(nn.RMSNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
        if self.attention_dropout:
            self.dropout = nn.Dropout(rate=self.dropout_rate, deterministic=(not self.dropout_flag),)

class PostNonLinear(nn.Module):

    global_config: ml_collections.ConfigDict
    o_data_dim: int
    operation_list: tuple = ("LN",) # ["Dropout", "LN", "ResidualLN"]
    dropout_rate: float = 0.
    norm_method: str = "layernorm" # ["layernorm", "rmsnorm"]
    accumulated_scale: float = 1.0
    execute_residual: bool = True

    def setup(self):
        # *** PreAttention ***
        # 1. dropout
        # 2. PostLN -> V

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        supported_operations = ("Dropout", "LN", "ResidualLN")
        if len(self.operation_list) > 0:
            for item in self.operation_list:
                assert item in supported_operations, "unsupported operation."
        
        self.dropout = True if "Dropout" in self.operation_list and self.dropout_rate > 1e-3 else False
        self.ln = True if "LN" in self.operation_list else False
        self.residual_ln = True if "ResidualLN" in self.operation_list else False

        self.zero = jnp.array([0.], self._dtype)

        self._init_parameter()

    def __call__(self, act, o_data, o_mask=jnp.array([1.]), accumulated_act=0.):
        ### Shapes:
        ### o_data: (B,Q,c)
        ### o_mask: (B,Q)

        residual_act = act
        o_act = o_data

        if self.dropout:
            o_act = self.dropout(o_act)

        if self.execute_residual:
            accumulated_act += o_act * self.accumulated_scale # zhenyu: * float will not change its dtype
            residual_act = residual_act + o_act

        if self.ln:
            residual_act = masked_layer_norm(self.norm, residual_act, mask=o_mask)

        if self.residual_ln:
            accumulated_act = masked_layer_norm(self.norm, accumulated_act, mask=o_mask)

        return residual_act, accumulated_act
    
    def _init_parameter(self):
        if self.dropout:
            self.dropout = nn.Dropout(rate=self.dropout_rate, deterministic=(not self.dropout_flag))
        if self.ln or self.residual_ln:         
            if self.norm_method == "layernorm": ## Liyh: makeshift
                self.norm = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))
            elif self.norm_method == "rmsnorm":
                self.norm = ActFuncWrapper(nn.RMSNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32))

class FeedForwardNet(nn.Module):

    global_config: ml_collections.ConfigDict
    input_dim: int
    output_dim: int
    intermediate_dim: int = 4
    init_sigma: float = 0.02
    swish_beta: float = 1.
    init_method: str = "AF2", ### ["AF2", "GLM"]

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        
        intermediate_dim = 4 if self.intermediate_dim is None else self.intermediate_dim
        # self.hidden_dim = int(intermediate_dim * 2 * self.input_dim //3)
        if (intermediate_dim * 2 * self.input_dim) % 3 == 0 :
            self.hidden_dim = int(intermediate_dim * 2 * self.input_dim //3)
        else:
            self.hidden_dim = intermediate_dim * self.input_dim # 512

        self.silu = ActFuncWrapper(nn.silu) ## Liyh: using silu or swish beta? zhenyu: silu
        self.wmatmul = TransMatMul(transpose_b=True)

        if self.init_method == "AF2":
            self._init_parameter_af2()
        else:
            self._init_parameter()
    
    def __call__(self, input):

        act = input
        # assert act.dtype == self._dtype, f"act.dtype: {act.dtype}"
        # act = self.wmatmul(act, self.W_weight)
        act = self.W_linear(act) # 128 -> 1024
        act1, act2 = jnp.split(act, 2, axis=-1) # 1024 -> 512, 512
        act = self.silu(act1) * act2 # 512
        # act = self.wmatmul(act, self.V_weight)
        act = self.V_linear(act) # 512 -> 128

        # act = jnp.asarray(act, self._dtype)
        return act
    
    def _init_parameter_af2(self):
        '''init parameter'''
        # self.W_weight = self.param("W_weight", lecun_normal(),
        #                            [self.hidden_dim*2, self.input_dim], jnp.float32) ## Liyh: need to check this Zhenyu: checked
        # self.V_weight = self.param("V_weight", lecun_normal(),
        #                            [self.output_dim, self.hidden_dim], jnp.float32)
        self.W_linear = nn.Dense(features=self.hidden_dim*2, kernel_init=nn.initializers.lecun_normal(),
                                 use_bias=False, dtype=self._dtype, param_dtype=jnp.float32)
        self.V_linear = nn.Dense(features=self.output_dim, kernel_init=nn.initializers.lecun_normal(),
                                 use_bias=False, dtype=self._dtype, param_dtype=jnp.float32)
                                 
    
    def _init_parameter(self):
        '''init parameter'''
        # self.W_weight = self.param("W_weight", normal(stddev=0.02),
        #                            [self.hidden_dim*2, self.input_dim], jnp.float32)
        # self.V_weight = self.param("V_weight", normal(stddev=0.02),
        #                            [self.output_dim, self.hidden_dim], jnp.float32)
        self.W_linear = nn.Dense(features=self.hidden_dim*2, kernel_init=nn.initializers.normal(stddev=0.02),
                                 use_bias=False, dtype=self._dtype, param_dtype=jnp.float32)
        self.V_linear = nn.Dense(features=self.output_dim, kernel_init=nn.initializers.normal(stddev=0.02),
                                 use_bias=False, dtype=self._dtype, param_dtype=jnp.float32)
        
        

class Transition(nn.Module):

    global_config: ml_collections.ConfigDict
    input_dim: int
    intermediate_dim: int = None
    output_dim: int = None
    init_sigma: float = 0.02
    init_method: str = "AF2"
    dropout_rate: float = 0.
    norm_method: str = "rmsnorm"
    swish_beta: float = 1.

    def setup(self):
        
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        output_dim = self.input_dim if self.output_dim is None else self.output_dim

        self.pre_ffn = PreNonLinear(
            global_config=self.global_config,
            q_data_dim=self.input_dim,
            operation_list=["LN"],
            dropout_rate=self.dropout_rate,
            norm_method=self.norm_method,
            )
        
        self.ffn = FeedForwardNet(
            global_config=self.global_config,
            input_dim=self.input_dim,
            intermediate_dim=self.intermediate_dim,
            output_dim=output_dim,
            init_sigma=self.init_sigma,
            init_method=self.init_method,
            swish_beta=self.swish_beta,
        )

        self.post_ffn = PostNonLinear(
            global_config=self.global_config,
            o_data_dim=output_dim,
            operation_list=["Dropout"],
            dropout_rate=self.dropout_rate,
            norm_method=self.norm_method,
            execute_residual=True,
            accumulated_scale=1.0,
            )
    
    def __call__(self, input, input_mask=None):
        ln_masks = (input_mask, None, None)
        pre_ffn = self.pre_ffn(q_data=input, attention_masks=ln_masks)
        act = pre_ffn[0]
        ffn = self.ffn(act)
        output, _ = self.post_ffn(input, ffn, input_mask)
        return output

### Outerproduct
# @ZhangJ. 支持batch-wise操作:
class OuterProduct(nn.Module):

    global_config: ml_collections.ConfigDict
    input_dim: int
    output_dim: int
    outerproduct_dim: int
    init_sigma: float = 0.02

    def setup(self):
        
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        
        self.left_projection = nn.Dense(features=self.outerproduct_dim, kernel_init=nn.initializers.lecun_normal(),
                                        use_bias=True, bias_init=nn.initializers.zeros_init(),
                                        param_dtype=jnp.float32, dtype=self._dtype)
        self.right_projection = nn.Dense(features=self.outerproduct_dim, kernel_init=nn.initializers.lecun_normal(),
                                        use_bias=True, bias_init=nn.initializers.zeros_init(),
                                        param_dtype=jnp.float32, dtype=self._dtype)
        self.linear_output = nn.Dense(features=self.output_dim, kernel_init=nn.initializers.zeros_init(),
                                        use_bias=True, bias_init=nn.initializers.zeros_init(),
                                        param_dtype=jnp.float32, dtype=self._dtype)        
                
    def __call__(self, seq_act, seq_mask):
        ### act:(B,Q,c); msa_mask:(B,Q); cutoff_value:(B,Q).

        act = seq_act # (B,Q,c)
        mask = jnp.expand_dims(seq_mask, -1) # (B,Q,1)

        act_shape = act.shape
        out_shape = act_shape[:-1] + (-1,)
        
        act = jnp.reshape(act, (-1, act_shape[-1])) # (...,c)
        
        # (B*Nres1,C1==32) -> (B,Nres1,C1):
        left_act = mask * jnp.reshape(self.left_projection(act), out_shape)
        # left_act = mask * jnp.reshape(jnp.add(jnp.matmul(act, self.left_projection_weight.T), self.left_projection_bias), out_shape)
        # (B*Nres2,C2==32) -> (B,Nres2,C2):
        # right_act = mask * jnp.reshape(jnp.add(jnp.matmul(act, self.right_projection_weight.T), self.right_projection_bias), out_shape)
        right_act = mask * jnp.reshape(self.right_projection(act), out_shape)
        b, c = left_act.shape # Nres1,C1
        d, e = right_act.shape # Nres2,C2

        # (B,Nres1,C1) -> (B,Nseq1=1,Nres1,C1) -> (B,C1,Nres1,Nseq1) -> (B,C1*Nres1,Nseq1):
        left_act = jnp.reshape(jnp.transpose(jnp.expand_dims(left_act, 1), (2, 1, 0)), (-1, 1))
        # (B,Nres2*C2,Nseq2=1):
        right_act = jnp.reshape(right_act, (-1, 1))

        # (B,C1*Nres1,Nseq1)@(B,Nres2*C2,Nseq2=1)->(B,C1*Nres1,Nres2*C2):
        act = jnp.matmul(left_act.astype(self._dtype), right_act.T.astype(self._dtype))
        # -> (B,C1,Nres1,Nres2,C2)->(B,Nres1,Nres2,C1,C2)->(B,Nres1,Nres2,C1*C2):
        act = jnp.reshape(jnp.transpose(jnp.reshape(act, (c, b, d, e)), (1, 2, 0, 3)), (b, d, c * e))

        # (B,Nres1,Nres2,C1*C2):
        act_shape_update = act.shape
        # (B*Nres1*Nres2,C1*C2):
        act = jnp.reshape(act, (-1, act_shape_update[-1]))
        # (B,Nres1,Nres2,C):
        act = jnp.reshape(self.linear_output(act), act_shape_update[:-1]+(-1,))
        # act = jnp.reshape(jnp.add(jnp.matmul(act, self.linear_output_weight.T), self.linear_output_bias), act_shape_update[:-1]+(-1,)).astype(self._dtype)
        return act
