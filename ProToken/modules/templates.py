"""Batch Template Embedding"""
import jax
import jax.numpy as jnp

from flax import linen as nn

from .transformer_blocks import Transition
from common.residue_constants import atom_order
from common.multimer_block import multimer_rigids_get_unit_vector
from common.config_load import Config
from .basic import ActFuncWrapper
import ml_collections


class TemplatePairStack(nn.Module):
    '''multimer template pair stack'''

    global_config: ml_collections.ConfigDict
    input_dim: int
    intermediate_dim: int = None
    output_dim: int = None
    init_sigma: float = 0.02
    init_method: str = "AF2"
    dropout_rate: float = 0.
    norm_method: str = "rmsnorm"

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        
        output_dim = self.input_dim if self.output_dim is None else self.output_dim

        self.pair_transition = Transition(global_config=self.global_config,
                                          input_dim=self.input_dim, # 128
                                          intermediate_dim=self.intermediate_dim,
                                          output_dim=output_dim,
                                          init_sigma=self.init_sigma,
                                          init_method=self.init_method,
                                          dropout_rate=self.dropout_rate,
                                          norm_method=self.norm_method,
                                          swish_beta=1.,)
        
    def __call__(self, pair_act, pair_mask):
        # pair_act:(Nres,Nres,128); pair_mask:(Nres,Nres)
        act_shape = pair_act.shape
        flatten_shape = (act_shape[0], -1, act_shape[-1])
        act = jnp.reshape(pair_act, flatten_shape)
        mask = jnp.reshape(pair_mask, flatten_shape[:-1])

        act = self.pair_transition(act, mask)
        updated_pair_act = jnp.reshape(act, act_shape) # (Nres,Nres,128)
        return updated_pair_act

class FlashSingleTemplateEmbedding(nn.Module):
    '''multimer single template embedding'''

    global_config: ml_collections.ConfigDict
    num_channels: int
    num_block: int
    init_sigma: float = 0.02
    init_method: str = "AF2"
    dropout_rate: float = 0.
    norm_method: str = "rmsnorm"

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout
        self.norm_small = self.global_config.norm_small
        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.template_mask_2d_temp_dense = nn.Dense(features=self.num_channels, # 1 -> 128
                                                    kernel_init=nn.initializers.lecun_normal(), 
                                                    dtype=self._dtype, param_dtype=jnp.float32)
        self.unit_vector_0 = nn.Dense(features=self.num_channels, # 1 -> 128
                                      kernel_init=nn.initializers.lecun_normal(), 
                                      dtype=self._dtype, param_dtype=jnp.float32)
        self.unit_vector_1 = nn.Dense(features=self.num_channels, # 1 -> 128
                                      kernel_init=nn.initializers.lecun_normal(), 
                                      dtype=self._dtype, param_dtype=jnp.float32)
        self.unit_vector_2 = nn.Dense(features=self.num_channels, # 1 -> 128
                                      kernel_init=nn.initializers.lecun_normal(), 
                                      dtype=self._dtype, param_dtype=jnp.float32)
        
        self.backbone_mask_2d_dense = nn.Dense(features=self.num_channels, # 1 -> 128
                                               kernel_init=nn.initializers.lecun_normal(), 
                                               dtype=self._dtype, param_dtype=jnp.float32)
        
        self.embedding2d = nn.Dense(features=self.num_channels, # 128 -> 128
                                    kernel_init=nn.initializers.lecun_normal(), 
                                    dtype=self._dtype, param_dtype=jnp.float32)
        
        template_layers = []
        for _ in range(self.num_block):
            template_pair_stack_block = TemplatePairStack(global_config=self.global_config,
                                                          input_dim=self.num_channels, # 128
                                                          init_sigma=self.init_sigma,
                                                          init_method=self.init_method,
                                                          dropout_rate=self.dropout_rate,
                                                          norm_method=self.norm_method,)
            template_layers.append(template_pair_stack_block)
        self.template_pair_stack = template_layers
        
        self.n, self.ca, self.c = [atom_order[a] for a in ('N', 'CA', 'C')]

        self.output_layer_norm = ActFuncWrapper(nn.LayerNorm(epsilon=self.norm_small, dtype=self._dtype, param_dtype=jnp.float32)) # 128 dim

    def __call__(self, pair_activations, # [Nres, Nres, 128]
                       template_all_atom_positions, # [Nres, 37, 3]
                       template_all_atom_mask,  # [Nres, 37]
                       template_pseudo_beta_mask, # [Nres,]
                       padding_mask_2d): # [Nres, Nres]

        template_mask_2d_temp = jnp.expand_dims(template_pseudo_beta_mask, -1) * \
                                jnp.expand_dims(template_pseudo_beta_mask, -2)

        act_tmp = self.template_mask_2d_temp_dense((jnp.expand_dims(template_mask_2d_temp, -1)))

        backbone_mask = template_all_atom_mask[:, self.n] * \
                        template_all_atom_mask[:, self.ca] * \
                        template_all_atom_mask[:, self.c]
        
        # zhenyu: use fp32 from line122 to line132 for stability
        template_all_atom_positions_fp32 = jnp.asarray(template_all_atom_positions, jnp.float32)
        unit_vector = multimer_rigids_get_unit_vector(template_all_atom_positions_fp32[:, self.n],
                                                      template_all_atom_positions_fp32[:, self.ca],
                                                      template_all_atom_positions_fp32[:, self.c])

        backbone_mask_2d = jnp.expand_dims(backbone_mask, -1) * jnp.expand_dims(backbone_mask, -2)

        unit_vector = (jnp.expand_dims(backbone_mask_2d * unit_vector[0], -1).astype(self._dtype),
                       jnp.expand_dims(backbone_mask_2d * unit_vector[1], -1).astype(self._dtype),
                       jnp.expand_dims(backbone_mask_2d * unit_vector[2], -1).astype(self._dtype)) # return to self._dtype
        
        act_tmp += self.unit_vector_0(unit_vector[0])
        act_tmp += self.unit_vector_1(unit_vector[1])
        act_tmp += self.unit_vector_2(unit_vector[2])
        act_tmp += self.backbone_mask_2d_dense(jnp.expand_dims(backbone_mask_2d, -1))
        act_tmp += self.embedding2d(pair_activations)
        
        act_output = act_tmp
        for j in range(self.num_block):
            act_output = self.template_pair_stack[j](act_output, padding_mask_2d) # [Nres, Nres, 128], [Nres, Nres] -> [Nres, Nres, 128]
        
        act_output = self.output_layer_norm(act_output)

        return act_output # (Nres,Nres,128)