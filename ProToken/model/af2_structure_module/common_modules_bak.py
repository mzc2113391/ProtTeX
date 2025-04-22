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

"""A collection of common Haiku modules for use in protein folding."""
import numbers
from typing import Union, Sequence

# import haiku as hk
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978,
                                            dtype=np.float32)


def get_initializer_scale(initializer_name, input_shape):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == 'zeros':
        w_init = nn.initializers.constant(0.0)
    else:
        # fan-in scaling
        scale = 1.
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == 'relu':
            scale *= 2.

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = nn.initializers.truncated_normal(stddev=stddev)
    
    return w_init


class Linear(nn.Module):
    """Protein folding specific Linear module.
    This differs from the standard Haiku Linear in a few ways:
    * It supports inputs and outputs of arbitrary rank
    * Initializers are specified by strings
    """
    num_output: Union[int, Sequence[int]]
    initializer: str = 'linear'
    num_input_dims: int = 1
    use_bias: bool = True
    bias_init: float = 0.
    precision: float = None
    name: str = "linear"
    
    @nn.compact
    def __call__(self, inputs):
        """Connects Module.
        Args:
            inputs: Tensor with at least num_input_dims dimensions.
        Returns:
            output of shape [...] + num_output.
        """
        #### Initialization
        if isinstance(self.num_output, numbers.Integral):
            output_shape = (self.num_output,)
        else:
            output_shape = tuple(self.num_output)
        num_output_dims = len(output_shape)
        
        #### __call__ function in hk.module
        num_input_dims = self.num_input_dims

        if self.num_input_dims > 0:
            in_shape = inputs.shape[-self.num_input_dims:]
        else:
            in_shape = ()

        weight_init = get_initializer_scale(self.initializer, in_shape)

        in_letters = 'abcde'[:self.num_input_dims]
        out_letters = 'hijkl'[:num_output_dims]

        weight_shape = in_shape + output_shape
        # weights = hk.get_parameter('weights', weight_shape, inputs.dtype,
        #                             weight_init)
        weights = self.param("weights", weight_init, weight_shape, inputs.dtype)

        equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

        output = jnp.einsum(equation, inputs, weights, precision=self.precision)

        if self.use_bias:
            # bias = hk.get_parameter('bias', self.output_shape, inputs.dtype,
            #                         hk.initializers.Constant(self.bias_init))
            bias = self.param("bias", nn.initializers.constant(self.bias_init), output_shape, inputs.dtype)
            output += bias

        return output

class LayerNorm(nn.Module):
    """LayerNorm module.
    Equivalent to hk.LayerNorm but with different parameter shapes: they are
    always vectors rather than possibly higher-rank tensors. This makes it easier
    to change the layout whilst keep the model weight-compatible.
    """
    axis: int
    create_scale: bool
    create_offset: bool
    eps: float = 1e-5
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_fast_variance: bool = False
    name: str = "layer_norm"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        #### Initialize
        layer_norm = nn.LayerNorm(
            eps = self.eps,
            dtype = jnp.float32,
            param_dtype = jnp.float32, 
            use_bias = self.create_offset,
            use_scale = self.create_scale,
            scale_init = self.scale_init,
            bias_init = self.bias_init,
            use_fast_variance = self.use_fast_variance,
        )
        
        #### Calculate
        is_bf16 = (x.dtype == jnp.bfloat16)
        if is_bf16:
            x = x.astype(jnp.float32)
            
        out = layer_norm(x)

        if is_bf16:
            out = out.astype(jnp.bfloat16)

        return out