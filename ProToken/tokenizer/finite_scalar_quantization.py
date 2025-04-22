#### copy from codes in FSQ paper

import itertools
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

Codeword = jax.Array
Indices = jax.Array

def round_ste(z):
    """Round with straight through gradients."""
    zhat = jnp.round(z)
    return z + jax.lax.stop_gradient(zhat - z)

class FSQTokenizer():
    """Quantizer."""

    def __init__(self, config):
        self.config = config
        self._levels = self.config.levels
        self._eps = self.config.eps
        self._levels_np = jnp.array(np.asarray(self._levels))
        self._basis = jnp.array(
            np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)
        )

        self._implicit_codebook = jnp.array(
            self.indexes_to_codes(np.arange(self.codebook_size))
        )
        
        self.lower_bound = -(self._levels_np // 2)
        self.upper_bound = self._levels_np // 2 - (1 - self._levels_np % 2)
        
    def __call__(self, x, quantize=True):
        return self.quantize(x, quantize)

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: jax.Array) -> jax.Array:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def quantize(self, z: jax.Array, quantize=True) -> Codeword:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = self.bound(z)
        if (quantize):
            quantized = round_ste(quantized)

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: Codeword) -> Indices:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices: Indices) -> Codeword:
        """Inverse of `indexes_to_codes`."""
        indices = indices[..., jnp.newaxis]
        codes_non_centered = jnp.mod(
            jnp.floor_divide(indices, self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(codes_non_centered)
    
    def count_codes(self, zhat: jax.Array, mask: jax.Array) -> jax.Array:
        """Returns the code usage of each code in the codebook."""
        indices = self.codes_to_indexes(zhat)
        return jnp.bincount(indices.flatten(), weights=mask.flatten(), length=self.codebook_size).astype(jnp.float32)
    
    def get_act_bound(self):
        fsq_lower_bound = jnp.array(self.lower_bound, dtype=jnp.float32)
        fsq_upper_bound = jnp.array(self.upper_bound, dtype=jnp.float32)
        
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        
        fsq_act_lower_bound = jnp.arctanh(
            (fsq_lower_bound + 0.25 + offset) / half_l) - shift
        fsq_act_upper_bound = jnp.arctanh(
            (fsq_upper_bound - 0.25 + offset) / half_l) - shift
        
        return fsq_act_lower_bound, fsq_act_upper_bound