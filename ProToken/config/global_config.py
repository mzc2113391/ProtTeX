"""Global config"""
import ml_collections
import copy

# Global config
GLOBAL_CONFIG = {'sharding': True,
                 'bf16_flag': True,
                 'jax_small': 1e-5,
                 'norm_small': 1e-5,
                 'remat_flag': True,
                 'use_dropout': False,
                 'global_dropout_rate': 0.00,}
GLOBAL_CONFIG = ml_collections.ConfigDict(copy.deepcopy(GLOBAL_CONFIG))