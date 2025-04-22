import jax
import jax.numpy as jnp

# @ZhangJ. 注意, epsilon可以设置成1e-8 (防止FP32的下溢出)；或者在此处临时使用FP64执行运算
def _l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / jnp.sqrt(
        jnp.maximum(jnp.sum(x**2, axis=axis, keepdims=True), epsilon))
        
def square_euclidean_distance(x, y, axis=-1, normalized=False):
    # normalize = True, cos similarity, normalize = False, mse loss
    aggregate_fn = jnp.sum if normalized else jnp.mean
    x2 = aggregate_fn(x**2, axis=axis)
    y2 = aggregate_fn(y**2, axis=axis)
    dot_product = aggregate_fn(x * y, axis=axis)
    return x2 + y2 - 2 * dot_product

def parameter_weight_decay(params):
    """Apply weight decay to parameters."""
    
    loss = jax.tree_util.tree_map(
        lambda p: jnp.mean(
                jnp.square(p.reshape(-1))
            ) if p.ndim == 2 else 0, params)
    loss = jnp.sum(
        jnp.array(jax.tree_util.tree_leaves(loss))
    )
    
    return loss