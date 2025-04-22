### Implement of backbone fape, copy from AF2 code (multimer)

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

Float = Union[float, jnp.ndarray]

# Python-version-specific alias (Python 2: unicode; Python 3: str)
Text = str

from model import geometry, utils
from model.geometry import utils as geometry_utils
from common.config_load import Config
import functools

def backbone_loss_affine(gt_rigid_affine: jnp.ndarray, # (..., 7)
                  gt_frames_mask: jnp.ndarray,
                  gt_positions_mask: jnp.ndarray,
                  target_rigid_affine: jnp.ndarray, # (..., 7)
                  config: Config,
                  no_clamp_mask: jnp.ndarray,
                  pair_mask: jnp.ndarray,
                  ) -> Tuple[Float, jnp.ndarray]:
    gt_rigid_transaltion = geometry.Vec3Array.from_array(gt_rigid_affine[..., 4:])
    gt_rigid_rotation = geometry.Rot3Array.from_quaternion(
        *geometry_utils.unstack(gt_rigid_affine[..., :4], axis=-1)
    )
    gt_rigid = geometry.Rigid3Array(gt_rigid_rotation, gt_rigid_transaltion)
    
    target_rigid_transaltion = geometry.Vec3Array.from_array(target_rigid_affine[..., 4:])
    target_rigid_rotation = geometry.Rot3Array.from_quaternion(
        *geometry_utils.unstack(target_rigid_affine[..., :4], axis=-1)
    )
    target_rigid = geometry.Rigid3Array(target_rigid_rotation, target_rigid_transaltion)
    
    return backbone_loss(gt_rigid, gt_frames_mask, gt_positions_mask, target_rigid, config, no_clamp_mask, pair_mask)

def backbone_loss_affine_with_weights(gt_rigid_affine: jnp.ndarray, # (..., 7)
                  gt_frames_mask: jnp.ndarray,
                  gt_positions_mask: jnp.ndarray,
                  target_rigid_affine: jnp.ndarray, # (..., 7)
                  config: Config,
                  no_clamp_mask: jnp.ndarray,
                  pair_mask: jnp.ndarray,
                  IPA_weights: jnp.ndarray,
                  ) -> Tuple[Float, jnp.ndarray]:
    gt_rigid_transaltion = geometry.Vec3Array.from_array(gt_rigid_affine[..., 4:])
    gt_rigid_rotation = geometry.Rot3Array.from_quaternion(
        *geometry_utils.unstack(gt_rigid_affine[..., :4], axis=-1)
    )
    gt_rigid = geometry.Rigid3Array(gt_rigid_rotation, gt_rigid_transaltion)
    
    target_rigid_transaltion = geometry.Vec3Array.from_array(target_rigid_affine[..., 4:])
    target_rigid_rotation = geometry.Rot3Array.from_quaternion(
        *geometry_utils.unstack(target_rigid_affine[..., :4], axis=-1)
    )
    target_rigid = geometry.Rigid3Array(target_rigid_rotation, target_rigid_transaltion)
    
    return backbone_loss_with_weights(gt_rigid, gt_frames_mask, gt_positions_mask, target_rigid, config, no_clamp_mask, pair_mask, IPA_weights)


def backbone_loss_array(gt_rigid_array: jnp.ndarray, # (..., 3, 4)
                  gt_frames_mask: jnp.ndarray,
                  gt_positions_mask: jnp.ndarray,
                  target_rigid_array: jnp.ndarray, # (..., 3, 4)
                  config: Config,
                  no_clamp_mask: jnp.ndarray,
                  pair_mask: jnp.ndarray,
                  ) -> Tuple[Float, jnp.ndarray]:
    
    gt_rigid = geometry.Rigid3Array.from_array(gt_rigid_array)
    target_rigid = geometry.Rigid3Array.from_array(target_rigid_array)
    return backbone_loss(gt_rigid, gt_frames_mask, gt_positions_mask, target_rigid, config, no_clamp_mask, pair_mask)

def backbone_loss(gt_rigid: geometry.Rigid3Array,
                  gt_frames_mask: jnp.ndarray,
                  gt_positions_mask: jnp.ndarray,
                  target_rigid: geometry.Rigid3Array,
                  config: Config,
                  no_clamp_mask: jnp.ndarray,
                  pair_mask: jnp.ndarray
                  ) -> Tuple[Float, jnp.ndarray]:
    """Backbone FAPE Loss."""
    loss_fn = functools.partial(
        frame_aligned_point_error,
        l1_clamp_distance=config.fape.atom_clamp_distance,
        length_scale=config.fape.loss_unit_distance)

    loss_fn = jax.vmap(loss_fn, (0, None, None, 0, None, None, 0, None))
    fape, fape_no_clamp = loss_fn(target_rigid, gt_rigid, gt_frames_mask,
                    target_rigid.translation, gt_rigid.translation,
                    gt_positions_mask, no_clamp_mask, pair_mask)

    return jnp.mean(fape), fape[-1], fape_no_clamp[-1]

def backbone_loss_with_weights(gt_rigid: geometry.Rigid3Array,
                  gt_frames_mask: jnp.ndarray,
                  gt_positions_mask: jnp.ndarray,
                  target_rigid: geometry.Rigid3Array,
                  config: dict,
                  no_clamp_mask: jnp.ndarray,
                  pair_mask: jnp.ndarray,
                  IPA_weights: jnp.ndarray
                  ) -> Tuple[Float, jnp.ndarray]:
    """Backbone FAPE Loss."""
    loss_fn = functools.partial(
        frame_aligned_point_error,
        l1_clamp_min_distance=config["fape"]["atom_clamp_min_distance"], 
        l1_clamp_distance=config["fape"]["atom_clamp_distance"],
        length_scale=config["fape"]["loss_unit_distance"])

    loss_fn = jax.vmap(loss_fn, (0, None, None, 0, None, None, 0, None))
    fape, fape_no_clamp = loss_fn(target_rigid, gt_rigid, gt_frames_mask,
                    target_rigid.translation, gt_rigid.translation,
                    gt_positions_mask, no_clamp_mask, pair_mask)

    return jnp.sum(fape * IPA_weights), fape[-1], fape_no_clamp[-1]


def frame_aligned_point_error(
    pred_frames: geometry.Rigid3Array,  # shape (num_frames)
    target_frames: geometry.Rigid3Array,  # shape (num_frames)
    frames_mask: jnp.ndarray,  # shape (num_frames)
    pred_positions: geometry.Vec3Array,  # shape (num_positions)
    target_positions: geometry.Vec3Array,  # shape (num_positions)
    positions_mask: jnp.ndarray,  # shape (num_positions)
    no_clamp_mask: jnp.ndarray, # shape (num_frames)
    pair_mask: jnp.ndarray,  # shape (num_frames, num_posiitons)
    l1_clamp_distance: float,
    l1_clamp_min_distance=0.0, 
    length_scale=20.,
    epsilon=1e-4) -> jnp.ndarray:  # shape ()
    """Measure point error under different alignements.

    Computes error between two structures with B points
    under A alignments derived form the given pairs of frames.
    Args:
    pred_frames: num_frames reference frames for 'pred_positions'.
    target_frames: num_frames reference frames for 'target_positions'.
    frames_mask: Mask for frame pairs to use.
    pred_positions: num_positions predicted positions of the structure.
    target_positions: num_positions target positions of the structure.
    positions_mask: Mask on which positions to score.
    pair_mask: A (num_frames, num_positions) mask to use in the loss, useful
        for separating intra from inter chain losses.
    l1_clamp_distance: Distance cutoff on error beyond which gradients will
        be zero.
    length_scale: length scale to divide loss by.
    epsilon: small value used to regularize denominator for masked average.
    Returns:
    Masked Frame aligned point error.
    """
    # For now we do not allow any batch dimensions.
    assert len(pred_frames.rotation.shape) == 1
    assert len(target_frames.rotation.shape) == 1
    assert frames_mask.ndim == 1
    assert pred_positions.x.ndim == 1
    assert target_positions.x.ndim == 1
    assert positions_mask.ndim == 1

    # Compute array of predicted positions in the predicted frames.
    # geometry.Vec3Array (num_frames, num_positions)
    local_pred_pos = pred_frames[:, None].inverse().apply_to_point(
        pred_positions[None, :])

    # Compute array of target positions in the target frames.
    # geometry.Vec3Array (num_frames, num_positions)
    local_target_pos = target_frames[:, None].inverse().apply_to_point(
        target_positions[None, :])

    # Compute errors between the structures.
    # jnp.ndarray (num_frames, num_positions)
    error_dist = geometry.euclidean_distance(local_pred_pos, local_target_pos,
                                            epsilon)

    clipped_error_dist = jnp.clip(error_dist, l1_clamp_min_distance, l1_clamp_distance)
    clipped_error_dist = clipped_error_dist * no_clamp_mask[:, None] \
                        + error_dist * (1 - no_clamp_mask[:, None])

    normed_error = clipped_error_dist / length_scale
    normed_error *= jnp.expand_dims(frames_mask, axis=-1)
    normed_error *= jnp.expand_dims(positions_mask, axis=-2)
    
    # no clamped loss 
    normed_error_no_clamp = error_dist / length_scale
    normed_error_no_clamp *= jnp.expand_dims(frames_mask, axis=-1)
    normed_error_no_clamp *= jnp.expand_dims(positions_mask, axis=-2)
    
    if pair_mask is not None:
        normed_error *= pair_mask
        normed_error_no_clamp *= pair_mask

    mask = (jnp.expand_dims(frames_mask, axis=-1) *
            jnp.expand_dims(positions_mask, axis=-2))
    if pair_mask is not None:
        mask *= pair_mask
    normalization_factor = jnp.sum(mask, axis=(-1, -2))
    return (jnp.sum(normed_error, axis=(-2, -1)) / (epsilon + normalization_factor)), \
            (jnp.sum(normed_error_no_clamp, axis=(-2, -1)) / (epsilon + normalization_factor))