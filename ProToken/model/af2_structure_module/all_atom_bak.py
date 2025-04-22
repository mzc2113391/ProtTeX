import functools
import numbers
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from common import residue_constants
from model import geometry
from model import prng
from model import utils
from model import common_modules_bak
from model.geometry import utils as geometry_utils
# import haiku as hk
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

def _make_restype_atom37_mask():
    """Mask of which atoms are present for which residue type in atom37."""
    # create the corresponding mask
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
            restype_name = residue_constants.restype_1to3[restype_letter]
            atom_names = residue_constants.residue_atoms[restype_name]
            for atom_name in atom_names:
                atom_type = residue_constants.atom_order[atom_name]
                restype_atom37_mask[restype, atom_type] = 1
    return restype_atom37_mask

def _make_restype_atom14_mask():
    """Mask of which atoms are present for which residue type in atom14."""
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
            atom_names = residue_constants.restype_name_to_atom14_names[
                residue_constants.restype_1to3[rt]]
            restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    restype_atom14_mask.append([0.] * 14)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)
    return restype_atom14_mask

def _make_restype_atom37_to_atom14():
    """Map from atom37 to atom14 per residue type."""
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]]
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types
        ])

    restype_atom37_to_atom14.append([0] * 37)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    return restype_atom37_to_atom14

RESTYPE_ATOM37_MASK = _make_restype_atom37_mask()
RESTYPE_ATOM14_MASK = _make_restype_atom14_mask()
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom37_to_atom14()

def get_atom37_mask(aatype):
    return utils.batched_gather(jnp.asarray(RESTYPE_ATOM37_MASK), aatype)

def get_atom14_mask(aatype):
    return utils.batched_gather(jnp.asarray(RESTYPE_ATOM14_MASK), aatype)

def get_atom37_to_atom14_map(aatype):
    return utils.batched_gather(jnp.asarray(RESTYPE_ATOM37_TO_ATOM14), aatype)

def make_transform_from_reference(
    a_xyz: geometry.Vec3Array,
    b_xyz: geometry.Vec3Array,
    c_xyz: geometry.Vec3Array) -> geometry.Rigid3Array:
    """Returns rotation and translation matrices to convert from reference.

    Note that this method does not take care of symmetries. If you provide the
    coordinates in the non-standard way, the A atom will end up in the negative
    y-axis rather than in the positive y-axis. You need to take care of such
    cases in your code.

    Args:
    a_xyz: A Vec3Array.
    b_xyz: A Vec3Array.
    c_xyz: A Vec3Array.

    Returns:
    A Rigid3Array which, when applied to coordinates in a canonicalized
    reference frame, will give coordinates approximately equal
    the original coordinates (in the global frame).
    """
    rotation = geometry.Rot3Array.from_two_vectors(c_xyz - b_xyz,
                                                    a_xyz - b_xyz)
    return geometry.Rigid3Array(rotation, b_xyz)

def atom14_to_atom37(atom14_data: jnp.ndarray,  # (N, 14, ...)
                     aatype: jnp.ndarray
                    ) -> jnp.ndarray:  # (N, 37, ...)
    """Convert atom14 to atom37 representation."""

    assert len(atom14_data.shape) in [2, 3]
    idx_atom37_to_atom14 = get_atom37_to_atom14_map(aatype)
    atom37_data = utils.batched_gather(
        atom14_data, idx_atom37_to_atom14, batch_dims=1)
    atom37_mask = get_atom37_mask(aatype)
    if len(atom14_data.shape) == 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == 3:
        atom37_data *= atom37_mask[:, :, None].astype(atom37_data.dtype)
    return atom37_data

def torsion_angles_to_frames(
    aatype: jnp.ndarray,  # (N)
    backb_to_global: geometry.Rigid3Array,  # (N)
    torsion_angles_sin_cos: jnp.ndarray  # (N, 7, 2)
) -> geometry.Rigid3Array:  # (N, 8)
    """Compute rigid group frames from torsion angles."""
    
    assert len(aatype.shape) == 1, (
        f'Expected array of rank 1, got array with shape: {aatype.shape}.')
    assert len(backb_to_global.rotation.shape) == 1, (
        f'Expected array of rank 1, got array with shape: '
        f'{backb_to_global.rotation.shape}')
    assert len(torsion_angles_sin_cos.shape) == 3, (
        f'Expected array of rank 3, got array with shape: '
        f'{torsion_angles_sin_cos.shape}')
    assert torsion_angles_sin_cos.shape[1] == 7, (
        f'wrong shape {torsion_angles_sin_cos.shape}')
    assert torsion_angles_sin_cos.shape[2] == 2, (
        f'wrong shape {torsion_angles_sin_cos.shape}')

    # Gather the default frames for all rigid groups.
    # geometry.Rigid3Array with shape (N, 8)
    m = utils.batched_gather(residue_constants.restype_rigid_group_default_frame,
                            aatype)
    default_frames = geometry.Rigid3Array.from_array4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues, = aatype.shape
    sin_angles = jnp.concatenate([jnp.zeros([num_residues, 1]), sin_angles],
                                axis=-1)
    cos_angles = jnp.concatenate([jnp.ones([num_residues, 1]), cos_angles],
                                axis=-1)
    zeros = jnp.zeros_like(sin_angles)
    ones = jnp.ones_like(sin_angles)

    # all_rots are geometry.Rot3Array with shape (N, 8)
    all_rots = geometry.Rot3Array(ones, zeros, zeros,
                                    zeros, cos_angles, -sin_angles,
                                    zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = default_frames.compose_rotation(all_rots)

    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.

    chi1_frame_to_backb = all_frames[:, 4]
    chi2_frame_to_backb = chi1_frame_to_backb @ all_frames[:, 5]
    chi3_frame_to_backb = chi2_frame_to_backb @ all_frames[:, 6]
    chi4_frame_to_backb = chi3_frame_to_backb @ all_frames[:, 7]

    all_frames_to_backb = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x, axis=-1), all_frames[:, 0:5],
        chi2_frame_to_backb[:, None], chi3_frame_to_backb[:, None],
        chi4_frame_to_backb[:, None])

    # Create the global frames.
    # shape (N, 8)
    all_frames_to_global = backb_to_global[:, None] @ all_frames_to_backb

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    aatype: jnp.ndarray,  # (N)
    all_frames_to_global: geometry.Rigid3Array  # (N, 8)
) -> geometry.Vec3Array:  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group."""

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = utils.batched_gather(
        residue_constants.restype_atom14_to_rigid_group, aatype)
    group_mask = jax.nn.one_hot(
        residx_to_group_idx, num_classes=8)  # shape (N, 14, 8)

    # geometry.Rigid3Array with shape (N, 14)
    map_atoms_to_global = jax.tree_util.tree_map(
        lambda x: jnp.sum(x[:, None, :] * group_mask, axis=-1),
        all_frames_to_global)

    # Gather the literature atom positions for each residue.
    # geometry.Vec3Array with shape (N, 14)
    lit_positions = geometry.Vec3Array.from_array(
        utils.batched_gather(
            residue_constants.restype_atom14_rigid_group_positions, aatype))

    # Transform each atom from its local frame to the global frame.
    # geometry.Vec3Array with shape (N, 14)
    pred_positions = map_atoms_to_global.apply_to_point(lit_positions)

    # Mask out non-existing atoms.
    mask = utils.batched_gather(residue_constants.restype_atom14_mask, aatype)
    pred_positions = pred_positions * mask

    return pred_positions


def make_transform_from_reference(
    a_xyz: geometry.Vec3Array,
    b_xyz: geometry.Vec3Array,
    c_xyz: geometry.Vec3Array) -> geometry.Rigid3Array:
    """Returns rotation and translation matrices to convert from reference.

    Note that this method does not take care of symmetries. If you provide the
    coordinates in the non-standard way, the A atom will end up in the negative
    y-axis rather than in the positive y-axis. You need to take care of such
    cases in your code.

    Args:
        a_xyz: A Vec3Array.
        b_xyz: A Vec3Array.
        c_xyz: A Vec3Array.

    Returns:
        A Rigid3Array which, when applied to coordinates in a canonicalized
        reference frame, will give coordinates approximately equal
        the original coordinates (in the global frame).
    """
    rotation = geometry.Rot3Array.from_two_vectors(c_xyz - b_xyz,
                                                    a_xyz - b_xyz)
    return geometry.Rigid3Array(rotation, b_xyz)