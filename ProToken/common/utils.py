"""utils module"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import common_utils
from . import geometry

def dgram_from_positions(positions, num_bins, min_bin, max_bin, ret_type):
    """Compute distogram from amino acid positions.

    Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
    everything larger than `max_bin`.

    Returns:
    Distogram with the specified number of bins.
    """

    def squared_difference(x, y):
        return jnp.square(x - y)

    lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
    lower_breaks = jnp.square(lower_breaks)
    upper_breaks = jnp.concatenate([lower_breaks[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1)
    dist2 = jnp.sum(squared_difference(jnp.expand_dims(positions, axis=-2),
                                       jnp.expand_dims(positions, axis=-3)), axis=-1, keepdims=True)
    dgram = ((dist2 > lower_breaks).astype(ret_type) * (dist2 < upper_breaks).astype(ret_type))
    return dgram

def dgram_from_positions_soft(positions, num_bins, min_bin, max_bin, temp=2.0):
  '''soft positions to dgram converter'''
  lower_breaks = jnp.append(-1e8,jnp.linspace(min_bin, max_bin, num_bins))
  upper_breaks = jnp.append(lower_breaks[1:],1e8)
  dist = jnp.sqrt(jnp.square(positions[...,:,None,:] - positions[...,None,:,:]).sum(-1,keepdims=True) + 1e-8)
  o = jax.nn.sigmoid((dist - lower_breaks)/temp) * jax.nn.sigmoid((upper_breaks - dist)/temp)
  o = o/(o.sum(-1,keepdims=True) + 1e-8)
  return o[...,1:]


def batch_rigids_from_tensor4x4(m):
    """Construct Rigids object from an 4x4 array.

    Here the 4x4 is representing the transformation in homogeneous coordinates.

    Args:
    m: Array representing transformations in homogeneous coordinates.
    Returns:
    Rigids object corresponding to transformations m
    """
    rotation = (m[:, :, :, 0, 0], m[:, :, :, 0, 1], m[:, :, :, 0, 2],
                m[:, :, :, 1, 0], m[:, :, :, 1, 1], m[:, :, :, 1, 2],
                m[:, :, :, 2, 0], m[:, :, :, 2, 1], m[:, :, :, 2, 2])
    trans = (m[:, :, :, 0, 3], m[:, :, :, 1, 3], m[:, :, :, 2, 3])
    rigid = (rotation, trans)
    return rigid

def batch_torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos, restype_rigid_group_default_frame):
    """Compute rigid group frames from torsion angles."""

    # Gather the default frames for all rigid groups.
    m = jnp.take(restype_rigid_group_default_frame, aatype, 0) # [num_batch, seq_len, 8, 4, 4]
    default_frames = batch_rigids_from_tensor4x4(m) # [num_batch, seq_len, 8]
    
    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0] # [num_batch, seq_len, 7]
    cos_angles = torsion_angles_sin_cos[..., 1] # [num_batch, seq_len, 7]

    # insert zero rotation for backbone group.
    num_batch, num_residues, = aatype.shape
    sin_angles = jnp.concatenate([jnp.zeros([num_batch, num_residues, 1]), sin_angles], axis=-1)
    cos_angles = jnp.concatenate([jnp.ones([num_batch, num_residues, 1]), cos_angles], axis=-1)
    zeros = jnp.zeros_like(sin_angles)
    ones = jnp.ones_like(sin_angles)  # [num_batch, seq_len, 8]
    

    all_rots = (ones, zeros, zeros,
                zeros, cos_angles, -sin_angles,
                zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = geometry.rigids_mul_rots(default_frames, all_rots)

    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    chi2_frame_to_frame = ((all_frames[0][0][:, :, 5], all_frames[0][1][:, :, 5], all_frames[0][2][:, :, 5],
                            all_frames[0][3][:, :, 5], all_frames[0][4][:, :, 5], all_frames[0][5][:, :, 5],
                            all_frames[0][6][:, :, 5], all_frames[0][7][:, :, 5], all_frames[0][8][:, :, 5]),
                           (all_frames[1][0][:, :, 5], all_frames[1][1][:, :, 5], all_frames[1][2][:, :, 5]))
    chi3_frame_to_frame = ((all_frames[0][0][:, :, 6], all_frames[0][1][:, :, 6], all_frames[0][2][:, :, 6],
                            all_frames[0][3][:, :, 6], all_frames[0][4][:, :, 6], all_frames[0][5][:, :, 6],
                            all_frames[0][6][:, :, 6], all_frames[0][7][:, :, 6], all_frames[0][8][:, :, 6]),
                           (all_frames[1][0][:, :, 6], all_frames[1][1][:, :, 6], all_frames[1][2][:, :, 6]))

    chi4_frame_to_frame = ((all_frames[0][0][:, :, 7], all_frames[0][1][:, :, 7], all_frames[0][2][:, :, 7],
                            all_frames[0][3][:, :, 7], all_frames[0][4][:, :, 7], all_frames[0][5][:, :, 7],
                            all_frames[0][6][:, :, 7], all_frames[0][7][:, :, 7], all_frames[0][8][:, :, 7]),
                           (all_frames[1][0][:, :, 7], all_frames[1][1][:, :, 7], all_frames[1][2][:, :, 7]))

    chi1_frame_to_backb = ((all_frames[0][0][:, :, 4], all_frames[0][1][:, :, 4], all_frames[0][2][:, :, 4],
                            all_frames[0][3][:, :, 4], all_frames[0][4][:, :, 4], all_frames[0][5][:, :, 4],
                            all_frames[0][6][:, :, 4], all_frames[0][7][:, :, 4], all_frames[0][8][:, :, 4]),
                           (all_frames[1][0][:, :, 4], all_frames[1][1][:, :, 4], all_frames[1][2][:, :, 4]))

    chi2_frame_to_backb = geometry.rigids_mul_rigids(chi1_frame_to_backb, chi2_frame_to_frame)
    chi3_frame_to_backb = geometry.rigids_mul_rigids(chi2_frame_to_backb, chi3_frame_to_frame)
    chi4_frame_to_backb = geometry.rigids_mul_rigids(chi3_frame_to_backb, chi4_frame_to_frame)

    # Recombine them to a Rigids with shape (N, 8).
    all_frames_to_backb = batch_rigids_concate_all(all_frames, chi2_frame_to_backb,
                                                   chi3_frame_to_backb, chi4_frame_to_backb)

    backb_to_global = (geometry.rots_expand_dims(backb_to_global[0], -1),
                       geometry.vecs_expand_dims(backb_to_global[1], -1))
    # Create the global frames.
    all_frames_to_global = geometry.rigids_mul_rigids(backb_to_global, all_frames_to_backb)
    return all_frames_to_global

def batch_rigids_concate_all(xall, x5, x6, x7):
    """rigids concate all."""
    x5 = (geometry.rots_expand_dims(x5[0], -1), geometry.vecs_expand_dims(x5[1], -1))
    x6 = (geometry.rots_expand_dims(x6[0], -1), geometry.vecs_expand_dims(x6[1], -1))
    x7 = (geometry.rots_expand_dims(x7[0], -1), geometry.vecs_expand_dims(x7[1], -1))
    xall_rot = xall[0]
    xall_rot_slice = []
    for val in xall_rot:
        xall_rot_slice.append(val[:, :, 0:5])
    xall_trans = xall[1]
    xall_trans_slice = []
    for val in xall_trans:
        xall_trans_slice.append(val[:, :, 0:5])
    xall = (xall_rot_slice, xall_trans_slice)
    res_rot = []
    for i in range(9):
        res_rot.append(jnp.concatenate((xall[0][i], x5[0][i], x6[0][i], x7[0][i]), axis=-1))
    res_trans = []
    for i in range(3):
        res_trans.append(jnp.concatenate((xall[1][i], x5[1][i], x6[1][i], x7[1][i]), axis=-1))
    return (res_rot, res_trans)

def batch_frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global, restype_atom14_to_rigid_group,
                                                  restype_atom14_rigid_group_positions, restype_atom14_mask):  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

    Args:
    aatype: aatype for each residue.
    all_frames_to_global: All per residue coordinate frames.
    Returns:
    Positions of all atom coordinates in global frame.
    """

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = jnp.take(restype_atom14_to_rigid_group, aatype, 0)
    group_mask = common_utils.onehot(residx_to_group_idx, 8)
    # Rigids with shape (N, 14)
    map_atoms_to_global = batch_map_atoms_to_global_func(all_frames_to_global, group_mask)
    # Gather the literature atom positions for each residue.
    # Vecs with shape (N, 14)
    lit_positions = geometry.vecs_from_tensor(jnp.take(restype_atom14_rigid_group_positions, aatype, 0))

    # Transform each atom from its local frame to the global frame.
    # Vecs with shape (N, 14)
    pred_positions = geometry.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    mask = jnp.take(restype_atom14_mask, aatype, 0)
    pred_positions = geometry.vecs_scale(pred_positions, mask)
    return pred_positions


def batch_map_atoms_to_global_func(all_frames, group_mask):
    """map atoms to global."""
    all_frames_rot = all_frames[0]
    all_frames_trans = all_frames[1]
    rot = geometry.rots_scale(geometry.rots_expand_dims(all_frames_rot, 2), group_mask)
    res_rot = []
    for val in rot:
        res_rot.append(jnp.sum(val, axis=-1))
    trans = geometry.vecs_scale(geometry.vecs_expand_dims(all_frames_trans, 2), group_mask)
    res_trans = []
    for val in trans:
        res_trans.append(jnp.sum(val, axis=-1))
    return (res_rot, res_trans)




def torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos, restype_rigid_group_default_frame):
    """Compute rigid group frames from torsion angles."""

    # Gather the default frames for all rigid groups.
    m = jnp.take(restype_rigid_group_default_frame, aatype, 0)

    default_frames = rigids_from_tensor4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues, = aatype.shape
    sin_angles = jnp.concatenate([jnp.zeros([num_residues, 1]), sin_angles], axis=-1)
    cos_angles = jnp.concatenate([jnp.ones([num_residues, 1]), cos_angles], axis=-1)
    zeros = jnp.zeros_like(sin_angles)
    ones = jnp.ones_like(sin_angles)

    all_rots = (ones, zeros, zeros,
                zeros, cos_angles, -sin_angles,
                zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = geometry.rigids_mul_rots(default_frames, all_rots)
    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    chi2_frame_to_frame = ((all_frames[0][0][:, 5], all_frames[0][1][:, 5], all_frames[0][2][:, 5],
                            all_frames[0][3][:, 5], all_frames[0][4][:, 5], all_frames[0][5][:, 5],
                            all_frames[0][6][:, 5], all_frames[0][7][:, 5], all_frames[0][8][:, 5]),
                           (all_frames[1][0][:, 5], all_frames[1][1][:, 5], all_frames[1][2][:, 5]))
    chi3_frame_to_frame = ((all_frames[0][0][:, 6], all_frames[0][1][:, 6], all_frames[0][2][:, 6],
                            all_frames[0][3][:, 6], all_frames[0][4][:, 6], all_frames[0][5][:, 6],
                            all_frames[0][6][:, 6], all_frames[0][7][:, 6], all_frames[0][8][:, 6]),
                           (all_frames[1][0][:, 6], all_frames[1][1][:, 6], all_frames[1][2][:, 6]))

    chi4_frame_to_frame = ((all_frames[0][0][:, 7], all_frames[0][1][:, 7], all_frames[0][2][:, 7],
                            all_frames[0][3][:, 7], all_frames[0][4][:, 7], all_frames[0][5][:, 7],
                            all_frames[0][6][:, 7], all_frames[0][7][:, 7], all_frames[0][8][:, 7]),
                           (all_frames[1][0][:, 7], all_frames[1][1][:, 7], all_frames[1][2][:, 7]))

    chi1_frame_to_backb = ((all_frames[0][0][:, 4], all_frames[0][1][:, 4], all_frames[0][2][:, 4],
                            all_frames[0][3][:, 4], all_frames[0][4][:, 4], all_frames[0][5][:, 4],
                            all_frames[0][6][:, 4], all_frames[0][7][:, 4], all_frames[0][8][:, 4]),
                           (all_frames[1][0][:, 4], all_frames[1][1][:, 4], all_frames[1][2][:, 4]))

    chi2_frame_to_backb = geometry.rigids_mul_rigids(chi1_frame_to_backb, chi2_frame_to_frame)
    chi3_frame_to_backb = geometry.rigids_mul_rigids(chi2_frame_to_backb, chi3_frame_to_frame)
    chi4_frame_to_backb = geometry.rigids_mul_rigids(chi3_frame_to_backb, chi4_frame_to_frame)

    # Recombine them to a Rigids with shape (N, 8).
    all_frames_to_backb = rigids_concate_all(all_frames, chi2_frame_to_backb,
                                             chi3_frame_to_backb, chi4_frame_to_backb)

    backb_to_global = (geometry.rots_expand_dims(backb_to_global[0], -1),
                       geometry.vecs_expand_dims(backb_to_global[1], -1))
    # Create the global frames.
    all_frames_to_global = geometry.rigids_mul_rigids(backb_to_global, all_frames_to_backb)
    return all_frames_to_global

def rigids_from_tensor4x4(m):
    """Construct Rigids object from an 4x4 array.

    Here the 4x4 is representing the transformation in homogeneous coordinates.

    Args:
    m: Array representing transformations in homogeneous coordinates.
    Returns:
    Rigids object corresponding to transformations m
    """
    rotation = (m[..., 0, 0], m[..., 0, 1], m[..., 0, 2],
                m[..., 1, 0], m[..., 1, 1], m[..., 1, 2],
                m[..., 2, 0], m[..., 2, 1], m[..., 2, 2])
    trans = (m[..., 0, 3], m[..., 1, 3], m[..., 2, 3])
    rigid = (rotation, trans)
    return rigid

def rigids_concate_all(xall, x5, x6, x7):
    """rigids concate all."""
    x5 = (geometry.rots_expand_dims(x5[0], -1), geometry.vecs_expand_dims(x5[1], -1))
    x6 = (geometry.rots_expand_dims(x6[0], -1), geometry.vecs_expand_dims(x6[1], -1))
    x7 = (geometry.rots_expand_dims(x7[0], -1), geometry.vecs_expand_dims(x7[1], -1))
    xall_rot = xall[0]
    xall_rot_slice = []
    for val in xall_rot:
        xall_rot_slice.append(val[:, 0:5])
    xall_trans = xall[1]
    xall_trans_slice = []
    for val in xall_trans:
        xall_trans_slice.append(val[:, 0:5])
    xall = (xall_rot_slice, xall_trans_slice)
    res_rot = []
    for i in range(9):
        res_rot.append(jnp.concatenate((xall[0][i], x5[0][i], x6[0][i], x7[0][i]), axis=-1))
    res_trans = []
    for i in range(3):
        res_trans.append(jnp.concatenate((xall[1][i], x5[1][i], x6[1][i], x7[1][i]), axis=-1))
    return (res_rot, res_trans)




def frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global, restype_atom14_to_rigid_group,
                                                  restype_atom14_rigid_group_positions, restype_atom14_mask):  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

    Args:
    aatype: aatype for each residue.
    all_frames_to_global: All per residue coordinate frames.
    Returns:
    Positions of all atom coordinates in global frame.
    """

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = jnp.take(restype_atom14_to_rigid_group, aatype, 0)
    group_mask = nn.one_hot(num_classes=8, x=residx_to_group_idx)

    # Rigids with shape (N, 14)
    map_atoms_to_global = map_atoms_to_global_func(all_frames_to_global, group_mask)

    # Gather the literature atom positions for each residue.
    # Vecs with shape (N, 14)
    lit_positions = geometry.vecs_from_tensor(jnp.take(restype_atom14_rigid_group_positions, aatype, 0))

    # Transform each atom from its local frame to the global frame.
    # Vecs with shape (N, 14)
    pred_positions = geometry.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    mask = jnp.take(restype_atom14_mask, aatype, 0)

    pred_positions = geometry.vecs_scale(pred_positions, mask)

    return pred_positions


def map_atoms_to_global_func(all_frames, group_mask):
    """map atoms to global."""
    all_frames_rot = all_frames[0]
    all_frames_trans = all_frames[1]
    rot = geometry.rots_scale(geometry.rots_expand_dims(all_frames_rot, 1), group_mask)
    res_rot = []
    for val in rot:
        res_rot.append(jnp.sum(val, axis=-1))
    trans = geometry.vecs_scale(geometry.vecs_expand_dims(all_frames_trans, 1), group_mask)
    res_trans = []
    for val in trans:
        res_trans.append(jnp.sum(val, axis=-1))
    return (res_rot, res_trans)