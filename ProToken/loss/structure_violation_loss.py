### Implement of structure violations, copy from AF2 code (multimer)

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

Float = Union[float, jnp.ndarray]

# Python-version-specific alias (Python 2: unicode; Python 3: str)
Text = str

from common import residue_constants
from common.config_load import Config
from model import geometry, utils

def _make_restype_atom14_to_atom37():
	"""Map from atom14 to atom37 per residue type."""
	restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
	for rt in residue_constants.restypes:
		atom_names = residue_constants.restype_name_to_atom14_names[
			residue_constants.restype_1to3[rt]]
		restype_atom14_to_atom37.append([
			(residue_constants.atom_order[name] if name else 0)
			for name in atom_names
		])
	# Add dummy mapping for restype 'UNK'
	restype_atom14_to_atom37.append([0] * 14)
	restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
	return restype_atom14_to_atom37

RESTYPE_ATOM14_TO_ATOM37 = _make_restype_atom14_to_atom37()
def get_atom14_to_atom37_map(aatype):
	return utils.batched_gather(jnp.asarray(RESTYPE_ATOM14_TO_ATOM37), aatype)

def structural_violation_loss(mask: jnp.ndarray,
                              violations: Mapping[str, Float],
                              ) -> Float:
	"""Computes Loss for structural Violations."""
	# Put all violation losses together to one large loss.
	num_atoms = jnp.sum(mask).astype(jnp.float32) + 1e-6
	between_residues = violations['between_residues']
	within_residues = violations['within_residues']
	return (between_residues['bonds_c_n_loss_mean'] +
			between_residues['angles_ca_c_n_loss_mean']  +
			between_residues['angles_c_n_ca_loss_mean'] +
			jnp.sum(between_residues['clashes_per_atom_loss_sum'] +
					within_residues['per_atom_loss_sum']) / num_atoms
			)

def find_structural_violations_array(
		aatype: jnp.ndarray,
		residue_index: jnp.ndarray,
		mask: jnp.ndarray,
		pred_positions: jnp.ndarray,  # (N, 14, 3)
		config: dict,
		asym_id: jnp.ndarray,
    ) -> Dict[str, Any]:
    
    return find_structural_violations(
		aatype,
		residue_index,
		mask,
		geometry.Vec3Array.from_array(pred_positions),
		config,
		asym_id,
	)

def find_structural_violations(
		aatype: jnp.ndarray,
		residue_index: jnp.ndarray,
		mask: jnp.ndarray,
		pred_positions: geometry.Vec3Array,  # (N, 14)
		config: dict,
		asym_id: jnp.ndarray,
    ) -> Dict[str, Any]:
	"""Computes several checks for structural Violations."""

	# Compute between residue backbone violations of bonds and angles.
	connection_violations = between_residue_bond_loss(
		pred_atom_positions=pred_positions,
		pred_atom_mask=mask.astype(jnp.float32),
		residue_index=residue_index.astype(jnp.float32),
		aatype=aatype,
		tolerance_factor_soft=config["structural_violation"]["violation_tolerance_factor"],
		tolerance_factor_hard=config["structural_violation"]["violation_tolerance_factor"])

	# Compute the van der Waals radius for every atom
	# (the first letter of the atom name is the element type).
	# shape (N, 14)
	atomtype_radius = jnp.array([
		residue_constants.van_der_waals_radius[name[0]]
		for name in residue_constants.atom_types
	])
	residx_atom14_to_atom37 = get_atom14_to_atom37_map(aatype)
	atom_radius = mask * utils.batched_gather(atomtype_radius,
											residx_atom14_to_atom37)

	# Compute the between residue clash loss.
	between_residue_clashes = between_residue_clash_loss(
		pred_positions=pred_positions,
		atom_exists=mask,
		atom_radius=atom_radius,
		residue_index=residue_index,
		overlap_tolerance_soft=config["structural_violation"]["clash_overlap_tolerance"],
		overlap_tolerance_hard=config["structural_violation"]["clash_overlap_tolerance"],
		asym_id=asym_id)

	# Compute all within-residue violations (clashes,
	# bond length and angle violations).
	restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
		overlap_tolerance=config["structural_violation"]["clash_overlap_tolerance"],
		bond_length_tolerance_factor=config["structural_violation"]["violation_tolerance_factor"])
	dists_lower_bound = utils.batched_gather(restype_atom14_bounds['lower_bound'],
											aatype)
	dists_upper_bound = utils.batched_gather(restype_atom14_bounds['upper_bound'],
											aatype)
	within_residue_violations = within_residue_violation_loss(
		pred_positions=pred_positions,
		atom_exists=mask,
		dists_lower_bound=dists_lower_bound,
		dists_upper_bound=dists_upper_bound,
		tighten_bounds_for_loss=0.0)

	# Combine them to a single per-residue violation mask (used later for LDDT).
	per_residue_violations_mask = jnp.max(jnp.stack([
		connection_violations['per_residue_violation_mask'],
		jnp.max(between_residue_clashes['per_atom_clash_mask'], axis=-1),
		jnp.max(within_residue_violations['per_atom_violations'],
				axis=-1)]), axis=0)

	return {
		'between_residues': {
			'bonds_c_n_loss_mean':
				connection_violations['c_n_loss_mean'],  # ()
			'angles_ca_c_n_loss_mean':
				connection_violations['ca_c_n_loss_mean'],  # ()
			'angles_c_n_ca_loss_mean':
				connection_violations['c_n_ca_loss_mean'],  # ()
			'connections_per_residue_loss_sum':
				connection_violations['per_residue_loss_sum'],  # (N)
			'connections_per_residue_violation_mask':
				connection_violations['per_residue_violation_mask'],  # (N)
			'clashes_mean_loss':
				between_residue_clashes['mean_loss'],  # ()
			'clashes_per_atom_loss_sum':
				between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
			'clashes_per_atom_clash_mask':
				between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
		},
		'within_residues': {
			'per_atom_loss_sum':
				within_residue_violations['per_atom_loss_sum'],  # (N, 14)
			'per_atom_violations':
				within_residue_violations['per_atom_violations'],  # (N, 14),
		},
		'total_per_residue_violations_mask':
			per_residue_violations_mask,  # (N)
	}


def compute_violation_metrics(
    residue_index: jnp.ndarray,
    mask: jnp.ndarray,
    seq_mask: jnp.ndarray,
    pred_positions: geometry.Vec3Array,  # (N, 14)
    violations: Mapping[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
	"""Compute several metrics to assess the structural violations."""
	ret = {}
	between_residues = violations['between_residues']
	within_residues = violations['within_residues']
	extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
		positions=pred_positions,
		mask=mask.astype(jnp.float32),
		residue_index=residue_index.astype(jnp.float32))
	ret['violations_extreme_ca_ca_distance'] = extreme_ca_ca_violations
	ret['violations_between_residue_bond'] = utils.mask_mean(
		mask=seq_mask,
		value=between_residues['connections_per_residue_violation_mask'])
	ret['violations_between_residue_clash'] = utils.mask_mean(
		mask=seq_mask,
		value=jnp.max(between_residues['clashes_per_atom_clash_mask'], axis=-1))
	ret['violations_within_residue'] = utils.mask_mean(
		mask=seq_mask,
		value=jnp.max(within_residues['per_atom_violations'], axis=-1))
	ret['violations_per_residue'] = utils.mask_mean(
		mask=seq_mask, value=violations['total_per_residue_violations_mask'])
	return ret


def between_residue_bond_loss(
    pred_atom_positions: geometry.Vec3Array,  # (N, 37(14))
    pred_atom_mask: jnp.ndarray,  # (N, 37(14))
    residue_index: jnp.ndarray,  # (N)
    aatype: jnp.ndarray,  # (N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0) -> Dict[Text, jnp.ndarray]:
	"""Flat-bottom loss to penalize structural violations between residues."""

	assert len(pred_atom_positions.shape) == 2
	assert len(pred_atom_mask.shape) == 2
	assert len(residue_index.shape) == 1
	assert len(aatype.shape) == 1

	# Get the positions of the relevant backbone atoms.
	this_ca_pos = pred_atom_positions[:-1, 1]  # (N - 1)
	this_ca_mask = pred_atom_mask[:-1, 1]         # (N - 1)
	this_c_pos = pred_atom_positions[:-1, 2]  # (N - 1)
	this_c_mask = pred_atom_mask[:-1, 2]          # (N - 1)
	next_n_pos = pred_atom_positions[1:, 0]  # (N - 1)
	next_n_mask = pred_atom_mask[1:, 0]           # (N - 1)
	next_ca_pos = pred_atom_positions[1:, 1]  # (N - 1)
	next_ca_mask = pred_atom_mask[1:, 1]          # (N - 1)
	has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
		jnp.float32)

	# Compute loss for the C--N bond.
	c_n_bond_length = geometry.euclidean_distance(this_c_pos, next_n_pos, 1e-6)

	# The C-N bond to proline has slightly different length because of the ring.
	next_is_proline = (
		aatype[1:] == residue_constants.restype_order['P']).astype(jnp.float32)
	gt_length = (
		(1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
		+ next_is_proline * residue_constants.between_res_bond_length_c_n[1])
	gt_stddev = (
		(1. - next_is_proline) *
		residue_constants.between_res_bond_length_stddev_c_n[0] +
		next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
	c_n_bond_length_error = jnp.sqrt(1e-6 +
									jnp.square(c_n_bond_length - gt_length))
	c_n_loss_per_residue = jax.nn.relu(
		c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
	mask = this_c_mask * next_n_mask * has_no_gap_mask
	c_n_loss = jnp.sum(mask * c_n_loss_per_residue) / (jnp.sum(mask) + 1e-6)
	c_n_violation_mask = mask * (
		c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

	# Compute loss for the angles.
	c_ca_unit_vec = (this_ca_pos - this_c_pos).normalized(1e-6)
	c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length
	n_ca_unit_vec = (next_ca_pos - next_n_pos).normalized(1e-6)

	ca_c_n_cos_angle = c_ca_unit_vec.dot(c_n_unit_vec)
	gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
	gt_stddev = residue_constants.between_res_cos_angles_ca_c_n[1]
	ca_c_n_cos_angle_error = jnp.sqrt(
		1e-6 + jnp.square(ca_c_n_cos_angle - gt_angle))
	ca_c_n_loss_per_residue = jax.nn.relu(
		ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
	mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
	ca_c_n_loss = jnp.sum(mask * ca_c_n_loss_per_residue) / (jnp.sum(mask) + 1e-6)
	ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error >
									(tolerance_factor_hard * gt_stddev))

	c_n_ca_cos_angle = (-c_n_unit_vec).dot(n_ca_unit_vec)
	gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
	gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
	c_n_ca_cos_angle_error = jnp.sqrt(
		1e-6 + jnp.square(c_n_ca_cos_angle - gt_angle))
	c_n_ca_loss_per_residue = jax.nn.relu(
		c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
	mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
	c_n_ca_loss = jnp.sum(mask * c_n_ca_loss_per_residue) / (jnp.sum(mask) + 1e-6)
	c_n_ca_violation_mask = mask * (
		c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

	# Compute a per residue loss (equally distribute the loss to both
	# neighbouring residues).
	per_residue_loss_sum = (c_n_loss_per_residue +
							ca_c_n_loss_per_residue +
							c_n_ca_loss_per_residue)
	per_residue_loss_sum = 0.5 * (jnp.pad(per_residue_loss_sum, [[0, 1]]) +
								jnp.pad(per_residue_loss_sum, [[1, 0]]))

	# Compute hard violations.
	violation_mask = jnp.max(
		jnp.stack([c_n_violation_mask,
					ca_c_n_violation_mask,
					c_n_ca_violation_mask]), axis=0)
	violation_mask = jnp.maximum(
		jnp.pad(violation_mask, [[0, 1]]),
		jnp.pad(violation_mask, [[1, 0]]))

	return {'c_n_loss_mean': c_n_loss,  # shape ()
			'ca_c_n_loss_mean': ca_c_n_loss,  # shape ()
			'c_n_ca_loss_mean': c_n_ca_loss,  # shape ()
			'per_residue_loss_sum': per_residue_loss_sum,  # shape (N)
			'per_residue_violation_mask': violation_mask  # shape (N)
			}


def between_residue_clash_loss(
    pred_positions: geometry.Vec3Array,  # (N, 14)
    atom_exists: jnp.ndarray,  # (N, 14)
    atom_radius: jnp.ndarray,  # (N, 14)
    residue_index: jnp.ndarray,  # (N)
    asym_id: jnp.ndarray,  # (N)
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5) -> Dict[Text, jnp.ndarray]:
	"""Loss to penalize steric clashes between residues."""
	assert len(pred_positions.shape) == 2
	assert len(atom_exists.shape) == 2
	assert len(atom_radius.shape) == 2
	assert len(residue_index.shape) == 1

	# Create the distance matrix.
	# (N, N, 14, 14)
	dists = geometry.euclidean_distance(pred_positions[:, None, :, None],
										pred_positions[None, :, None, :], 1e-10)

	# Create the mask for valid distances.
	# shape (N, N, 14, 14)
	dists_mask = (atom_exists[:, None, :, None] * atom_exists[None, :, None, :])

	# Mask out all the duplicate entries in the lower triangular matrix.
	# Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
	# are handled separately.
	dists_mask *= (
		residue_index[:, None, None, None] < residue_index[None, :, None, None])

	# Backbone C--N bond between subsequent residues is no clash.
	c_one_hot = jax.nn.one_hot(2, num_classes=14)
	n_one_hot = jax.nn.one_hot(0, num_classes=14)
	neighbour_mask = ((residue_index[:, None] + 1) == residue_index[None, :])
	neighbour_mask &= (asym_id[:, None] == asym_id[None, :])
	neighbour_mask = neighbour_mask[..., None, None]
	c_n_bonds = neighbour_mask * c_one_hot[None, None, :,
											None] * n_one_hot[None, None, None, :]
	dists_mask *= (1. - c_n_bonds)

	# Disulfide bridge between two cysteines is no clash.
	cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
	cys_sg_one_hot = jax.nn.one_hot(cys_sg_idx, num_classes=14)
	disulfide_bonds = (cys_sg_one_hot[None, None, :, None] *
						cys_sg_one_hot[None, None, None, :])
	dists_mask *= (1. - disulfide_bonds)

	# Compute the lower bound for the allowed distances.
	# shape (N, N, 14, 14)
	dists_lower_bound = dists_mask * (
		atom_radius[:, None, :, None] + atom_radius[None, :, None, :])

	# Compute the error.
	# shape (N, N, 14, 14)
	dists_to_low_error = dists_mask * jax.nn.relu(
		dists_lower_bound - overlap_tolerance_soft - dists)

	# Compute the mean loss.
	# shape ()
	mean_loss = (jnp.sum(dists_to_low_error)
				/ (1e-6 + jnp.sum(dists_mask)))

	# Compute the per atom loss sum.
	# shape (N, 14)
	per_atom_loss_sum = (jnp.sum(dists_to_low_error, axis=[0, 2]) +
						jnp.sum(dists_to_low_error, axis=[1, 3]))

	# Compute the hard clash mask.
	# shape (N, N, 14, 14)
	clash_mask = dists_mask * (
		dists < (dists_lower_bound - overlap_tolerance_hard))

	# Compute the per atom clash.
	# shape (N, 14)
	per_atom_clash_mask = jnp.maximum(
		jnp.max(clash_mask, axis=[0, 2]),
		jnp.max(clash_mask, axis=[1, 3]))

	return {'mean_loss': mean_loss,  # shape ()
			'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
			'per_atom_clash_mask': per_atom_clash_mask  # shape (N, 14)
			}
 
 
def within_residue_violation_loss(
    pred_positions: geometry.Vec3Array,  # (N, 14)
    atom_exists: jnp.ndarray,  # (N, 14)
    dists_lower_bound: jnp.ndarray,  # (N, 14, 14)
    dists_upper_bound: jnp.ndarray,  # (N, 14, 14)
    tighten_bounds_for_loss=0.0,
) -> Dict[Text, jnp.ndarray]:
	"""Find within-residue violations."""
	assert len(pred_positions.shape) == 2
	assert len(atom_exists.shape) == 2
	assert len(dists_lower_bound.shape) == 3
	assert len(dists_upper_bound.shape) == 3

	# Compute the mask for each residue.
	# shape (N, 14, 14)
	dists_masks = (1. - jnp.eye(14, 14)[None])
	dists_masks *= (atom_exists[:, :, None] * atom_exists[:, None, :])

	# Distance matrix
	# shape (N, 14, 14)
	dists = geometry.euclidean_distance(pred_positions[:, :, None],
										pred_positions[:, None, :], 1e-10)

	# Compute the loss.
	# shape (N, 14, 14)
	dists_to_low_error = jax.nn.relu(
		dists_lower_bound + tighten_bounds_for_loss - dists)
	dists_to_high_error = jax.nn.relu(
		dists + tighten_bounds_for_loss - dists_upper_bound)
	loss = dists_masks * (dists_to_low_error + dists_to_high_error)

	# Compute the per atom loss sum.
	# shape (N, 14)
	per_atom_loss_sum = (jnp.sum(loss, axis=1) +
						jnp.sum(loss, axis=2))

	# Compute the violations mask.
	# shape (N, 14, 14)
	violations = dists_masks * ((dists < dists_lower_bound) |
								(dists > dists_upper_bound))

	# Compute the per atom violations.
	# shape (N, 14)
	per_atom_violations = jnp.maximum(
		jnp.max(violations, axis=1), jnp.max(violations, axis=2))

	return {'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
			'per_atom_violations': per_atom_violations  # shape (N, 14)
			}
 
 
def extreme_ca_ca_distance_violations(
    positions: geometry.Vec3Array,  # (N, 37(14))
    mask: jnp.ndarray,  # (N, 37(14))
    residue_index: jnp.ndarray,  # (N)
    max_angstrom_tolerance=1.5
    ) -> jnp.ndarray:
	"""Counts residues whose Ca is a large distance from its neighbor."""
	this_ca_pos = positions[:-1, 1]  # (N - 1,)
	this_ca_mask = mask[:-1, 1]         # (N - 1)
	next_ca_pos = positions[1:, 1]  # (N - 1,)
	next_ca_mask = mask[1:, 1]  # (N - 1)
	has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
		jnp.float32)
	ca_ca_distance = geometry.euclidean_distance(this_ca_pos, next_ca_pos, 1e-6)
	violations = (ca_ca_distance -
				residue_constants.ca_ca) > max_angstrom_tolerance
	mask = this_ca_mask * next_ca_mask * has_no_gap_mask
	return utils.mask_mean(mask=mask, value=violations)