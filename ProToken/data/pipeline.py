import io, os
from Bio.PDB import PDBParser
from typing import Any, Mapping, Optional
import dataclasses
import numpy as np
import math
import jax

# import MDAnalysis as mda
# from MDAnalysis.analysis import dihedrals
# import sys
# sys.path.append('.')

from common import residue_constants
from common.residue_constants import restype_3to1, restype_order, restype_num, atom_type_num, atom_types, atom_order, \
                                     order_restype_with_x_and_gap, chi_angles_mask, chi_angles_atoms, restypes, restype_1to3
from common.geometry import rigids_from_3_points, vecs_from_tensor, rots_from_tensor_np, rigids_mul_rots
from common import protein

from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

protoken_feature_content = ["seq_mask", "fake_aatype", "aatype", "residue_index", 
                            "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
                            "backbone_affine_tensor", "torsion_angles_sin_cos", "torsion_angles_mask"]
protoken_input_content = ["seq_mask", "fake_aatype", "residue_index", 
                          "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
                          "backbone_affine_tensor", "torsion_angles_sin_cos", "torsion_angles_mask"]
protoken_input_content_with_loss = ["seq_mask", "fake_aatype", "residue_index", 
                          "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
                          "backbone_affine_tensor", "backbone_affine_tensor", 
                          "torsion_angles_sin_cos", "torsion_angles_mask"]

protoken_dtype_dic = {"aatype": np.int32, "fake_aatype": np.int32,
                      "seq_mask": np.int32, "residue_index": np.int32,
                      "template_all_atom_positions": np.float32, 
                      "template_all_atom_masks": np.int32,
                      "template_pseudo_beta": np.float32,
                      "backbone_affine_tensor": np.float32,
                      "backbone_affine_tensor_label": np.float32,
                      "torsion_angles_sin_cos": np.float32,
                      "torsion_angles_mask": np.int32,
                      "atom14_atom_exists": np.float32,
                      "ca_coords": np.float32,}

def pseudo_beta_fn_np(aatype, all_atom_positions, all_atom_mask):
  """Create pseudo beta features."""
  
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']

  is_gly = np.equal(aatype, residue_constants.restype_order['G'])
  is_gly_tile = np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3])
  pseudo_beta = np.where(is_gly_tile, all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :])

  if all_atom_mask is not None:
    pseudo_beta_mask = np.where(is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta
  

@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]
    
    
def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

  WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

  Args:
    pdb_str: The contents of the pdb file
    chain_id: If None, then the pdb file must contain a single chain (which
      will be parsed). If chain_id is specified (e.g. A), then only that chain
      is parsed.

  Returns:
    A new `Protein` parsed from the pdb contents.
  """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser()
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError(
                'Only single chain PDBs are supported when chain_id not specified. '
                f'Found {len(chains)} chains.')
        chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []

    for res in chain:
        if res.id[2] != ' ':
            raise ValueError(
                f'PDB contains an insertion code at chain {chain.id} and residue '
                f'index {res.id[1]}. These are not supported.')
        res_shortname = restype_3to1.get(res.resname, 'X')
        restype_idx = restype_order.get(
            res_shortname, restype_num)
        pos = np.zeros((atom_type_num, 3))
        mask = np.zeros((atom_type_num,))
        res_b_factors = np.zeros((atom_type_num,))
        for atom in res:
            if atom.name not in atom_types:
                continue
            pos[atom_order[atom.name]] = atom.coord
            mask[atom_order[atom.name]] = 1.
            res_b_factors[atom_order[atom.name]] = atom.bfactor
        if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip it.
            continue
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors))


def to_tensor(rotation, translation):
    """get affine based on rotation and translation"""
    quaternion = rot_to_quat(rotation)
    return np.concatenate(
        [quaternion] +
        [np.expand_dims(x, axis=-1) for x in translation],
        axis=-1)

def rot_to_quat(rot, unstack_inputs=False):
    """transfer the rotation matrix to quaternion matrix"""
    if unstack_inputs:
        rot = [np.moveaxis(x, -1, 0) for x in np.moveaxis(rot, -2, 0)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy],
         [zy - yz, xx - yy - zz, xy + yx, xz + zx],
         [xz - zx, xy + yx, yy - xx - zz, yz + zy],
         [yx - xy, xz + zx, yz + zy, zz - xx - yy]]

    k = (1. / 3.) * np.stack([np.stack(x, axis=-1) for x in k],
                             axis=-2)
    # compute eigenvalues
    _, qs = np.linalg.eigh(k)
    return qs[..., -1]

def gather(params, indices, axis=0):
    """gather operation"""
    func = lambda p, i: np.take(p, i, axis=axis)
    return func(params, indices)


def np_gather_ops(params, indices, axis=0, batch_dims=0, is_multimer=False):
    """np gather operation"""
    if is_multimer:
        assert axis < 0 or axis - batch_dims >= 0
        ranges = []
        for i, s in enumerate(params.shape[:batch_dims]):
            r = np.arange(s)
            r = np.resize(r, (1,) * i + r.shape + (1,) * (len(indices.shape) - i - 1))
            ranges.append(r)
        remaining_dims = [slice(None) for _ in range(len(params.shape) - batch_dims)]
        remaining_dims[axis - batch_dims if axis >= 0 else axis] = indices
        ranges.extend(remaining_dims)
        return params[tuple(ranges)]

    if batch_dims == 0:
        return gather(params, indices)
    result = []
    if batch_dims == 1:
        for p, i in zip(params, indices):
            axis = axis - batch_dims if axis - batch_dims > 0 else 0
            r = gather(p, i, axis=axis)
            result.append(r)
        return np.stack(result)
    for p, i in zip(params[0], indices[0]):
        r = gather(p, i, axis=axis)
        result.append(r)
    res = np.stack(result)
    return res.reshape((1,) + res.shape)

def atom37_to_backbone_affine_tensor_np(
        aatype,
        all_atom_positions,
        all_atom_mask,
):
    r"""
    """
    flat_aatype = np.reshape(aatype, [-1])
    all_atom_positions = np.reshape(all_atom_positions, [-1, 37, 3])
    all_atom_mask = np.reshape(all_atom_mask, [-1, 37])

    rigid_group_names_res = np.full([21, 8, 3], '', dtype=object)

    # group 0: backbone frame
    rigid_group_names_res[:, 0, :] = ['C', 'CA', 'N']

    # group 3: 'psi'
    rigid_group_names_res[:, 3, :] = ['CA', 'C', 'O']

    # group 4,5,6,7: 'chi1,2,3,4'
    for restype, letter in enumerate(restypes):
        restype_name = restype_1to3[letter]
        for chi_idx in range(4):
            if chi_angles_mask[restype][chi_idx]:
                atom_names = chi_angles_atoms[restype_name][chi_idx]
                rigid_group_names_res[restype, chi_idx + 4, :] = atom_names[1:]

    lookup_table = atom_order.copy()
    lookup_table[''] = 0
    rigid_group_atom37_idx_restype = np.vectorize(lambda x: lookup_table[x])(
        rigid_group_names_res)

    rigid_group_atom37_idx_residx = np_gather_ops(
        rigid_group_atom37_idx_restype, flat_aatype)

    base_atom_pos = np_gather_ops(
        all_atom_positions,
        rigid_group_atom37_idx_residx,
        batch_dims=1)

    gt_frames = rigids_from_3_points(
        point_on_neg_x_axis=vecs_from_tensor(base_atom_pos[:, :, 0, :]),
        origin=vecs_from_tensor(base_atom_pos[:, :, 1, :]),
        point_on_xy_plane=vecs_from_tensor(base_atom_pos[:, :, 2, :]))

    rotations = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rotations[0, 0, 0] = -1
    rotations[0, 2, 2] = -1
    gt_frames = rigids_mul_rots(gt_frames, rots_from_tensor_np(rotations))

    rotation = [[gt_frames[0][0], gt_frames[0][1], gt_frames[0][2]],
                [gt_frames[0][3], gt_frames[0][4], gt_frames[0][5]],
                [gt_frames[0][6], gt_frames[0][7], gt_frames[0][8]]]
    translation = [gt_frames[1][0], gt_frames[1][1], gt_frames[1][2]]
    backbone_affine_tensor = to_tensor(rotation, translation)[:, 0, :]
    return backbone_affine_tensor

def calculate_dihedral_angle_np(A, B, C, D):
    a = B-A
    b = C-B
    c = D-C
    n1, n2 = np.cross(a, b), np.cross(b, c)
    b_ = np.cross(n1, n2)
    mask_ = np.sum(b*b_, axis=-1)
    mask = mask_ > 0
    angles_candidate_1 = np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2)/\
            (np.maximum(np.linalg.norm(n1, axis=1)*np.linalg.norm(n2, axis=1), 1e-6)), -1.0, 1.0))
    angles_candidate_2 = -np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2)/\
            (np.maximum(np.linalg.norm(n1, axis=1)*np.linalg.norm(n2, axis=1), 1e-6)), -1.0, 1.0))
    angles = np.where(mask, angles_candidate_1, angles_candidate_2)
    return angles

def get_ppo_angles_sin_cos(atom37_positions):
    n_pos = atom37_positions[:, 0, :]
    ca_pos = atom37_positions[:, 1, :]
    c_pos = atom37_positions[:, 2, :]
    # phi: C(n), N(n), CA(n+1), C(n+1), 
    a1, a2, a3, a4 = c_pos[:-1], n_pos[1:], ca_pos[1:], c_pos[1:]
    phi_angle_values = calculate_dihedral_angle_np(a1, a2, a3, a4)
    phi_angle_values = np.concatenate([np.zeros(1), phi_angle_values])
    # psi: N(n), CA(n), C(n), N(n+1)
    a1, a2, a3, a4 = n_pos[:-1], ca_pos[:-1], c_pos[:-1], n_pos[1:]
    psi_angle_values = calculate_dihedral_angle_np(a1, a2, a3, a4)
    psi_angle_values = np.concatenate([psi_angle_values, np.zeros(1)])
    # omega: CA(n), C(n+1), N(n+1), CA(n+1)
    a1, a2, a3, a4 = ca_pos[:-1], c_pos[:-1], n_pos[1:], ca_pos[1:]
    omega_angle_values = calculate_dihedral_angle_np(a1, a2, a3, a4)
    omega_angle_values = np.concatenate([omega_angle_values, np.zeros(1)])
    
    ppo_angle_tensor = np.stack([phi_angle_values, psi_angle_values, omega_angle_values], axis=1)
    ppo_angle_sin_cos = np.concatenate([np.sin(ppo_angle_tensor),  np.cos(ppo_angle_tensor)], axis=1)
    ppo_anlge_mask = np.ones(ppo_angle_tensor.shape, dtype=np.int32)
    ppo_anlge_mask[0, 0] = 0
    ppo_anlge_mask[-1, 1] = 0
    ppo_anlge_mask[-1, 2] = 0
    return ppo_angle_sin_cos, ppo_anlge_mask

def protoken_input_generator(pdb_path, NRES=512, crop_start_idx_preset=None):
    # if pdb_path_label is None:
    #     pdb_path_label = pdb_path
    with open(pdb_path, 'r') as f:
        prot_pdb = from_pdb_string(f.read())
        f.close()
    # with open(pdb_path_label, 'r') as f:
    #     prot_pdb_label = from_pdb_string(f.read())
    #     f.close()
    aatype = prot_pdb.aatype
    residue_index = prot_pdb.residue_index
#### change the residue index to start from 1
    residue_index = residue_index - residue_index[0] +1

    seq_len = len(aatype)

    if seq_len > 10000000000:
        # crop to NRES
        crop_mode = True
        CROP_LEN = NRES
        NRES = seq_len
        crop_start_idx = np.random.randint(0, seq_len - CROP_LEN + 1)
        crop_end_idx = crop_start_idx + CROP_LEN
        if crop_start_idx_preset is not None:
            crop_start_idx = crop_start_idx_preset
            crop_end_idx = crop_start_idx + CROP_LEN 
    else:
        # padding to NRES
        crop_mode = False
    if not crop_mode:
        crop_start_idx = 0

    ### create input features:
    # seq_mask & aatype & residue_index
    input_features_nopad = {}
    input_features_nopad['aatype'] = aatype
    input_features_nopad['seq_mask']  = np.ones(aatype.shape, dtype=np.float32)
    input_features_nopad['residue_index'] = residue_index
    # backbone_affine_tensor
    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)
    backbone_affine_tensor = atom37_to_backbone_affine_tensor_np(aatype, atom37_positions, atom37_mask)
    input_features_nopad['backbone_affine_tensor'] = backbone_affine_tensor
    # backbone_affine_tensor_label
    # aatype_label = prot_pdb_label.aatype
    # atom37_positions_label = prot_pdb_label.atom_positions.astype(np.float32)
    # atom37_mask_label = prot_pdb_label.atom_mask.astype(np.float32)
    # backbone_affine_tensor_label = atom37_to_backbone_affine_tensor_np(aatype_label, atom37_positions_label, atom37_mask_label)
    # input_features_nopad['backbone_affine_tensor_label'] = backbone_affine_tensor_label
    # template_all_atom_masks & template_all_atom_positions
    fake_aatype = np.ones(aatype.shape, dtype=np.int64)*7
    # fake_fasta = ''.join([order_restype_with_x_and_gap[x] for x in fake_aatype])
    GLY_MASK_ATOM37 = np.array([1,1,1,0,1]+[0]*32).astype(np.float32)
    fake_atom37_positions = atom37_positions * GLY_MASK_ATOM37.reshape(1,-1,1)
    fake_atom37_mask = atom37_mask * GLY_MASK_ATOM37.reshape(1,-1)
    GLY_MASK_ATOM14 = np.array([1,1,1,1,]+[0]*10).astype(np.float32)
    input_features_nopad['fake_aatype'] = fake_aatype
    input_features_nopad['template_all_atom_positions'] = fake_atom37_positions
    input_features_nopad['template_all_atom_masks'] = fake_atom37_mask
    input_features_nopad['atom14_atom_exists'] = np.ones([seq_len, 14], dtype=np.float32) * GLY_MASK_ATOM14.reshape(1,-1)
    
    # template pseudo beta
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn_np(fake_aatype, fake_atom37_positions, fake_atom37_mask)
    input_features_nopad['template_pseudo_beta'] = pseudo_beta
    
    # torsion_angles_sin_cos, torsion_angles_mask
    # angle_sin_cos, anlge_mask = get_angle_sin_cos(pdb_path)
    angle_sin_cos, anlge_mask = get_ppo_angles_sin_cos(fake_atom37_positions)
    input_features_nopad['torsion_angles_sin_cos'] = angle_sin_cos
    input_features_nopad['torsion_angles_mask'] = anlge_mask
    
    # dist_gt_perms, dist_mask_perms, perms_padding_mask
    # ca_coord_label = atom37_positions_label[:,1,:].reshape(-1,3)
    # ca_coord_pad_label = np.pad(ca_coord_label, ((0, NUM_RES-ca_coord_label.shape[0]), (0,0)), 'constant', constant_values=0)
    # ca_coord_pad_label = ca_coord_pad_label[None, ...] # (1, NUM_RES, 3)
    
    # dist_mtx_mask_label = np.asarray(np.sum(ca_coord_pad_label, axis=-1) != 0, dtype=np.float32)
    # dist_mtx_mask_2d_label = dist_mtx_mask_label[:,None] * dist_mtx_mask_label[None,:]
    # ca_distmtx_pad_label = np.sqrt(((ca_coord_pad_label[:,None,:] - ca_coord_pad_label[None,:,:])**2).sum(-1)) * dist_mtx_mask_2d_label
    # dist_gt_perms = ca_distmtx_pad_label.reshape(1,1,NUM_RES,NUM_RES)
    # perms_mask = np.zeros([NUM_RES, NUM_RES])
    # for ien in range(EXCLUDE_NEIGHBOR):
    #     perms_mask += np.diag(np.ones(NUM_RES-ien), k=ien) + np.diag(np.ones(NUM_RES-ien), k=-ien)
    # perms_mask -= np.eye(NUM_RES)
    # dist_mtx_mask_2d_label = dist_mtx_mask_2d_label * (1 - perms_mask)
    # dist_mask_perms = dist_mtx_mask_2d_label.reshape(1,1,NUM_RES,NUM_RES)
    # perms_padding_mask = np.array([1]).reshape(1,1)
    # pad features to NRES according to the shape of value and add batch dim
    input_features_pad = {}
    for k, v in input_features_nopad.items():
        pad_shape = list(v.shape)
        pad_shape[0] = NRES - pad_shape[0]
        pad_value = np.zeros(pad_shape, dtype=v.dtype)
        if k == 'backbone_affine_tensor':
            pad_value[...,0] = 1.0
        # if k == 'backbone_affine_tensor_label':
        #     pad_value[...,0] = 1.0
        input_features_pad[k] = np.concatenate([v, pad_value], axis=0).astype(protoken_dtype_dic[k])[None,...]
    # input_features_pad['dist_gt_perms'] = dist_gt_perms
    # input_features_pad['dist_mask_perms'] = dist_mask_perms
    # input_features_pad['ca_coords'] = ca_coord_pad_label
    # input_features_pad['perms_padding_mask'] = perms_padding_mask
    if crop_mode:
        input_features_pad['aatype'] = input_features_pad['aatype'][:, crop_start_idx:crop_end_idx]
        input_features_pad['fake_aatype'] = input_features_pad['fake_aatype'][:, crop_start_idx:crop_end_idx]
        input_features_pad['seq_mask'] = input_features_pad['seq_mask'][:, crop_start_idx:crop_end_idx]
        input_features_pad['residue_index'] = input_features_pad['residue_index'][:, crop_start_idx:crop_end_idx]
        input_features_pad['template_all_atom_masks'] = input_features_pad['template_all_atom_masks'][:, crop_start_idx:crop_end_idx, :]
        input_features_pad['template_all_atom_positions'] = input_features_pad['template_all_atom_positions'][:, crop_start_idx:crop_end_idx, :, :]
        input_features_pad['template_pseudo_beta'] = input_features_pad['template_pseudo_beta'][:, crop_start_idx:crop_end_idx, :]
        input_features_pad['backbone_affine_tensor'] = input_features_pad['backbone_affine_tensor'][:, crop_start_idx:crop_end_idx, :]
        input_features_pad['torsion_angles_sin_cos'] = input_features_pad['torsion_angles_sin_cos'][:, crop_start_idx:crop_end_idx, :]
        input_features_pad['torsion_angles_mask'] = input_features_pad['torsion_angles_mask'][:, crop_start_idx:crop_end_idx, :]
        # input_features_pad['atom14_atom_exists'] = input_features_pad['atom14_atom_exists'][:, crop_start_idx:crop_end_idx, :]
        # input_features_pad['ca_coords'] = input_features_pad['ca_coords'][:, crop_start_idx:crop_end_idx, :]

    input_feature_dict = {x: input_features_pad[x] for x in protoken_feature_content}
    return input_feature_dict, crop_start_idx, seq_len

# ["seq_mask", "aatype", "residue_index", 
#  "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
#  "backbone_affine_tensor", "torsion_angles_sin_cos", "torsion_angles_mask"]


def renum_pdb_str(pdb_str, Ls=None, renum=True, offset=1):
  if Ls is not None:
    L_init = 0
    new_chain = {}
    for L,c in zip(Ls, alphabet_list):
      new_chain.update({i:c for i in range(L_init,L_init+L)})
      L_init += L  

  n,num,pdb_out = 0,offset,[]
  resnum_ = None
  chain_ = None
  new_chain_ = new_chain[0]
  for line in pdb_str.split("\n"):
    if line[:4] == "ATOM":
      chain = line[21:22]
      resnum = int(line[22:22+5])
      if resnum_ is None: resnum_ = resnum
      if chain_ is None: chain_ = chain
      if resnum != resnum_ or chain != chain_:
        num += (resnum - resnum_)  
        n += 1
        resnum_,chain_ = resnum,chain
      if Ls is not None:
        if new_chain[n] != new_chain_:
          num = offset
          new_chain_ = new_chain[n]
      N = num if renum else resnum
      if Ls is None: pdb_out.append("%s%4i%s" % (line[:22],N,line[26:]))
      else: pdb_out.append("%s%s%4i%s" % (line[:21],new_chain[n],N,line[26:]))        
  return "\n".join(pdb_out)

def save_pdb_from_aux(aux, filename=None, renum_pdb=True):
  '''
  save pdb coordinates (if filename provided, otherwise return as string)
  - set get_best=False, to get the last sampled sequence
  '''
  aux = jax.tree_map(np.asarray, aux)
  p = {k:aux[k] for k in ["aatype","residue_index","atom_positions","atom_mask"]}        
  p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][...,None]
  Ls = [len(aux['aatype'])]

  def to_pdb_str(x, n=None):
    p_str = protein.to_pdb(protein.Protein(**x))
    p_str = "\n".join(p_str.splitlines()[1:-2])
    if renum_pdb: p_str = renum_pdb_str(p_str, Ls)
    if n is not None:
      p_str = f"MODEL{n:8}\n{p_str}\nENDMDL\n"
    return p_str

  if p["atom_positions"].ndim == 4:
    if p["aatype"].ndim == 3: p["aatype"] = p["aatype"].argmax(-1)
    p_str = ""
    for n in range(p["atom_positions"].shape[0]):
      p_str += to_pdb_str(jax.tree_map(lambda x:x[n],p), n+1)
    p_str += "END\n"
  else:
    if p["aatype"].ndim == 2: p["aatype"] = p["aatype"].argmax(-1)
    p_str = to_pdb_str(p)
  if filename is None: 
    return p_str, Ls
  else: 
    with open(filename, 'w') as f:
      f.write(p_str)

def calculate_tmscore_rmsd(path_1, path_2, 
                      tmscore_path='/data1/apps/TMalign/TMalign'):
    tmscore_result = os.popen(f'{tmscore_path} {path_1} {path_2}').read().splitlines()
    tmlines = [i for i in tmscore_result if i.startswith('TM-score=')]
    rmsdlines = [i for i in tmscore_result if 'RMSD=' in i]
    tmscores = [float(i.split('TM-score=')[-1].strip().split('(')[0].strip()) for i in tmlines]
    rmsds = [float(i.split('RMSD=')[-1].strip().split(',')[0].strip()) for i in rmsdlines]
    tmscore = max(tmscores)
    rmsd = max(rmsds)
    return tmscore, rmsd