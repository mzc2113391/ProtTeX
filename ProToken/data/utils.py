import numpy as np
import jax 
import jax.numpy as jnp 
from common import residue_constants
import os
from biopandas.pdb import PandasPdb

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
  """Create pseudo beta features."""
  
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']

  is_gly = jnp.equal(aatype, residue_constants.restype_order['G'])
  is_gly_tile = jnp.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3])
  pseudo_beta = jnp.where(is_gly_tile, all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :])

  if all_atom_mask is not None:
    pseudo_beta_mask = jnp.where(is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.astype(jnp.float32)
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta

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

##### numpy version is implemented in data/preprocess.py
def calculate_dihedral_angle_jnp(A, B, C, D):
    a = B-A
    b = C-B
    c = D-C
    n1, n2 = jnp.cross(a, b), jnp.cross(b, c)
    b_ = jnp.cross(n1, n2)
    mask_ = jnp.sum(b*b_, axis=-1)
    mask = mask_ > 0
    angles_candidate_1 = jnp.arccos(jnp.clip(
        jnp.einsum('ij,ij->i', n1, n2)/\
            (jnp.maximum(
                jnp.linalg.norm(n1, axis=1)*jnp.linalg.norm(n2, axis=1), 1e-6)
            ), -1.0, 1.0)) # * 180 / jnp.pi
    angles_candidate_2 = -jnp.arccos(jnp.clip(
        jnp.einsum('ij,ij->i', n1, n2)/\
            (jnp.maximum(
                jnp.linalg.norm(n1, axis=1)*jnp.linalg.norm(n2, axis=1), 1e-6)
            ), -1.0, 1.0)) # * 180 / jnp.pi
    angles = jnp.where(mask, angles_candidate_1, angles_candidate_2)
    return angles

def calculate_dihedral_angle_sin_cos_jnp(A, B, C, D, seq_mask):
    _dtype = A.dtype
    A, B, C, D = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), (A, B, C, D))
    ##### arccos is not a numerical stable op, calculate sin/cos directly
    a = B-A
    b = C-B
    c = D-C
    n1, n2 = jnp.cross(a, b), jnp.cross(b, c)
    b_ = jnp.cross(n1, n2)
    sign_mask_ = jnp.sum(b*b_, axis=-1)
    sign_mask = sign_mask_ > 0
    
    n1 = n1 + jnp.expand_dims(1.0 - seq_mask, axis=-1).astype(jnp.float32) * 1e-6 ##### prevent nan bug
    n2 = n2 + jnp.expand_dims(1.0 - seq_mask, axis=-1).astype(jnp.float32) * 1e-6 ##### prevent nan bug
    norm = jnp.maximum(jnp.linalg.norm(n1, axis=1)*jnp.linalg.norm(n2, axis=1), 1e-6)
    
    angles_cos = jnp.clip(jnp.einsum('ij,ij->i', n1, n2) / norm, -1.0, 1.0)
    angles_sin_candidate_1 = jnp.sqrt(jnp.maximum(1.0 - angles_cos**2, 1e-6))
    angles_sin_candidate_2 = -angles_sin_candidate_1
    angles_sin = jnp.where(sign_mask, angles_sin_candidate_1, angles_sin_candidate_2)
    return angles_sin.astype(_dtype), angles_cos.astype(_dtype)

def get_ppo_angles_sin_cos(atom37_positions, seq_mask):
    n_pos = atom37_positions[:, 0, :]
    ca_pos = atom37_positions[:, 1, :]
    c_pos = atom37_positions[:, 2, :]
    # phi: CA(n), C(n), N(n), CA(n+1)
    a1, a2, a3, a4 = c_pos[:-1], n_pos[1:], ca_pos[1:], c_pos[1:]
    phi_angle_sin, phi_angle_cos = calculate_dihedral_angle_sin_cos_jnp(a1, a2, a3, a4, seq_mask[1:])
    phi_angle_sin = jnp.concatenate([jnp.zeros(1), phi_angle_sin])
    phi_angle_cos = jnp.concatenate([jnp.ones(1) * jnp.cos(0.0), phi_angle_cos])
    # psi: N(n), CA(n), C(n), N(n+1)
    a1, a2, a3, a4 = n_pos[:-1], ca_pos[:-1], c_pos[:-1], n_pos[1:]
    psi_angle_sin, psi_angle_cos = calculate_dihedral_angle_sin_cos_jnp(a1, a2, a3, a4, seq_mask[:-1])
    psi_angle_sin = jnp.concatenate([psi_angle_sin, jnp.zeros(1)])
    psi_angle_cos = jnp.concatenate([psi_angle_cos, jnp.ones(1) * jnp.cos(0.0)])
    # omega: CA(n), C(n+1), N(n+1), CA(n+1)
    a1, a2, a3, a4 = ca_pos[:-1], c_pos[:-1], n_pos[1:], ca_pos[1:]
    omega_angle_sin, omega_angle_cos = calculate_dihedral_angle_sin_cos_jnp(a1, a2, a3, a4, seq_mask[:-1])
    omega_angle_sin = jnp.concatenate([omega_angle_sin, jnp.zeros(1)])
    omega_angle_cos = jnp.concatenate([omega_angle_cos, jnp.ones(1) * jnp.cos(0.0)])
    
    ppo_angle_sin_cos = jnp.concatenate([phi_angle_sin[..., None], psi_angle_sin[..., None], omega_angle_sin[..., None], 
                                         phi_angle_cos[..., None], psi_angle_cos[..., None], omega_angle_cos[..., None]], axis=-1)

    return ppo_angle_sin_cos

def make_data_pair(batch_input, reconstructed_structure_dict, rng_key,
                   feature_names, num_adversarial_samples, nsamples_per_device,
                   recycle_vq_codes=False, num_data_pairs=1):

    def map_key_idx(x):
        return feature_names.index(x)
    
    data_pairs = []
    rand_indexes = jax.random.choice(
            rng_key, 
            a=jnp.arange(0, 
                         nsamples_per_device - num_adversarial_samples, dtype=jnp.int32), 
            shape=(num_data_pairs,), 
            replace=False
        )
    for l in range(num_data_pairs):
        rand_idx = rand_indexes[l]
        data_pair = {
            "pos": jax.tree_util.tree_map(lambda x:x[rand_idx], batch_input),
            "neg": jax.tree_util.tree_map(lambda x:x[rand_idx], batch_input) 
        }
        data_pair["neg"][map_key_idx("backbone_affine_tensor")] \
            = reconstructed_structure_dict["reconstructed_backbone_affine_tensor"][rand_idx]
        reconstructed_atom37_positions = \
            reconstructed_structure_dict["reconstructed_atom_positions"][rand_idx]
        
        data_pair["neg"][map_key_idx("template_all_atom_positions")] = \
                                        reconstructed_atom37_positions
        data_pair["neg"][map_key_idx("template_pseudo_beta")] = \
                pseudo_beta_fn(data_pair["pos"][map_key_idx("aatype")],
                                reconstructed_atom37_positions,
                                all_atom_mask=None)
        torsion_angles_sin_cos, _ = get_ppo_angles_sin_cos(reconstructed_atom37_positions)
        data_pair["neg"][map_key_idx("torsion_angles_sin_cos")] = torsion_angles_sin_cos
        
        if recycle_vq_codes:
            data_pair["neg"][map_key_idx("prev_vq_codes")] = \
                reconstructed_structure_dict["vq_codes"][rand_idx]
        data_pairs.append(data_pair)

    return data_pairs

def make_2d_features(data_dict, nres, exlucde_neighbor):
    mask_2d = data_dict['seq_mask'][:,:,None] *\
              data_dict['seq_mask'][:,None,:]
    data_dict['dist_gt_perms'] = jnp.expand_dims(jnp.linalg.norm(
        (data_dict['ca_coords'][:, :, None, :] - data_dict['ca_coords'][:, None, :, :]), axis=-1
    ) * mask_2d, axis=1) ### add perms dimension
    
    perms_mask = jnp.triu(jnp.ones((nres, nres), dtype=jnp.int32), -exlucde_neighbor+1) \
                * jnp.tril(jnp.ones((nres, nres), dtype=jnp.int32), exlucde_neighbor-1)
    dist_mask_perms = mask_2d * (1 - perms_mask)[None, :, :]
    dist_mask_perms = dist_mask_perms.reshape(-1, 1, nres, nres) ### add perms dimension
    
    data_dict['dist_mask_perms'] = dist_mask_perms 

    return data_dict

def mask_aatype(aatype, decoding_steps=10):
    batch_size = aatype.shape[0]
    # decoding_step = \
    #     np.random.randint(0, decoding_steps, 
    #                       size=(batch_size,)).astype(np.float32) / float(decoding_steps) # (B, )
    decoding_step = np.random.uniform(0, decoding_steps, 
                                      size=(batch_size, )).astype(np.float32) / float(decoding_steps)
    
    mask_ratio = np.cos(np.pi * decoding_step * 0.5) # (B,)
    mask = np.random.rand(*aatype.shape) < mask_ratio[:, None] # (B, L)
    
    mask_token = np.ones_like(aatype, dtype=np.int32) * 20 # 20 is the mask
    masked_aatype = np.where(mask, mask_token, aatype)
    
    return masked_aatype

def make_label_mask_from_extra_mask(seq_mask, extra_mask, ca_coords, cutoff=8.0, is_np=True):
    ca_coords = ca_coords.astype(jnp.float32)
    np_ = np if is_np else jnp
    distance_mtx = np_.linalg.norm(ca_coords[..., None, :] - ca_coords[..., None, :, :], axis=-1)
    
    #### not masked by seq_mask, but masked by extra_mask
    true_mask = np_.logical_and(np_.logical_not(extra_mask), seq_mask)
    true_mask_2d = np_.logical_or(true_mask[..., None], true_mask[..., None, :])
    distance_mtx = distance_mtx + 1e5 * (np_.logical_not(true_mask_2d).astype(jnp.float32))
    
    label_mask = np_.logical_not(np_.min(distance_mtx, axis=-1) < cutoff)
    
    return jnp.logical_and(label_mask, seq_mask)

def crop_pdb_edges(input_pdb_path, output_pdb_path, percent=0.1, terminal='C'):
    # terminal: 'N' or 'C' or 'NC'
    seq = ''.join(os.popen(f'pdb_tofasta {input_pdb_path}').read().splitlines()[1:])
    seq_len = len(seq)
    if terminal == 'N':
        N_crop_len = int(seq_len * percent)
        C_crop_len = len(seq)  
    elif terminal == 'C':
        N_crop_len = 0
        C_crop_len = len(seq) - int(seq_len * percent)
    elif terminal == 'NC':
        N_crop_len = int(seq_len * percent/2)
        C_crop_len = len(seq) - int(seq_len * percent/2)
    else:
        raise ValueError('terminal should be N or C or NC')
    
    # load and crop pdb
    ppdb = PandasPdb().read_pdb(input_pdb_path)
    residue_index_list = ppdb.df['ATOM']['residue_number'].unique()
    residue_index_list = sorted(residue_index_list)
    N_crop_residue_list = residue_index_list[:N_crop_len]
    C_crop_residue_list = residue_index_list[C_crop_len:]
    crop_residue_list = N_crop_residue_list + C_crop_residue_list
    ppdb.df['ATOM'] = ppdb.df['ATOM'][~ppdb.df['ATOM']['residue_number'].isin(crop_residue_list)]
    ppdb.to_pdb(output_pdb_path, records=['ATOM'])

    return crop_residue_list


def calculate_core_residue_index(input_pdb_path, select_residue_index):

    ppdb = PandasPdb().read_pdb(input_pdb_path)
    residue_index_list = ppdb.df['ATOM']['residue_number'].unique()
    residue_index_list = sorted(residue_index_list)

    rest_residue_index = [i for i in residue_index_list if i not in select_residue_index]
    ca_coords_dict = {}
    for i in residue_index_list:
        ca_coords_dict[i] = ppdb.df['ATOM'][(ppdb.df['ATOM']['residue_number'] == i) & (ppdb.df['ATOM']['atom_name'] == 'CA')][['x_coord', 'y_coord', 'z_coord']].values[0]

    neighbor_residue_index = []
    for k_ in rest_residue_index:
        for j_ in select_residue_index:
            if np.linalg.norm(ca_coords_dict[k_] - ca_coords_dict[j_]) <= 8.0:
                neighbor_residue_index.append(k_)
                break
    # print(neighbor_residue_index)
    core_residue_index = [i for i in residue_index_list if i not in neighbor_residue_index \
                          and i not in select_residue_index]

    return core_residue_index
   

