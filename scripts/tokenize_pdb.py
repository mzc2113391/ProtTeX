import sys
sys.path.append('./ProToken')
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
from functools import partial
from model.encoder import VQ_Encoder
from model.decoder import VQ_Decoder, Protein_Decoder
from tokenizer.vector_quantization import VQTokenizer
from inference.inference_new import EncoderCell,InferenceCell

from train.utils import logger, make_rng_dict
from common.config_load import load_config
from data.pipeline import protoken_input_generator, protoken_input_content, \
    protoken_feature_content, \
    save_pdb_from_aux, calculate_tmscore_rmsd
import datetime

from config.global_config import GLOBAL_CONFIG
from config.train_vq_config import TRAINING_CONFIG
import argparse
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.jax_utils import replicate
import pickle as pkl
import numpy as np 


def arg_parse():
    parser = argparse.ArgumentParser(description='Inputs for main.py')
    # model config
    parser.add_argument('--encoder_config', default="./ProToken/config/encoder.yaml", help='encoder config')
    parser.add_argument('--decoder_config', default="./ProToken/config/decoder.yaml", help='decoder config')
    parser.add_argument('--pdb_path', default="./input/input.pdb", help='pdb_path')
    parser.add_argument('--vq_config', default='./ProToken/config/vq.yaml', help='vq config')
    parser.add_argument('--load_ckpt_path', 
                        default='./ProToken/ckpts/protoken_params_130000.pkl',
                        type=str, help='Location of checkpoint file.')
    parser.add_argument('--random_seed', type=int, default=8888, help="random seed")
    parser.add_argument('--np_random_seed', type=int, default=18888, help="np random seed")

    arguments = parser.parse_args()
    return arguments

args = arg_parse()



# jax.distributed.initialize(
#     coordinator_address=args.coordinator_address, #"128.5.20.2:8888",
#     num_processes=args.num_processes,
#     process_id=args.rank,
#     local_device_ids=[0,1,2,3,4,5,6,7]
# )
RANK = jax.process_index()
print("Multiprocess initializing, Hello from ", RANK)

def convert_feat_aux_to_numpy(feat_, aux_result):
    feat_converted = {}
    for key, value in feat_.items():
        if key in ['seq_mask', 'fake_aatype', 'aatype', 'residue_index', 'template_all_atom_masks']:
            feat_converted[key] = np.asarray(value).astype(np.int32)
        else:
            feat_converted[key] = np.asarray(value).astype(np.float32)

    aux_result_converted = {
        'recon_pos': np.asarray(aux_result['recon_pos']).astype(np.float32),
        'code_indices': np.asarray(aux_result['code_indices']).astype(np.float32)
    }

    return feat_converted, aux_result_converted


def process_pdb(input_gt_pdb_path,out_pkl_path,feat_,aux_result,seq_len_tmp):
    gt_pdb_saving_path = f"{os.path.dirname(out_pkl_path)}/{(os.path.basename(input_gt_pdb_path)).split('.')[0]}_dit_recon.pdb"
    tmp_save_dict = {}
    tmp_save_subdict = {}
    tmp_save_subdict['seq_mask'] = feat_['seq_mask'][0]
    tmp_save_subdict['aatype'] = feat_['aatype'][0]
    tmp_save_subdict['residue_index'] = feat_['residue_index'][0]
    tmp_save_subdict['code_indices'] = aux_result['code_indices'][0]
    tmp_save_subdict['seq_len'] = seq_len_tmp

    tmp_aux_result_gt = {
        "aatype": feat_['aatype'][0].astype(np.int32),
        "residue_index": feat_['residue_index'][0].astype(np.int32),
        "atom_positions": aux_result['recon_pos'][0].astype(np.float32),
        "atom_mask": feat_["template_all_atom_masks"][0].astype(np.float32),
        "plddt": aux_result["code_indices"][0].astype(np.float32) / 100
    }

    save_pdb_from_aux(tmp_aux_result_gt, gt_pdb_saving_path)

    tmscore_gt, rmsd_gt = calculate_tmscore_rmsd(input_gt_pdb_path, gt_pdb_saving_path)

    print("tmscore_gt", tmscore_gt)

    tmp_save_subdict["tmscore"] = tmscore_gt
    tmp_save_subdict["rmsd"] = rmsd_gt

    tmp_save_dict = tmp_save_subdict

    with open(out_pkl_path, 'wb') as f:
        pkl.dump(tmp_save_dict, f)

def infer():
    ##### constants 
    NRES = 1024 # 256

    ##### initialize models
    encoder_cfg = load_config(args.encoder_config)
    decoder_cfg = load_config(args.decoder_config)
    encoder_cfg.seq_len = NRES 
    decoder_cfg.seq_len = NRES
    vq_cfg = load_config(args.vq_config)
    pdb_path = args.pdb_path
    out_pkl_path = f"{os.path.dirname(pdb_path)}/{(os.path.basename(pdb_path)).split('.')[0]}_dit_recon.pkl"

    modules = {
        "encoder": {"module": VQ_Encoder, 
                    "args": {"global_config": GLOBAL_CONFIG, "cfg": encoder_cfg}, 
                    "freeze": False},
        "vq_decoder": {"module": VQ_Decoder, 
                       "args": {"global_config": GLOBAL_CONFIG, "cfg": decoder_cfg, "pre_layer_norm": False}, 
                       "freeze": False},
        "protein_decoder": {"module": Protein_Decoder, 
                        "args": {"global_config": GLOBAL_CONFIG, "cfg": decoder_cfg}, 
                        "freeze": False},
        "vq_tokenizer": {"module": VQTokenizer, 
                         "args": {"config": vq_cfg, "dtype": jnp.float32}, 
                         "freeze": False},
        "project_in": {"module": nn.Dense, 
                       "args": {"features": vq_cfg.dim_code, "kernel_init": nn.initializers.lecun_normal(), "use_bias": False}, 
                       "freeze": False},
        "project_out": {"module": nn.Dense, 
                       "args": {"features": vq_cfg.dim_in, "kernel_init": nn.initializers.lecun_normal(), "use_bias": False},
                       "freeze": False},
    }

    if args.load_ckpt_path:
        ##### load params
        with open(args.load_ckpt_path, "rb") as f:
            params = pkl.load(f)
            params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)

    for k, v in modules.items():
        modules[k]["module"] = v["module"](**v["args"])
        if v["freeze"]:
            partial_params = {"params": params["params"].pop(k)}
            modules[k]["module"] = partial(modules[k]["module"].apply, partial_params)
    
    inference_cell = InferenceCell(
        global_config=GLOBAL_CONFIG,
        train_cfg = TRAINING_CONFIG,
        encoder=modules["encoder"]["module"],
        vq_tokenizer=modules["vq_tokenizer"]["module"],
        vq_decoder=modules["vq_decoder"]["module"],
        protein_decoder=modules["protein_decoder"]["module"],
        project_in=modules["project_in"]["module"],
        project_out=modules["project_out"]["module"],
        quantize=bool(vq_cfg.quantize)
    )

    rng_key = jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.np_random_seed)

    ##### replicate params & opt_state
    params = replicate(params)

    inference_cell_jit = jax.jit(inference_cell.apply)
    inference_cell_jvj = jax.jit(jax.vmap(inference_cell_jit, in_axes=[None] + [0] * 9))
    inference_cell_pjvj = jax.pmap(jax.jit(inference_cell_jvj), axis_name="i")


    
    # breakpoint()
    # test_base_path = valid_metadata_dict[0][0]['result_dir']
    # test_path_1 = f'{test_base_path}/gt.pdb'
    # test_path_2 = f'{test_base_path}/seq_cut_02.pdb'
    # test_path_3 = f'{test_base_path}/space_cut_02.pdb'
    # test_path_4 = f'{test_base_path}/gaussian_noise_007.pdb'
    feat_ = {}
    for k_ in protoken_feature_content:
        feat_[k_] = []
    start_time_load = datetime.datetime.now()
    # for path_ in [test_path_1, test_path_2, test_path_3, test_path_4, \
    #             test_path_1, test_path_2, test_path_3, test_path_4,]:
    
    # concurrently load input features
    future_ = protoken_input_generator(pdb_path,NRES=NRES, crop_start_idx_preset=0)
####insertion code error
    batched_feature_tmp, crop_start_idx_tmp, seq_len_tmp = future_
    batched_feature_tmp = jax.tree_map(lambda x: jnp.array(x), batched_feature_tmp)
    for k_ in protoken_feature_content:
        feat_[k_].append(batched_feature_tmp[k_])
    feat_ = {k_: jnp.concatenate(v_, axis=0) for k_, v_ in feat_.items()}
    end_time_load = datetime.datetime.now()
    print("Time to load: ", end_time_load - start_time_load)
    # breakpoint()

    net_rng_key, rng_key = make_rng_dict(rng_key,
                                        ["fape_clamp_key",],
                                            num_rngs_per_key=feat_['fake_aatype'].shape[0],
                                        # num_rngs_per_key=64,
                                        squeeze=False)
    #### reshape inputs 
    reshape_func = lambda x:x.reshape(1, x.shape[0]//1, *x.shape[1:])
    feat_reshape = jax.tree_util.tree_map(reshape_func, feat_)
    net_rng_key_reshape = jax.tree_util.tree_map(reshape_func, net_rng_key)
    # feat_ = jax.tree_util.tree_map(lambda x: jnp.tile(x[None,...], (8,)+tuple([1]*len(x.shape))), feat_)
    input_feature_tmp = [feat_reshape[name] for name in protoken_input_content]
    # breakpoint()

    start_time_inference = datetime.datetime.now()
    aux_result = inference_cell_pjvj(params, *input_feature_tmp, rngs=net_rng_key_reshape)
    end_time_inference = datetime.datetime.now()
    print("Time to inference: ", end_time_inference - start_time_inference)
    aux_result = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), aux_result)
    # save results:
    # seq_mask, aatype, residue_index, code_indices
    start_time_save = datetime.datetime.now()
    feat_converted, aux_result_converted = convert_feat_aux_to_numpy(feat_, aux_result)

    process_pdb(pdb_path, out_pkl_path, feat_converted, aux_result_converted, seq_len_tmp)

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as pool:
    #     pool.imap(
    #         process_pdb,
    #         [(i16_, path_, tmp_input_path_list, tmp_result_path_list, feat_converted, aux_result_converted) for i16_, path_ in enumerate(tmp_result_path_list)]
    #     )
    #     pool.close()
    #     pool.join()
    end_time_save = datetime.datetime.now()
    print("Time to save: ", end_time_save - start_time_save)

if __name__ == "__main__":
    infer()