import jax
import jax.numpy as jnp
import pickle as pkl
import numpy as np
import argparse
import concurrent.futures
import warnings
import os
import multiprocessing
import json
import re
warnings.filterwarnings("ignore")
# biopthon's warnings is annoying
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import sys
sys.path.append('./ProToken')

GLY_MASK_ATOM37 = np.array([1,1,1,0,1]+[0]*32).astype(np.float32)

def extract_protein_structure(text):
    pattern = r"< protein structure>(.*?)</ protein structure>"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None

def feature_generate(vq_indexes,padding_len=1024):
    input_features_nopad = {}
    seq_len = len(vq_indexes)
    true_aatype = np.ones((seq_len,), dtype=np.int32)
    aatype = np.ones((seq_len,), dtype=np.int32)
    seq_mask = np.ones((seq_len,), dtype=np.float32)
    residue_index = np.arange(seq_len, dtype=np.int32) + 1
    all_atom_masks = np.ones((seq_len, 37), dtype=np.float32)
    template_all_atom_masks = all_atom_masks * GLY_MASK_ATOM37.reshape(1,-1)

    input_features_nopad["vq_indexes"] = np.array(vq_indexes,dtype=np.int32)
    input_features_nopad["seq_mask"] = seq_mask
    input_features_nopad["true_aatype"] = true_aatype
    input_features_nopad["fake_aatype"] = aatype
    input_features_nopad["residue_index"] = residue_index
    input_features_nopad["template_all_atom_masks"] = template_all_atom_masks
    input_features_pad = {}
    for k, v in input_features_nopad.items():
        pad_shape = list(v.shape)
        pad_shape[0] = padding_len - pad_shape[0]
        pad_value = np.zeros(pad_shape, dtype=v.dtype)
        if k == 'vq_indexes':
            pad_value[...] = 0
        input_features_pad[k] = np.concatenate([v, pad_value], axis=0).astype(input_features_nopad[k].dtype)[None,...]
    return input_features_pad




def arg_parse():
    parser = argparse.ArgumentParser(description='Inputs for main.py')
    # model config
    parser.add_argument('--encoder_config', default="./ProToken/config/encoder.yaml", help='encoder config')
    parser.add_argument('--decoder_config', default="./ProToken/config/decoder.yaml", help='decoder config')
    parser.add_argument('--input_path', default="./output/output_st.pkl", help='input path')
    parser.add_argument('--output_dir', default="./output", help="output directory")
    parser.add_argument('--vq_config', default='./ProToken/config/vq.yaml', help='vq config')
    parser.add_argument('--load_ckpt_path', 
                        default='./ProToken/ckpts/protoken_params_130000.pkl',
                        type=str, help='Location of checkpoint file.')
    parser.add_argument('--random_seed', type=int, default=8888, help="random seed")
    parser.add_argument('--np_random_seed', type=int, default=18888, help="np random seed")

    arguments = parser.parse_args()
    return arguments

args = arg_parse()

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

from functools import partial
from flax import linen as nn
from flax.jax_utils import replicate
from model.encoder import VQ_Encoder
from model.decoder import VQ_Decoder, Protein_Decoder
from tokenizer.vector_quantization import VQTokenizer
from inference.inference_new import EncoderCell,InferenceCell,DecoderCell

from train.utils import logger, make_rng_dict
from common.config_load import load_config
from data.pipeline import protoken_input_generator, protoken_input_content, \
    protoken_feature_content, \
    save_pdb_from_aux, calculate_tmscore_rmsd
import datetime

from config.global_config import GLOBAL_CONFIG
from config.train_vq_config import TRAINING_CONFIG

RANK = jax.process_index()
print("Multiprocess initializing, Hello from ", RANK)


def infer():
    ##### constants 
    NRES = 1024 # 256

    ##### initialize models
    encoder_cfg = load_config(args.encoder_config)
    decoder_cfg = load_config(args.decoder_config)
    encoder_cfg.seq_len = NRES 
    decoder_cfg.seq_len = NRES
    vq_cfg = load_config(args.vq_config)

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
    
    decoder_cell = DecoderCell(
        global_config=GLOBAL_CONFIG,
        vq_tokenizer=modules["vq_tokenizer"]["module"],
        vq_decoder=modules["vq_decoder"]["module"],
        protein_decoder=modules["protein_decoder"]["module"],
        project_out=modules["project_out"]["module"],
        quantize=bool(vq_cfg.quantize)
    )

    rng_key = jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.np_random_seed)
    decoder_cell_jit = jax.jit(decoder_cell.apply)

    with open('./tokenizer_metadata/character.json',"rb") as f:
        new_tokens = json.load(f)

    new_tokens = {v:k for k,v in enumerate(new_tokens)}
    
    with open(args.input_path, 'rb') as f:
        result_list = pkl.load(f)

    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    result_dict = {}

    result_dict["tmscore_list"] = []
    result_dict["rmsd_list"] = []
    result_dict["valid_index_list"] = []


    for index in range(len(result_list)):

        result_text = result_list[index]
        
        accesion = str(index)
        
        input = extract_protein_structure(result_text)
        
        try:
            input = [new_tokens[i] for i in input]
        except Exception:
            print("Invalid tokens, skiping this sequence.")
            continue
        
        result_dict["valid_index_list"].append(index)

        test_case = feature_generate(input)

        batched_feature_gt = jax.tree_map(lambda x: jnp.array(x), test_case)

        aux_results_part = decoder_cell_jit(params, batched_feature_gt["seq_mask"][0], 
                                        batched_feature_gt["fake_aatype"][0],
                                        batched_feature_gt["residue_index"][0],
                                        batched_feature_gt["vq_indexes"][0],)
        
        tmp_aux_result_part = {"aatype": batched_feature_gt["true_aatype"][0].astype(np.int32),
                                "residue_index": batched_feature_gt["residue_index"][0].astype(np.int32),
                                "atom_positions": aux_results_part["recon_pos"].astype(np.float32),
                                "atom_mask": batched_feature_gt["template_all_atom_masks"][0].astype(np.float32),
                                "plddt": aux_results_part["code_indices"].astype(np.float32)/100,}
        
        save_pdb_from_aux(tmp_aux_result_part, f"{save_path}/{accesion}_recon.pdb")
        
        print("index", index)

    with open(f'{save_path}/result.pkl', 'wb') as f:
        pkl.dump(result_dict, f)


if __name__ == "__main__":
    infer()
