import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
import json
import torch
import os
from tqdm import tqdm
import pickle as pkl
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]

restype_dict =  {i:restype  for i, restype in enumerate(restypes)}

def arg_parse():
    parser = argparse.ArgumentParser(description='Inputs for protein function inference')
    # model config
    parser.add_argument('--model_path', default="./model/ProtTeX", help='model path')
    parser.add_argument('--input_protein_pkl', default="./input/input_dit_recon.pkl", help='tokenized protein pkl')
    parser.add_argument('--character_aa_dict', default="./tokenizer_metadata/character_aa_dict.pkl", help='amino acid letter dict')
    parser.add_argument('--character_protoken', default="./tokenizer_metadata/character.json", help='protoken letter list')

    arguments = parser.parse_args()
    return arguments

def convert_sentence_to_template(user_prompt,input_aatoken,input_protoken):
    input = f"{user_prompt}\n< protein sequence>{input_aatoken}</ protein sequence>\n< protein structure>{input_protoken}</ protein structure>"
    template= f"<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return template

def sample(user_prompt,input_aatoken,input_protoken):
    user_input = convert_sentence_to_template(user_prompt,input_aatoken,input_protoken)
    input_ids = tokenizer.encode(user_input, return_tensors="pt",add_special_tokens=True).to(device)
    input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens = 1024,
    do_sample = False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True) 

    response = generated_text[len(input):].strip()

    # print("User input:",input)
    print("Response:",response)
    
    
    return generated_text,response

args = arg_parse()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

model = AutoModelForCausalLM.from_pretrained(args.model_path,device_map="auto",torch_dtype=torch.bfloat16)

with open(args.input_protein_pkl, 'rb') as f:
    input_pdb = pkl.load(f)

vq_indexes = input_pdb['code_indices'][:input_pdb['seq_len']]

aa_sequence = [restype_dict[i] for i in input_pdb['aatype'][:input_pdb['seq_len']]]

with open(args.character_aa_dict, 'rb') as f:
    character_aa_dict = pkl.load(f)

with open(args.character_protoken, 'rb') as f:
    character_protoken = json.load(f)
    
input_protoken = "".join([character_protoken[int(i)] for i in vq_indexes])

input_aatoken = "".join([character_aa_dict[i] for i in aa_sequence])

user_prompt = f"Considering the protein structure above, predict its biological function by examining its structural features and comparing it to functionally characterized proteins."

sample(user_prompt,input_aatoken,input_protoken)