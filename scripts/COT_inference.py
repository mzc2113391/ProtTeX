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
    parser.add_argument('--input_seq', default="AAAAAAAA", help='input amino acid sequence')
    parser.add_argument('--output_pkl', default="./output/output_st.pkl", help='output protein pkl')
    parser.add_argument('--character_aa_dict', default="./tokenizer_metadata/character_aa_dict.pkl", help='amino acid letter dict')
    parser.add_argument('--character_protoken', default="./tokenizer_metadata/character.json", help='protoken letter list')

    arguments = parser.parse_args()
    return arguments

def convert_sentence_to_template(user_prompt):
    template= f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return template

def sample(user_prompt):
    user_input = convert_sentence_to_template(user_prompt)
    input_ids = tokenizer.encode(user_input, return_tensors="pt",add_special_tokens=True).to(device)
    input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens = 1024,
    do_sample = True,
    num_beams=5,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True) 

    response = generated_text[len(input):].strip()
    
    
    return generated_text,response

args = arg_parse()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

model = AutoModelForCausalLM.from_pretrained(args.model_path,device_map="auto",torch_dtype=torch.bfloat16)

with open(args.character_aa_dict, 'rb') as f:
    character_aa_dict = pkl.load(f)

with open(args.character_protoken, 'rb') as f:
    character_protoken = json.load(f)

input_aatoken = "".join([character_aa_dict[i] for i in args.input_seq])

user_prompt = f"Please analyze the provided protein sequence and offer a detailed description of its subunit composition, including the organization and interactions between the subunits. Subsequently, based on the sequence and structural analysis, predict the overall protein structure.\n< protein sequence>{input_aatoken}</ protein sequence>"

generated_text, response = sample(user_prompt)

output_path = args.output_pkl

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'wb') as f:
    pkl.dump([response], f) 