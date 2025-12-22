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

def extract_protein_sequence(text):
    pattern = r"< protein sequence>(.*?)</ protein sequence>"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None
    
def extract_protein_structure(text):
    pattern = r"< protein structure>(.*?)</ protein structure>"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None

restype_dict =  {i:restype  for i, restype in enumerate(restypes)}

def arg_parse():
    parser = argparse.ArgumentParser(description='Inputs for protein function inference')
    # model config
    parser.add_argument('--model_path', default="./model/ProtTeX", help='model path')
    parser.add_argument('--character_aa_dict', default="./tokenizer_metadata/character_aa_dict.pkl", help='amino acid letter dict')
    parser.add_argument('--output_pkl', default="./output/output_st.pkl", help='output protein pkl')
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
    top_p = 0.9,
    temperature= 0.6,
    num_return_sequences=5,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

    response = [text[len(input):].strip() for text in generated_text]

    return generated_text,response

args = arg_parse()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

model = AutoModelForCausalLM.from_pretrained(args.model_path,device_map="auto",torch_dtype=torch.bfloat16)

with open(args.character_aa_dict, 'rb') as f:
    character_aa_dict = pkl.load(f)

character_2_aa = {v:k for k,v in character_aa_dict.items()}

with open(args.character_protoken, 'rb') as f:
    character_protoken = json.load(f)

user_prompt = f"esign a functional protein sequence with the following characteristics:\n1. The Mg(2+) binding site should be located in a region of the protein that is accessible to the ligand.\n2. The designed protein should have ITP diphosphatase activity, XTP diphosphatase activity, nucleotide binding to facilitate purine nucleoside triphosphate catabolic process. Subsequently, based on the description and sequence, predict the overall protein structure."

generated_text, response = sample(user_prompt)

response_new = []

sucess_count = 0
for item in response:
    try:
        input_aatoken = extract_protein_sequence(item)
        input_protoken = extract_protein_structure(item)
        aaseq = "".join([character_2_aa[i] for i in input_aatoken])
        print("Successfully designed protein sequence:", aaseq)
        print("Successfully designed Predicted protein structure tokens:", input_protoken)
        response_new.append(f"< protein sequence>{aaseq}</ protein sequence>\n< protein structure>{input_protoken}</ protein structure>")
        sucess_count += 1
    
    except Exception as e:
        print("Error extracting sequence or structure:", e)
        continue
        
print(f"Total successfully designed protein sequences and structures: {sucess_count}/{len(response)}")
        
with open(args.output_pkl, 'wb') as f:
    pkl.dump(response_new, f)
    print(f"Saved designed protein sequences and structures to {args.output_pkl}, please use detokenize.py to get PDB files.")