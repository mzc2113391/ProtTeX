# ProtTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models

This is the github repo for the paper Structure-In-Context Reasoning and Editing of Proteins with Large Language Models.

<p align="center"><img src="https://github.com/mzc2113391/ProtTeX/blob/main/figs/git_cover.png" width="100%"></p>

## Installation

ProtTeX is built upon [ProToken](https://github.com/issacAzazel/ProToken.git), we use a slightly modified version of the ProToken repository.

Create a conda environment for ProtTeX:
```bash
mamba env create -f environment.yaml
```
Download the ProToken model param from [Here](https://drive.google.com/file/d/11G4ImYm14f_s2jpqrpyav8jiM25JO3uE/view?usp=sharing) and put it in ./ProToken/ckpts

Download the ProtTeX model param from [Here](https://huggingface.co/mzcwd/ProtTeX) and put it in ./model/ProtTeX


## Using ProtTeX
### Function inference
To infer the function of protein, first tokenize it.

```bash
export CUDA_VISIBLE_DEVICES=0
python ./scripts/tokenize_pdb.py --pdb_path ./input/input.pdb
```

Then you can run the function inference example

```bash
python ./scripts/function_inference.py --input_protein_pkl ./input/input_dit_recon.pkl
#output: The provided protein structure has been assessed, and it is likely to play a role in 3'-5'-RNA exonuclease activity, nucleic acid binding according to its structural characteristics.
```

### Structure prediction
First generate language text.
```bash
python ./scripts/structure_prediction.py --input_seq MKIVLATRNKGKIREIEEILKDFPIELLSLADFPELPEVVEDGKTFEENAVKKAVTVAKATGLLALADDSGL --output_pkl ./output/output_st.pkl
```
Then detokenize to generate the structure.
```bash
python ./scripts/detokenize_pdb.py --input_path ./output/output_st.pkl --output_dir ./output
```

### Multiconformation sampling
First generate language text.
```bash
python ./scripts/multiconformation_samplling.py --input_seq MKIVLATRNKGKIREIEEILKDFPIELLSLADFPELPEVVEDGKTFEENAVKKAVTVAKATGLLALADDSGL --output_pkl ./output/output_st.pkl
```
Then detokenize to generate all structures.
```bash
python ./scripts/detokenize_pdb.py --input_path ./output/output_st.pkl --output_dir ./output
```

### Controllable design
First generate language text.
```bash
python ./scripts/controllable_design.py --input_seq MKIVLATRNKGKIREIEEILKDFPIELLSLADFPELPEVVEDGKTFEENAVKKAVTVAKATGLLALADDSGL --output_pkl ./output/output_st.pkl
```
Then detokenize to generate all structures.
```bash
python ./scripts/detokenize_pdb.py --input_path ./output/output_st.pkl --output_dir ./output
```

## Citation
```python
@article{lin2023tokenizing,
    title={Tokenizing Foldable Protein Structures with Machine-Learned Artificial Amino-Acid Vocabulary},
    author={Lin, Xiaohan and Chen, Zhenyu and Li, Yanheng and Ma, Zicheng and Fan, Chuanliu and Cao, Ziqiang and Feng, Shihao and Gao, Yi Qin and Zhang, Jun},
    year={2025},
    journal={Chemical Science},
    url={https://pubs.rsc.org/en/Content/ArticleLanding/2025/SC/D5SC02055G},
    doi={10.1039/D5SC02055G}
}
@article{ma2025prottexstructureincontextreasoningediting,
    title={ProtTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models}, 
    author={Zicheng Ma and Chuanliu Fan and Zhicong Wang and Zhenyu Chen and Xiaohan Lin and Yanheng Li and Shihao Feng and Jun Zhang and Ziqiang Cao and Yi Qin Gao},
    year={2025},
    journal={Journal of Chemical Information and Modeling},
    url={https://pubs.acs.org/doi/10.1021/acs.jcim.5c00585},
    doi={10.1021/acs.jcim.5c00585}
}
```


## Contact
For questions or further information, please contact [jzhang@cpl.ac.cn](jzhang@cpl.ac.cn).
