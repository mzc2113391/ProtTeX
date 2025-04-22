# ProtTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models

This is the github repo for the paper Structure-In-Context Reasoning and Editing of Proteins with Large Language Models. An early version is preprinted at [arxiv][https://arxiv.org/abs/2503.08179].

<p align="center"><img src="https://github.com/mzc2113391/ProtTeX/blob/main/figs/git_cover.png" width="100%"></p>

## Installation

ProtTeX is built upon [ProToken](https://github.com/issacAzazel/ProToken.git), we use a slightly modified version of the ProToken repository.

Create a conda environment for ProtTeX:
```bash
mamba env create -f environment.yml
```
Download the ProToken model param from [Here](https://drive.google.com/file/d/11G4ImYm14f_s2jpqrpyav8jiM25JO3uE/view?usp=sharing) and put it in ./ProToken/ckpts

Download the ProtTeX model param from [Here](https://huggingface.co/mzcwd/ProtTeX) and put it in ./model/ProtTeX


## Using ProtTeX
### function inference
To infer the function of protein, first tokenize it.

```bash
python ./scripts/tokenize.py --pdb_path ./input/input.pdb
```

Then you can run the function inference example

```bash
python ./scripts/function_inference.py --input_protein_pkl ./input/input_dit_recon.pkl
#output: The provided protein structure has been assessed, and it is likely to play a role in 3'-5'-RNA exonuclease activity, nucleic acid binding according to its structural characteristics.
```

## Citation
```python
@article{lin2023tokenizing,
    title={Tokenizing Foldable Protein Structures with Machine-Learned Artificial Amino-Acid Vocabulary},
    author={Lin, Xiaohan and Chen, Zhenyu and Li, Yanheng and Ma, Zicheng and Fan, Chuanliu and Cao, Ziqiang and Feng, Shihao and Gao, Yi Qin and Zhang, Jun},
    journal={bioRxiv},
    pages={2023--11},
    year={2023},
    publisher={Cold Spring Harbor Laboratory}
}
@article{ma2025prottexstructureincontextreasoningediting,
      title={ProtTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models}, 
      author={Zicheng Ma and Chuanliu Fan and Zhicong Wang and Zhenyu Chen and Xiaohan Lin and Yanheng Li and Shihao Feng and Jun Zhang and Ziqiang Cao and Yi Qin Gao},
      year={2025},
      journal={arXiv preprint arXiv:2503.08179},
      eprint={2503.08179},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2503.08179},
}
```


## Contact
For questions or further information, please contact [jzhang@cpl.ac.cn](jzhang@cpl.ac.cn).