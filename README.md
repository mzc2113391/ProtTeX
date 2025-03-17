# ProtTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models

This is the github repo for the paper Structure-In-Context Reasoning and Editing of Proteins with Large Language Models. An early version is preprinted at [arxiv][https://arxiv.org/abs/2503.08179].

<p align="center"><img src="https://github.com/mzc2113391/ProtTeX/blob/main/figs/cover.pdf" width="100%"></p>

## Installation

ProtTeX is built upon [ProToken](https://github.com/issacAzazel/ProToken.git).


```bash
git clone https://github.com/issacAzazel/ProToken.git
```
For protein tokenization, you should install the environment following above repository.

Then, you should install the basic environment for running Hugging Face models.

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Using ProtTeX
To infer the function of protein, first tokenize it.

```bash
python ./scripts/tokenize.py ./input/input.pdb
```

Then you can run the function inference example

```bash
python ./scripts/function_inference.py ./input/input_dit_recon.pkl
```

## Citation
```python
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