
# Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe



<p align="center">
    &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/YHLLEO/efficientmoe">HuggingFace</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2512.01252">Tech Report</a> &nbsp&nbsp
</p>


## 1. 🔥 Updates
- __[2025.3.20]__: Release the [code](https://github.com/yhlleo/EfficientMoE) of DSMoE and JiTMoE.


## 2. 📖 Introduction

We release the MoE Transformer that can be applied to both latent and pixel-space diffusion frameworks, employing DeepSeek-style expert modules, alternative intermediate widths, varying expert counts, and enhanced attention positional encodings. The models are already relased to Huggingface. The source codes are coming soon!<br>

## 3. Preparation

### 3.1 Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### 3.2 Installation

TODO

### 3.3 Training

TODO

### 3.4 Evaluation

TODO

## 4. Main results

### 4.1 Latent diffusion framework

 - Ours DSMoE v.s. [DiffMoE](https://arxiv.org/pdf/2503.14487) on 700K training steps with CFG = 1.0 (* refers to the reported results in the official paper):

| Model Name                 | # Act. Params | FID-50K↓  | Inception Score↑ |
|----------------------------|-------------------------|---------|----------------|
|DiffMoE-S-E16|32M|41.02|37.53|
|DSMoE-S-E16|33M|39.84|38.63|
|DSMoE-S-E48|30M|40.20|38.09|
|DiffMoE-B-E16|130M|20.83|70.26|
|DSMoE-B-E16|132M|20.33|71.42|
|DSMoE-B-E48|118M|19.46|72.69|
|DiffMoE-L-E16|458M|11.16 (14.41*)|107.74 (88.19*)|
|DSMoE-L-E16|465M|9.80|115.45|
|DSMoE-L-E48|436M|9.19|118.52|
|DSMoE-3B-E16|965M|7.52|135.29|

 - Ours DSMoE v.s. DiffMoE on 700K training steps with CFG = 1.5:

| Model Name                 | # Act. Params | FID-50K↓  | Inception Score↑ |
|----------------------------|-------------------------|---------|----------------|
|DiffMoE-S-E16|32M|15.47|94.04|
|DSMoE-S-E16|33M|14.53|97.55|
|DSMoE-S-E48|30M|14.81|96.51|
|DiffMoE-B-E16|130M|4.87|183.43|
|DSMoE-B-E16|132M|4.50|186.79|
|DSMoE-B-E48|118M|4.27|191.03|
|DiffMoE-L-E16|458M|2.84|256.57|
|DSMoE-L-E16|465M|2.59|272.55|
|DSMoE-L-E48|436M|2.55|278.35|
|DSMoE-3B-E16|965M|2.38|304.93|


### 4.2 Pixel-space diffusion framework 

-  Ours JiTMoE v.s. [JiT](https://arxiv.org/pdf/2511.13720) on 200 training epochs with CFG interval (* refers to the reported results in the official paper):

| Model Name                 | # Act. Params | FID-50K↓  | Inception Score↑ |
|----------------------------|-------------------------|---------|----------------|
|JiT-B/16|131M|4.81 (4.37*)| 222.32 (-)|
|JiTMoE-B/16-E16|133M|4.23| 245.53|
|JiT-L/16|459M| 3.19 (2.79*)| 309.72 (-)|
|JiTMoE-L/16-E16|465M|3.10| 311.34|


## 5. Acknowledgements
A large portion of codes in this repo is based on [DiffMoE](https://github.com/KlingTeam/DiffMoE), [JiT](https://github.com/LTH14/JiT), [DeepSeekMoE](https://github.com/deepseek-ai/DeepSeek-MoE)

## 6. 🌟 Citation

```
@article{liu2025efficient,
  title={Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recip},
  author={Liu Yahui and Yue Yang and Zhang Jingyuan and Sun Chenxi and Zhou Yang and Zeng Wencong and Tang Ruiming and Zhou Guorui},
  journal={arXiv preprint arXiv:2512.01252},
  year={2025}
}
```
