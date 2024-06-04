# StableFDG

This repo contains the PyTorch implementation of our paper, [StableFDG: Style and Attention Based Learning for Federated Domain Generalization at NeurIPS'23]([https://proceedings.neurips.cc/paper/2021/hash/076a8133735eb5d7552dc195b125a454-Abstract.html](https://openreview.net/pdf?id=IjZa2fQ8tL))

**Abstract:** Traditional federated learning (FL) algorithms operate under the assumption that the data distributions at training (source domains) and testing (target domain) are the same. The fact that domain shifts often occur in practice necessitates equipping FL methods with a domain generalization (DG) capability. However, existing DG algorithms face fundamental challenges in FL setups due to the lack of samples/domains in each client’s local dataset. In this paper, we propose StableFDG, a style and attention-based learning strategy for accomplishing federated domain generalization, introducing two key contributions. The first is style-based learning, which enables each client to explore novel styles beyond the original source domains in its local dataset, improving domain diversity based on the proposed style sharing, shifting, and exploration strategies. Our second contribution is an attention-based feature highlighter, which captures the similarities between the features of data samples in the same class, and emphasizes the important/common characteristics to better learn the domain-invariant characteristics of each class in data-poor FL scenarios. Experimental results show that StableFDG outperforms existing baselines on various DG benchmark datasets, demonstrating its efficacy.


## Requirements

This code was tested on the following environments:

* Ubuntu 18.04
* Python 3.7.13
* PyTorch 1.12.0
* CUDA 11.6

You can install all necessary packages from requirements.txt

```
pip install -r requirements.txt
```

## Experiments

* This code is based on Dassl.pytorch. To install dassl, please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation
* The datasets will be automatically downloaded from specific URLs for each dataset. 
* Experimental settings (e.g., dataset, source/target domain, etc.) can be changed in the following bash file: ```./StableFDG.sh```

### How to Run

* ```cd``` to ```scripts/```

```bash

./StableFDG.sh

```


## Acknowledgement

Our code is built upon the implementations at 1) [https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/README.md](https://github.com/KaiyangZhou/mixstyle-release/tree/master/imcls) , 2) https://github.com/leaderj1001/Attention-Augmented-Conv2d.
