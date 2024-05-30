# An Empirical Study of Pre-trained Model Selection for Out-of-Distribution Generalization and Calibration

## Abstract
In the realm of out-of-distribution (OOD) generalization tasks, fine-tuning pre-trained models has become a prevalent strategy. Different from most prior work that has focused on advancing learning algorithms, we systematically examined how pre-trained model size, pre-training data scale, and training strategies impact downstream generalization and uncertainty calibration. We evaluated 97 models across diverse pre-trained model sizes, five pre-training datasets, and five data augmentations through extensive experiments on four distribution shift datasets totaling over 100,000 GPU hours. Our results demonstrate the significant impact of pre-trained model selection, with optimal choices substantially improving OOD accuracy over algorithm improvement alone. We find larger models and bigger pre-training data improve OOD performance and calibration, in contrast to some prior studies that found modern deep networks to calibrate worse than classical shallow models. Our work underscores the overlooked importance of pre-trained model selection for out-of-distribution generalization and calibration.

## Prerequisites
- Python >= 3.6.5
- Pytorch >= 1.6.0
- cuDNN >= 7.6.2
- CUDA >= 10.0

## Downloads 
#### 4 Datasets For Domain Generalization Task
1. [VLCS](https://github.com/facebookresearch/DomainBed)
2. [PACS](https://github.com/facebookresearch/DomainBed)
3. [OfficeHome](https://github.com/facebookresearch/DomainBed)
4. [DomainNet](https://github.com/facebookresearch/DomainBed)
5. [WILDS Camelyon17](https://github.com/p-lambda/wilds)

#### Pre-Trained Models from Timm
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [docs](huggingface.co/docs/timm)

## Implementation
As for the DomainBed, we follow the official implementations shown in the links below.
- [DomainBed](https://github.com/facebookresearch/DomainBed)

## Citation 
- [Paper Link / Arxiv](https://arxiv.org/abs/2307.08187)

## Paper authors
- [Hiroki Naganuma \*](https://hiroki11x.github.io/)
- [Ryuichiro Hataya \*](https://mosko.tokyo/)
- [Ioannis Mitliagkas](http://mitliagkas.github.io/)


 \* denotes equal contribution

