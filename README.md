# SRCNN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092v3).

## Table of contents

- [SRCNN-PyTorch](#srcnn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Image Super-Resolution Using Deep Convolutional Networks](#about-image-super-resolution-using-deep-convolutional-networks)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
        - [Download train dataset](#download-train-dataset)
        - [Download test dataset](#download-test-dataset)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Image Super-Resolution Using Deep Convolutional Networks](#image-super-resolution-using-deep-convolutional-networks)

## About Image Super-Resolution Using Deep Convolutional Networks

If you're new to SRCNN, here's an abstract straight from the paper:

We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the
low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN)
that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods
can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes
all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical
on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend
our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

### Download train dataset

#### T91

- [Google Driver](https://drive.google.com/file/d/1_ED502vme5c6FgBaZZKF79CdWswEDAnq/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1UdlERjfIEU0cDX80XhKbTg?pwd=llot)

### Download test dataset

Contains Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `mode` change Set to valid mode.
- line 65: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 46: `start_epoch` change number of training iterations in the previous round.
- line 47: `resume` change weight address that needs to be loaded.

## Result

Source of original paper results: https://arxiv.org/pdf/1501.00092v3.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |
|:-------:|:-----:|:----------------:|
|  Set5   |   2   | 36.66(**36.66**) |
|  Set5   |   3   | 32.45(**32.62**) |
|  Set5   |   4   | 30.29(**30.20**) |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

## Credit

### Image Super-Resolution Using Deep Convolutional Networks

_Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang_ <br>

**Abstract** <br>
We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the
low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN)
that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods
can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes
all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical
on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend
our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

[[Paper]](https://arxiv.org/pdf/1501.00092) [[Author's implements(Caffe)]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_train.zip)

```bibtex
@misc{dong2014image,
    title={Image Super-Resolution Using Deep Convolutional Networks},
    author={Chao Dong and Chen Change Loy and Kaiming He and Xiaoou Tang},
    year={2014},
    eprint={1501.00092},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
