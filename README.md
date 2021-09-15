# SRCNN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092v3).

## Table of contents

- [SRCNN-PyTorch](#srcnn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Image Super-Resolution Using Deep Convolutional Networks](#about-image-super-resolution-using-deep-convolutional-networks)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
        - [Download train dataset](#download-train-dataset)
        - [Download val dataset](#download-val-dataset)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Image Super-Resolution Using Deep Convolutional Networks](#image-super-resolution-using-deep-convolutional-networks)

## About Image Super-Resolution Using Deep Convolutional Networks

If you're new to SRCNN, here's an abstract straight from the paper:

We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end
mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN)
that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional
sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that
handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet
demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore
different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we
extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/1zPGktAZtph5aMR_gJdV5Q6S5gJEDvY8r?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1n04tkTauMGLUp4asO1cY3w) access:`llot`

## Download datasets

### Download train dataset

- [Google Driver](https://drive.google.com/drive/folders/1PYizfnKq-UtRCDoSy79PGA4FC5HqAqch?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1Oa1oas0GOT78DX1IAX7svg) access: `llot`

### Download val dataset

Set5 dataset:

- [Google Driver](https://drive.google.com/file/d/1GJZztdiJ6oBmJe9Ntyyos_psMzM8KY4P/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1_B97Ga6thSi5h43Wuqyw0Q) access:`llot`

Set14 dataset:

- [Google Driver](https://drive.google.com/file/d/14bxrGB3Nej8vBqxLoqerGX2dhChQKJoa/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1wy_kf4Kkj2nSkgRUkaLzVA) access:`llot`

Bsd100 dataset:

- [Google Driver](https://drive.google.com/file/d/1RTlPATPBCfUufJspgTik5KUEzAuVcyFF/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1Ig8t3_G4Nzhl8MvPAvdzFA) access:`llot`

## Test

Modify the contents of the file as follows.

1. `config.py` line 32 `mode="train"` change to `mode="valid"`.
2. `config.py` line 81 `model.load_state_dict(torch.load(f"results/{exp_name}/g-best.pth", map_location=device))` change to `model.load_state_dict(torch.load("<YOUR-WEIGHTS-PATH>", map_location=device))`.
3. Run `python validate.py`.

## Train

Modify the contents of the file as follows.

1. `config.py` line 32 `mode="valid"` change to `mode="train"`.
2. Run `python train.py`.

If you want to load weights that you've trained before, modify the contents of the file as follows.

1. `config.py` line 32 `mode="valid"` change to `mode="train"`.
2. `config.py` line 49 `start_epoch=0` change to `start_epoch=<RESUME-EPOCH>`.
3. `config.py` line 50 `resume=False` change to `resume=True`.
4. `config.py` line 51 `resume_weight=""` change to `resume_weight="<YOUR-RESUME-WIGHTS-PATH>"`.
5. Run `python train.py`.

## Result

Source of original paper results: https://arxiv.org/pdf/1501.00092v3.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |        SSIM        |
| :-----: | :---: | :--------------: | :----------------: |
|  Set5   |   2   | 36.66(**36.25**) | 0.9542(**0.9537**) |
|  Set14  |   2   | 32.45(**31.99**) | 0.9067(**0.9051**) |
|  Set5   |   3   | 32.75(**32.51**) | 0.9090(**0.9085**) |
|  Set14  |   3   | 29.30(**28.83**) | 0.8215(**0.8199**) |
|  Set5   |   4   | 30.49(**30.12**) | 0.8628(**0.8592**) |
|  Set14  |   4   | 27.50(**27.03**) | 0.7513(**0.7489**) |

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png"/></span>

## Credit

### Image Super-Resolution Using Deep Convolutional Networks

_Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang_ <br>

**Abstract** <br>
We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end
mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN)
that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional
sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that
handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet
demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore
different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we
extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

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
