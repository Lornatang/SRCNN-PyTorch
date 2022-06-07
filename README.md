# SRCNN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092v3).

## Table of contents

- [SRCNN-PyTorch](#srcnn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
      - [Test](#test)
      - [Train SRCNN model](#train-srcnn-model)
      - [Resume train SRCNN model](#resume-train-srcnn-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Image Super-Resolution Using Deep Convolutional Networks](#image-super-resolution-using-deep-convolutional-networks)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file. 

### Test

- line 29: `upscale_factor` change to `2`.
- line 31: `mode` change to `test`.
- line 66: `model_path` change to `results/pretrained_models/srcnn_x2-T91-7d6e0623.pth.tar`.

### Train SRCNN model

- line 29: `upscale_factor` change to `2`.
- line 31: `mode` change to `train`.
- line 33: `exp_name` change to `SRCNN_x2`.

### Resume train SRCNN model

- line 31: `upscale_factor` change to `2`.
- line 33: `mode` change to `train`.
- line 35: `exp_name` change to `SRCNN_x2`.
- line 46: `resume` change to `samples/SRCNN_x2/epoch_xxx.pth.tar`.

## Result

Source of original paper results: [https://arxiv.org/pdf/1501.00092v3.pdf](https://arxiv.org/pdf/1501.00092v3.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

| Method | Scale |          Set5 (PSNR/SSIM)           |          Set14(PSNR/SSIM)           |     BSD200(PSNR/SSIM)      |
|:------:|:-----:|:-----------------------------------:|:-----------------------------------:|:--------------------------:|
| SRCNN  |   2   | 36.66(**36.72**)/0.9542(**0.9552**) | 32.45(**32.44**)/0.9067(**0.9066**) | 30.29(**-**)/0.8977(**-**) |
| SRCNN  |   3   | 32.75(**29.82**)/0.9090(**0.8904**) | 29.30(**27.42**)/0.8215(**0.8380**) | 27.18(**-**)/0.7971(**-**) |
| SRCNN  |   4   | 30.49(**25.34**)/0.8628(**0.7910**) | 27.50(**23.81**)/0.7513(**0.7366**) | 25.60(**-**)/0.7184(**-**) |

```bash
# Download `srcnn_x2-T91-7d6e0623.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python ./inference.py --inputs_path ./figure/butterfly_lr.png --output_path ./figure/butterfly_sr.png --weights_path ./results/pretrained_models/srcnn_x2-T91-7d6e0623.pth.tar
```

Inputs: <span align="center"><img width="252" height="252" src="figure/butterfly_lr.png"/></span>

Output: <span align="center"><img width="252" height="252" src="figure/butterfly_sr.png"/></span>

```text
Build SRCNN model successfully.
Load SRCNN model weights `./results/pretrained_models/srcnn_x2-T91-7d6e0623.pth.tar` successfully.
SR image save to `./figure/butterfly_sr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

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
