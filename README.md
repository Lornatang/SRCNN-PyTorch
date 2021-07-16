# SRCNN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092v3).

### Table of contents

- [SRCNN-PyTorch](#srcnn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Image Super-Resolution Using Deep Convolutional Networks](#about-image-super-resolution-using-deep-convolutional-networks)
    - [Installation](#installation)
      - [Clone and install requirements](#clone-and-install-requirements)
      - [Download all datasets](#download-all-datasets)
        - [Train datasets](#train-datasets)
        - [Test datasets](#test-datasets)
    - [Test](#test)
    - [Train (e.g T91)](#train-eg-t91)
    - [Model performance](#model-performance)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
      - [Image Super-Resolution Using Deep Convolutional Networks](#image-super-resolution-using-deep-convolutional-networks)

### About Image Super-Resolution Using Deep Convolutional Networks

If you're new to SRCNN, here's an abstract straight from the paper:

We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN) that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

### Installation

#### Clone and install requirements

```bash
git clone https://github.com/Lornatang/SRCNN-PyTorch.git
cd SRCNN-PyTorch/
pip install -r requirements.txt
```

#### Download all datasets

##### Train datasets

**T91 dataset**

The downloaded data set is placed in the `data` directory.

- [baiduclouddisk](https://pan.baidu.com/s/13fHKvBS6CKWbor9VjdrdKg) access: `llot`
- [googleclouddisk](https://drive.google.com/file/d/1qCxnfiqIEIMy6K5jy0wZm8K8294lDmOU/view?usp=sharing)

##### Test datasets

**Set5 dataset**

- [baiduclouddisk](https://pan.baidu.com/s/1_B97Ga6thSi5h43Wuqyw0Q) access: `llot`
- [gooleclouddisk](https://drive.google.com/file/d/10aObmC4_UtTui2luzBNcWjfoGkeSJi2G/view?usp=sharing)

**Set14 dataset**

- [baiduclouddisk](https://pan.baidu.com/s/1wy_kf4Kkj2nSkgRUkaLzVA) access: `llot`
- [googlecloudisk](https://drive.google.com/file/d/1-3xXHunN_WqTo1c1jVWCJa_0ZG-1LTdv/view?usp=sharing)

**BSD100 dataset**

- [baiduclouddisk](https://pan.baidu.com/s/1Ig8t3_G4Nzhl8MvPAvdzFA) access: `llot`
- [googlecloudisk](https://drive.google.com/file/d/1EVba9kKtXAbmV6esnfADjH1Ul-uThOeE/view?usp=sharing)


### Test

```text
usage: validate.py [-h] [--lr-dir LR_DIR] [--sr-dir SR_DIR] [--hr-dir HR_DIR] [--arch ARCH] [--scale {2,3,4}]
                   [--pretrained] [--model-path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --lr-dir LR_DIR       Path to lr datasets. (Default: `./data/Set5/LRbicx4`)
  --sr-dir SR_DIR       Path to sr datasets. (Default: `./sample/Set5`)
  --hr-dir HR_DIR       Path to hr datasets. (Default: `./data/Set5/GTmod12`)
  --arch ARCH           model architecture: srcnn_x2 | srcnn_x3 | srcnn_x4 (Default: `srcnn_x4`)
  --scale {2,3,4}       Low to high resolution scaling factor. (Default: 4)
  --pretrained          Use pre-trained model.
  --model-path MODEL_PATH
                        Path to weights.

# Example (Set5 dataset)
python validate.py --pretrained
```

### Train (e.g T91)

```text
usage: train.py [-h] [--dataroot DATAROOT] [--epochs N] [--batch-size BATCH_SIZE] [--lr LR] [--arch ARCH]
                [--scale {2,3,4}] [--pretrained] [--model-path MODEL_PATH] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (Default: `./data/T91/LRbicx4`)
  --epochs N            Number of total epochs to run. According to the 1e8 iters in the original paper.(Default:
                        4096)
  --batch-size BATCH_SIZE
                        mini-batch size (Default: 128)
  --lr LR               Learning rate. (Default: 0.0001)
  --arch ARCH           model architecture: srcnn_x2 | srcnn_x3 | srcnn_x4 (Default: `srcnn_x4`)
  --scale {2,3,4}       Low to high resolution scaling factor.
  --pretrained          Use pre-trained model.
  --model-path MODEL_PATH
                        Path to weights.
  --seed SEED           Seed for initializing training. (Default: 666)

# Example
python train.py
# If you want to load weights that you've trained before, run the following command.
python train.py --model-path sample/srcnn_x4_epoch10.pth
```

### Model performance

| Model | Params | FLOPs | CPU Speed | GPU Speed |
| :---: | :----: | :---: | :-------: | :-------: |
| srcnn | 0.01M  | 1.07G |   17ms    |    1ms    |

```text
# Example (CPU: Intel i9-10900X/GPU: Nvidia GeForce RTX 2080Ti)
python cal_model_complexity.py
```

### Result

Source of original paper results: https://arxiv.org/pdf/1609.04802v5.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |   PSNR   |   SSIM   |
| :-----: | :---: | :------: | :------: |
|  Set5   |   2   | -(**-**) | -(**-**) |
|  Set14  |   2   | -(**-**) | -(**-**) |
|  Set5   |   3   | -(**-**) | -(**-**) |
|  Set14  |   3   | -(**-**) | -(**-**) |
|  Set5   |   4   | -(**-**) | -(**-**) |
|  Set14  |   4   | -(**-**) | -(**-**) |


Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png" alt=""></span>

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Image Super-Resolution Using Deep Convolutional Networks

_Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang_ <br>

**Abstract** <br>
We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN) that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

[[Paper]](https://arxiv.org/pdf/1501.00092) [[Author's implements(Caffe)]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_train.zip)

```
@misc{dong2014image,
    title={Image Super-Resolution Using Deep Convolutional Networks},
    author={Chao Dong and Chen Change Loy and Kaiming He and Xiaoou Tang},
    year={2014},
    eprint={1501.00092},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
````
