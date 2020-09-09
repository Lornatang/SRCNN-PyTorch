# SRCNN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092).

### Table of contents
1. [About Image Super-Resolution Using Deep Convolutional Networks](#about-image-super-resolution-using-deep-convolutional-networks)
2. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights)
    * [Download dataset](#download-dataset)
3. [Test](#test)
4. [Train](#train-eg-div2k)
    * [Example](#example-eg-div2k)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Image Super-Resolution Using Deep Convolutional Networks

If you're new to SRGAN, here's an abstract straight from the paper:

We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN) that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.


### Installation

#### Clone and install requirements

```bash
git clone https://github.com/Lornatang/SRCNN-PyTorch.git
cd SRCNN-PyTorch/
pip install -r requirements.txt
```

#### Download pretrained weights

```bash
cd weights/
bash download_weights.sh
```

#### Download dataset

```bash
cd data/
bash download_dataset.sh
```

### Test

Evaluate the overall performance of the network.
```bash
usage: test.py [-h] [--dataroot DATAROOT] [--weights WEIGHTS] [--cuda]
               [--scale-factor {2,3,4}] [--manualSeed MANUALSEED]

PyTorch Super Resolution CNN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   The directory address where the image needs to be
                        processed. (default: `./data/Set5`).
  --weights WEIGHTS     Generator model name. (default:`weights/srcnn_X4.pth`)
  --cuda                Enables cuda
  --scale-factor {2,3,4}
                        Image scaling ratio. (default: `4`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)

# Example
python test.py --dataroot ./data/Set5 --weights ./weights/srcnn_X4.pth --scale-factor 4 --cuda
```

Evaluate the benchmark of validation data set in the network
```bash
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N] [-b N]
                         [--scale-factor SCALE_FACTOR] [--cuda] --weights
                         WEIGHTS [--manualSeed MANUALSEED]

PyTorch Super Resolution CNN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/DIV2K`)
  -j N, --workers N     Number of data loading workers. (default:0)
  -b N, --batch-size N  mini-batch size (default: 8), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --scale-factor SCALE_FACTOR
                        Low to high resolution scaling factor. (default:4).
  --cuda                Enables cuda
  --weights WEIGHTS     Path to weights.
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)
# Example
python test_benchmark.py --dataroot ./data/DIV2K --weights ./weights/srcnn_X4.pth --scale-factor 4 --cuda
```

### Train (e.g DIV2K)

```bash
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--epochs N] [-b N]
                [--lr LR] [--scale-factor SCALE_FACTOR] [-p N] [--cuda]
                [--weights WEIGHTS] [--manualSeed MANUALSEED]

PyTorch Super Resolution CNN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/DIV2K`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --epochs N            Number of total epochs to run. (default:200)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.0001)
  --scale-factor SCALE_FACTOR
                        Low to high resolution scaling factor. (default:4).
  -p N, --print-freq N  Print frequency. (default:5)
  --cuda                Enables cuda
  --weights WEIGHTS     Path to weights (to continue training).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)
```

#### Example (e.g DIV2K)

```bash
python train.py --dataroot ./data/DIV2K --scale-factor 4 --cuda
```

If you want to load weights that you've trained before, run the following command.

```bash
python train.py --dataroot ./data/DIV2K --scale-factor 4 --weights ./weights/srcnn_epoch_100.pth --cuda
```

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
