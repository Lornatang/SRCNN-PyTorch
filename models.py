# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from math import sqrt
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.hub import load_state_dict_from_url

__all__ = [
    "SRCNN", "srcnn_x2", "srcnn_x3", "srcnn_x4"
]


model_urls = {
    "srcnn_x2": "https://github.com/Lornatang/SRCNN-PyTorch/releases/download/v2.0/srcnn_x2.pth",
    "srcnn_x3": "https://github.com/Lornatang/SRCNN-PyTorch/releases/download/v2.0/srcnn_x3.pth",
    "srcnn_x4": "https://github.com/Lornatang/SRCNN-PyTorch/releases/download/v2.0/srcnn_x4.pth"
}


class SRCNN(nn.Module):
    r""" Construct SRCNN super-resolution model.
    
    Args:
        mode (optional, str): Because the SRCNN model is inconsistent in the training and testing mode.
                              If set to `train`, the convolutional layer does not need to fill the edge of 
                              the image, otherwise it is filled. (Default: `train`)
    """

    def __init__(self, mode: str = "train", init_weights: bool = True) -> None:
        super(SRCNN, self).__init__()
        # The model does not need to fill the edges during the training process, 
        # and needs to fill the edges during the testing mode.
        if mode == "train":
            padding = False
        elif mode == "eval":
            padding = True
        else:
            padding = True

        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 9, 1, 0 if not padding else 4),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.ReLU(True)
        )

        # Reconstruction the layer.
        self.reconstruction = nn.Conv2d(32, 1, 5, 1, 0 if not padding else 2)

        # Initialize model weights.
        if init_weights:
            self._initialize_weights()

    # The filter weights of each layer are initialized by random sampling and have 
    # a Gaussian distribution with zero mean and standard deviation of 0.001 (with a deviation of 0).
    def _initialize_weights(self) -> None:
        for m in self.features or self.map:
            if isinstance(m, nn.Conv2d):
                mean = 0.0
                std = sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out


def _srcnn(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> SRCNN:
    r""" `"Image Super-Resolution Using Deep Convolutional Networks" <https://arxiv.org/pdf/1501.00092v3.pdf>`_.
    
    Args:
        arch (str): SRCNN model architecture.
        pretrained (bool): Whether to load pre-trained model weights.
        progress (bool): If True, displays a progress bar of the download to stderr.

    Returns:
        torch.nn.Module.
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = SRCNN(**kwargs)
    # Load pre-training weights.
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def srcnn_x2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SRCNN:
    r""" `"Image Super-Resolution Using Deep Convolutional Networks" <https://arxiv.org/pdf/1501.00092v3.pdf>`_.
    
    Args:
        pretrained (optional, bool): Whether to load pre-trained model weights. (Default: `False`)
        progress (optional, bool): If `True`, displays a progress bar of the download to stderr.

    Returns:
        torch.nn.Module.
    """
    return _srcnn("srcnn_x2", pretrained, progress, **kwargs)


def srcnn_x3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SRCNN:
    r""" `"Image Super-Resolution Using Deep Convolutional Networks" <https://arxiv.org/pdf/1501.00092v3.pdf>`_.
    
    Args:
        pretrained (optional, bool): Whether to load pre-trained model weights. (Default: `False`)
        progress (optional, bool): If `True`, displays a progress bar of the download to stderr.

    Returns:
        torch.nn.Module.
    """
    return _srcnn("srcnn_x3", pretrained, progress, **kwargs)


def srcnn_x4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SRCNN:
    r""" `"Image Super-Resolution Using Deep Convolutional Networks" <https://arxiv.org/pdf/1501.00092v3.pdf>`_.
    
    Args:
        pretrained (optional, bool): Whether to load pre-trained model weights. (Default: `False`)
        progress (optional, bool): If `True`, displays a progress bar of the download to stderr.

    Returns:
        torch.nn.Module.
    """
    return _srcnn("srcnn_x4", pretrained, progress, **kwargs)
