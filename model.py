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

# ============================================================================
# File description: Realize the model definition function.
# ============================================================================
from math import sqrt

from torch import Tensor
from torch import nn


class SRCNN(nn.Module):
    r""" Constructing a super-resolution model of SRCNN
    
    Args:
        mode (optional, str): Because the SRCNN model is inconsistent in the training and testing phases.
                              If set to `train`, the convolutional layer does not need to fill the edge of the image, otherwise it is filled.
                              (Default: `train`)
    """

    def __init__(self, mode: str = "train") -> None:
        super(SRCNN, self).__init__()
        # The model does not need to fill the edges during the training process, and needs to fill the edges during the testing phase.
        if mode == "train":
            padding = False
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

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, 5, 1, 0 if not padding else 2)

        # Initialize model weights.
        self._initialize_weights()

    # The filter weight of each layer is a Gaussian distribution with zero mean and standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.features or self.map:
            if isinstance(m, nn.Conv2d):
                mean = 0.0
                std = sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

    # The tracking operator in the PyTorch model must be written like this.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
