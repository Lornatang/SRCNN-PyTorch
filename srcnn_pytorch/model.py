# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
# ==============================================================================
import math

from torch import nn
from torch.cuda import amp


class SRCNN(nn.Module):
    def __init__(self, init_weights=True):
        super(SRCNN, self).__init__()

        # Patch extraction and representation.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True)
        )

        # Non-linear mapping.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True)
        )

        # Reconstruction image.
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)

        if init_weights:
            self._initialize_weights()

    # The filter weights of each layer are initialized by drawing randomly from
    # a Gaussian distribution with a zero mean and a standard deviation of 0.001 (and a deviation of 0).
    def _initialize_weights(self):
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.map:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

    @amp.autocast()
    def forward(self, inputs):
        out = self.features(inputs)
        out = self.map(out)
        out = self.reconstruction(out)
        return out
