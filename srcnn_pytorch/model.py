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
from torch import nn
from torch.cuda import amp


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        # Patch extraction and representation.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

        # Non-linear mapping.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )

        # Reconstruction image.
        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

    @amp.autocast()
    def forward(self, inputs):
        out = self.features(inputs)
        out = self.map(out)
        out = self.reconstruction(out)
        return out
