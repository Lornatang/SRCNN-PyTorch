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
# ==============================================================================
import os
from typing import Tuple

import torch.utils.data
import torchvision.transforms.functional as F
import torchvision.transforms.functional_pil as F_pil
from PIL import Image
from torch import Tensor

__all__ = ["CustomDataset"]


class CustomDataset(torch.utils.data.Dataset):
    r""" Customize the data set loading function and prepare 
    low/high resolution image data in advance."""

    def __init__(self, dataroot: str) -> None:
        super(CustomDataset, self).__init__()
        # Get the index of all images in the high-resolution folder and 
        # low-resolution folder under the data set address.
        # Note: The high and low resolution file index should be corresponding.
        lr_dir_path = os.path.join(dataroot, "inputs")
        hr_dir_path = os.path.join(dataroot, "target")
        lr_filenames = os.listdir(lr_dir_path)
        hr_filenames = os.listdir(hr_dir_path)
        self.lr_filenames = [os.path.join(lr_dir_path, x) for x in lr_filenames]
        self.hr_filenames = [os.path.join(hr_dir_path, x) for x in hr_filenames]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = Image.open(self.lr_filenames[index]).convert("YCbCr")
        hr = Image.open(self.hr_filenames[index]).convert("YCbCr")

        # Data enhancement operation.
        if torch.rand(1).item() > 0.5:  # horizontal flip.
            lr = F_pil.hflip(lr)
            hr = F_pil.hflip(hr)
        if torch.rand(1).item() > 0.5:  # Flip up and down.
            lr = F_pil.vflip(lr)
            hr = F_pil.vflip(hr)

        # Only extract the image data of the Y channel.
        lr, _, _ = lr.split()
        hr, _, _ = hr.split()
        # Array data is converted to Tensor format.
        lr = F.to_tensor(lr)
        hr = F.to_tensor(hr)

        return lr, hr

    def __len__(self) -> int:
        return len(self.lr_filenames)
