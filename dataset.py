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

# ==============================================================================
# File description: Realize the function of data set preparation.
# ==============================================================================
import os
from typing import Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from imgproc import image2tensor

__all__ = ["BaseDataset"]


class BaseDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        dataroot (str): training data set address.
    """

    def __init__(self, dataroot: str) -> None:
        super(BaseDataset, self).__init__()
        # Get the index of all images in the high-resolution folder and low-resolution folder under the data set address.
        # Note: The high and low resolution file index should be corresponding.
        lr_dir_path = os.path.join(dataroot, "inputs")
        hr_dir_path = os.path.join(dataroot, "target")
        self.filenames = os.listdir(lr_dir_path)
        self.lr_filenames = [os.path.join(lr_dir_path, x) for x in self.filenames]
        self.hr_filenames = [os.path.join(hr_dir_path, x) for x in self.filenames]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = Image.open(self.lr_filenames[index]).convert("YCbCr")
        hr = Image.open(self.hr_filenames[index]).convert("YCbCr")

        # Only extract the image data of the Y channel.
        lr, _, _ = lr.split()
        hr, _, _ = hr.split()

        # `PIL.Image` image data is converted to `Tensor` format data.
        lr = image2tensor(lr)
        hr = image2tensor(hr)

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)
