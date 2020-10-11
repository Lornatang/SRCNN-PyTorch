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
import os

import torch.utils.data.dataset
import torchvision.transforms as transforms
from PIL import Image

from .utils import img2tensor


def check_image_file(filename):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.

    """
    return any(filename.endswith(extension) for extension in ["bmp", ".png",
                                                              ".jpg", ".jpeg",
                                                              ".png", ".PNG",
                                                              ".jpeg", ".JPEG"])


class DatasetFromFolder(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, input_dir, target_dir):
        r"""

        Args:
            input_dir (str): The directory address where the data image is stored.
            target_dir (str): The directory address where the target image is stored.
        """
        super(DatasetFromFolder, self).__init__()
        self.input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        self.target_filenames = [os.path.join(target_dir, x) for x in os.listdir(target_dir) if check_image_file(x)]
        self.input_transforms = transforms.Compose([
            transforms.Resize((33, 33), interpolation=Image.BICUBIC),
            img2tensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])
        self.target_transforms = transforms.Compose([
            img2tensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """

        input = Image.open(self.input_filenames[index]).convert("YCbCr")
        target = Image.open(self.target_filenames[index]).convert("YCbCr")

        input, _, _ = input.split()
        target, _, _ = target.split()

        input = self.input_transforms(input)
        target = self.target_transforms(target)

        return input, target

    def __len__(self):
        return len(self.input_filenames)
