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

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def check_image_file(filename):
    r"""Determine whether the files in the directory are in image format.

    Args:
        filename (str): The current path of the image

    Returns:
        Returns True if it is an image and False if it is not.

    """
    return any(filename.endswith(extension) for extension in [".bmp", ".BMP",
                                                              ".jpg", ".JPG",
                                                              ".png", ".PNG",
                                                              ".jpeg", ".JPEG"])


class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, target_dir, src_size=33, dst_size=21, upscale_factor=4):
        r""" Dataset loading base class.

        Args:
            data_dir (str): The directory address where the data image is stored.
            target_dir (str): The directory address where the target image is stored.
            src_size (int): Size of low resolution image. (Default: 33).
            dst_size (int): Size of high resolution image. (Default: 21).
            upscale_factor (int): How many times is the image upscale. (Default: 2).
        """
        super(DatasetFromFolder, self).__init__()
        # Traverse the image files under the dataset and add their absolute paths to the list.
        self.data_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if check_image_file(x)]
        self.target_filenames = [os.path.join(target_dir, x) for x in os.listdir(target_dir) if check_image_file(x)]

        # Normalize a tensor image with mean and standard deviation [-1, 1]
        self.data_transform = transforms.Compose([
            transforms.Resize((src_size * upscale_factor, src_size * upscale_factor), interpolation=Image.BICUBIC),
            transforms.Resize((src_size // upscale_factor, src_size // upscale_factor), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((dst_size, dst_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image and bicubic high resolution image.

        """
        inputs = Image.open(self.data_filenames[index]).convert("YCbCr")
        target = Image.open(self.target_filenames[index]).convert("YCbCr")

        inputs, _, _ = inputs.split()
        target, _, _ = target.split()

        inputs = self.data_transform(inputs)
        target = self.target_transform(target)

        return inputs, target

    def __len__(self):
        """

        Returns:
            Number of returned dataset files.
        """
        return len(self.data_filenames)
