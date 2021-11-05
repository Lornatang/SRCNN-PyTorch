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
"""Realize the function of dataset preparation."""
import io
import os

import lmdb
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

import imgproc

__all__ = ["BaseDataset", "LMDBDataset"]


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

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        lr_image_data = Image.open(self.lr_filenames[batch_index])
        hr_image_data = Image.open(self.hr_filenames[batch_index])

        # RGB convert YCbCr
        lr_ycbcr_image_data = lr_image_data.convert("YCbCr")
        hr_ycbcr_image_data = hr_image_data.convert("YCbCr")

        # Only extract the image data of the Y channel
        lr_y_image_data = lr_ycbcr_image_data.split()[0]
        hr_y_image_data = hr_ycbcr_image_data.split()[0]

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor_data = imgproc.image2tensor(lr_y_image_data, range_norm=False, half=False)
        hr_tensor_data = imgproc.image2tensor(hr_y_image_data, range_norm=False, half=False)

        return lr_tensor_data, hr_tensor_data

    def __len__(self) -> int:
        return len(self.filenames)


class LMDBDataset(Dataset):
    """Load the data set as a data set in the form of LMDB.

    Attributes:
        lr_datasets (list): Low-resolution image data in the dataset
        hr_datasets (list): High-resolution image data in the dataset

    """

    def __init__(self, lr_lmdb_path, hr_lmdb_path) -> None:
        super(LMDBDataset, self).__init__()
        # Create low/high resolution image array
        self.lr_datasets = []
        self.hr_datasets = []

        # Initialize the LMDB database file address
        self.lr_lmdb_path = lr_lmdb_path
        self.hr_lmdb_path = hr_lmdb_path

        # Write image data in LMDB database to memory
        self.read_lmdb_dataset()

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        lr_image_data = self.lr_datasets[batch_index]
        hr_image_data = self.hr_datasets[batch_index]

        # RGB convert YCbCr
        lr_ycbcr_image_data = lr_image_data.convert("YCbCr")
        hr_ycbcr_image_data = hr_image_data.convert("YCbCr")

        # Only extract the image data of the Y channel
        lr_y_image_data = lr_ycbcr_image_data.split()[0]
        hr_y_image_data = hr_ycbcr_image_data.split()[0]

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor_data = imgproc.image2tensor(lr_y_image_data, range_norm=False, half=False)
        hr_tensor_data = imgproc.image2tensor(hr_y_image_data, range_norm=False, half=False)

        return lr_tensor_data, hr_tensor_data

    def __len__(self) -> int:
        return len(self.hr_datasets)

    def read_lmdb_dataset(self) -> [list, list]:
        # Open two LMDB database writing environments to read low/high image data
        lr_lmdb_env = lmdb.open(self.lr_lmdb_path)
        hr_lmdb_env = lmdb.open(self.hr_lmdb_path)

        # Write the image data in the low-resolution LMDB data set to the memory
        for _, image_bytes in lr_lmdb_env.begin().cursor():
            image = Image.open(io.BytesIO(image_bytes))
            self.lr_datasets.append(image)

        # Write the image data in the high-resolution LMDB data set to the memory
        for _, image_bytes in hr_lmdb_env.begin().cursor():
            image = Image.open(io.BytesIO(image_bytes))
            self.hr_datasets.append(image)
