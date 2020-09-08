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
from os import listdir
from os.path import join

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    r"""Determine whether the files in the directory are in image format.

    Args:
        filename (str): The current path of the image

    Returns:
        Returns True if it is an image and False if it is not.

    """
    return any(
        filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DatasetFromFolder(Dataset):
    def __init__(self, images_dir, scale_factor):
        r""" Dataset loading base class.

        Args:
            images_dir (str): The directory address where the image is stored.
            scale_factor (int): Coefficient of image scale.
        """
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(images_dir, x) for x in listdir(images_dir)
                                if is_image_file(x)]

        crop_size = 32 - (32 % scale_factor)  # Valid crop size
        self.input_transform = transforms.Compose(
            [transforms.CenterCrop(crop_size),  # cropping the image
             transforms.Resize(crop_size // scale_factor),
             # subsampling the image (half size)
             transforms.Resize(crop_size, interpolation=Image.BICUBIC),
             # bicubic upsampling to get back the original size
             transforms.ToTensor()])
        self.target_transform = transforms.Compose(
            [transforms.CenterCrop(crop_size),
             # since it's the target, we keep its original quality
             transforms.ToTensor()])

    def __getitem__(self, index):
        r""" Get image source file

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image and high resolution image.

        """
        image = Image.open(self.image_filenames[index]).convert("YCbCr")
        inputs, _, _ = image.split()
        target = inputs.copy()

        inputs = self.input_transform(inputs)
        target = self.target_transform(target)

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)
