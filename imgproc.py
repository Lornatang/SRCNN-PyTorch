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
"""Realize the function of processing the dataset before training."""
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

__all__ = [
    "normalize", "unnormalize",
    "image2tensor", "tensor2image",
    "convert_rgb_to_y", "convert_rgb_to_ycbcr", "convert_ycbcr_to_rgb",
    "center_crop", "random_crop",
    "random_rotate", "random_horizontally_flip", "random_vertically_flip",
    "random_adjust_brightness", "random_adjust_contrast"
]


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize the ``OpenCV.imread`` or ``skimage.io.imread`` data.

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread`` or ``skimage.io.imread``.

    Returns:
        np.ndarray: normalized image data. Data range [0, 1].

    """
    return image.astype(np.float64) / 255.0


def unnormalize(image: np.ndarray) -> np.ndarray:
    """Un-normalize the ``OpenCV.imread`` or ``skimage.io.imread`` data.

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread`` or ``skimage.io.imread``.

    Returns:
        np.ndarray: Denormalized image data. Data range [0, 255].

    """
    return image.astype(np.float64) * 255.0


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert ``PIL.Image`` to Tensor.

    Args:
        image (np.ndarray): The image data read by ``PIL.Image``
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        torch.Tensor: normalized image data

    Examples:
        >>> image = Image.open("image.bmp")
        >>> tensor_image = image2tensor(image, range_norm=False, half=False)

    """
    tensor = F.to_tensor(image)
    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)
    if half:
        tensor = tensor.half()

    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) -> np.ndarray:
    """Converts ``torch.Tensor`` to ``PIL.Image``.

    Args:
        tensor (torch.Tensor): The image that needs to be converted to ``PIL.Image``
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        np.ndarray: Convert image data to support PIL library

    Examples:
        >>> tensor = torch.randn([1, 3, 128, 128])
        >>> image = tensor2image(tensor, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add_(1.0).div_(2.0)
    if half:
        tensor = tensor.half()
    image = Image.fromarray(tensor.squeeze_(0).mul_(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())

    return image


def convert_rgb_to_y(image: np.ndarray or torch.Tensor) -> torch.Tensor:
    """Convert RGB image or tensor image data to YCbCr(Y) format.

    Args:
        image (np.ndarray): RGB image data read by ``PIL.Image''.

    Returns:
        np.ndarray or torch.Tensor: Y image array data.

    """
    if type(image) == np.ndarray:
        return 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
    elif type(image) == torch.Tensor:
        if len(image.shape) == 4:
            image = image.squeeze_(0)
        return 16. + (64.738 * image[0, :, :] + 129.057 * image[1, :, :] + 25.064 * image[2, :, :]) / 256.
    else:
        raise Exception("Unknown Type", type(image))


def convert_rgb_to_ycbcr(image: np.ndarray or torch.Tensor) -> torch.Tensor:
    """Convert RGB image or tensor image data to YCbCr format.

    Args:
        image (np.ndarray): RGB image data read by ``PIL.Image''.


    Returns:
        np.ndarray or torch.Tensor: YCbCr image array data.

    """
    if type(image) == np.ndarray:
        y = 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
        cb = 128. + (-37.945 * image[:, :, 0] - 74.494 * image[:, :, 1] + 112.439 * image[:, :, 2]) / 256.
        cr = 128. + (112.439 * image[:, :, 0] - 94.154 * image[:, :, 1] - 18.285 * image[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(image) == torch.Tensor:
        if len(image.shape) == 4:
            image = image.squeeze(0)
        y = 16. + (64.738 * image[0, :, :] + 129.057 * image[1, :, :] + 25.064 * image[2, :, :]) / 256.
        cb = 128. + (-37.945 * image[0, :, :] - 74.494 * image[1, :, :] + 112.439 * image[2, :, :]) / 256.
        cr = 128. + (112.439 * image[0, :, :] - 94.154 * image[1, :, :] - 18.285 * image[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(image))


def convert_ycbcr_to_rgb(image: np.ndarray):
    """Convert YCbCr format image to RGB format.

    Args:
       image (np.ndarray): YCbCr image data read by ``PIL.Image''.

    Returns:
        np.ndarray: RGB image.

    """
    if type(image) == np.ndarray:
        r = 298.082 * image[:, :, 0] / 256. + 408.583 * image[:, :, 2] / 256. - 222.921
        g = 298.082 * image[:, :, 0] / 256. - 100.291 * image[:, :, 1] / 256. - 208.120 * image[:, :, 2] / 256. + 135.576
        b = 298.082 * image[:, :, 0] / 256. + 516.412 * image[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(image) == torch.Tensor:
        if len(image.shape) == 4:
            image = image.squeeze(0)
        r = 298.082 * image[0, :, :] / 256. + 408.583 * image[2, :, :] / 256. - 222.921
        g = 298.082 * image[0, :, :] / 256. - 100.291 * image[1, :, :] / 256. - 208.120 * image[2, :, :] / 256. + 135.576
        b = 298.082 * image[0, :, :] / 256. + 516.412 * image[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(image))


def center_crop(lr: np.ndarray, hr: np.ndarray, image_size: int, upscale_factor: int) -> list[np.ndarray, np.ndarray]:
    """Cut ``PIL.Image`` in the center area of the image.

    Args:
        lr (np.ndarray): Low-resolution image data read by ``PIL.Image``.
        hr (np.ndarray): High-resolution image data read by ``PIL.Image``.
        image_size (int): The size of the captured image area. It should be the size of the high-resolution image.
        upscale_factor (int): magnification factor.

    Returns:
        Randomly cropped low-resolution images and high-resolution images.

    """
    w, h = hr.size

    left = (w - image_size) // 2
    top = (h - image_size) // 2
    right = left + image_size
    bottom = top + image_size

    lr = lr.crop((left // upscale_factor,
                  top // upscale_factor,
                  right // upscale_factor,
                  bottom // upscale_factor))
    hr = hr.crop((left, top, right, bottom))

    return lr, hr


def random_crop(lr: np.ndarray, hr: np.ndarray, image_size: int, upscale_factor: int) -> list[np.ndarray, np.ndarray]:
    """Will ``PIL.Image`` randomly capture the specified area of the image.

    Args:
        lr (np.ndarray): Low-resolution image data read by ``PIL.Image``.
        hr (np.ndarray): High-resolution image data read by ``PIL.Image``.
        image_size (int): The size of the captured image area. It should be the size of the high-resolution image.
        upscale_factor (int): magnification factor.

    Returns:
        Randomly cropped low-resolution images and high-resolution images.

    """
    w, h = hr.size
    left = torch.randint(0, w - image_size + 1, size=(1,)).item()
    top = torch.randint(0, h - image_size + 1, size=(1,)).item()
    right = left + image_size
    bottom = top + image_size

    lr = lr.crop((left // upscale_factor,
                  top // upscale_factor,
                  right // upscale_factor,
                  bottom // upscale_factor))
    hr = hr.crop((left, top, right, bottom))

    return lr, hr


def random_rotate(lr: np.ndarray, hr: np.ndarray, angle: int) -> list[np.ndarray, np.ndarray]:
    """Will ``PIL.Image`` randomly rotate the image.

    Args:
        lr (np.ndarray): Low-resolution image data read by ``PIL.Image``.
        hr (np.ndarray): High-resolution image data read by ``PIL.Image``.
        angle (int): rotation angle, clockwise and counterclockwise rotation.

    Returns:
        Randomly rotated low-resolution images and high-resolution images.

    """
    angle = random.choice((+angle, -angle))
    lr = F.rotate(lr, angle)
    hr = F.rotate(hr, angle)

    return lr, hr


def random_horizontally_flip(lr: np.ndarray, hr: np.ndarray, p=0.5) -> list[np.ndarray, np.ndarray]:
    """Flip the ``PIL.Image`` image horizontally randomly.

    Args:
        lr (np.ndarray): Low-resolution image data read by ``PIL.Image``.
        hr (np.ndarray): High-resolution image data read by ``PIL.Image``.
        p (optional, float): rollover probability. (Default: 0.5)

    Returns:
        Low-resolution image and high-resolution image after random horizontal flip.

    """
    if torch.rand(1).item() > p:
        lr = F.hflip(lr)
        hr = F.hflip(hr)

    return lr, hr


def random_vertically_flip(lr: np.ndarray, hr: np.ndarray, p=0.5) -> list[np.ndarray, np.ndarray]:
    """Turn the ``PIL.Image`` image upside down randomly.

    Args:
        lr (np.ndarray): Low-resolution image data read by ``PIL.Image``.
        hr (np.ndarray): High-resolution image data read by ``PIL.Image``.
        p (optional, float): rollover probability. (Default: 0.5)

    Returns:
        Randomly rotated up and down low-resolution images and high-resolution images.

    """
    if torch.rand(1).item() > p:
        lr = F.vflip(lr)
        hr = F.vflip(hr)

    return lr, hr


def random_adjust_brightness(lr: np.ndarray, hr: np.ndarray) -> list[np.ndarray, np.ndarray]:
    """Set ``PIL.Image`` to randomly adjust the image brightness.

    Args:
        lr (np.ndarray): Low-resolution image data read by ``PIL.Image``.
        hr (np.ndarray): High-resolution image data read by ``PIL.Image``.

    Returns:
        Low-resolution image and high-resolution image with randomly adjusted brightness.

    """
    # Randomly adjust the brightness gain range.
    factor = random.uniform(0.5, 2)
    lr = F.adjust_brightness(lr, factor)
    hr = F.adjust_brightness(hr, factor)

    return lr, hr


def random_adjust_contrast(lr: np.ndarray, hr: np.ndarray) -> list[np.ndarray, np.ndarray]:
    """Set ``PIL.Image`` to randomly adjust the image contrast.

    Args:
        lr (np.ndarray): Low-resolution image data read by ``PIL.Image``.
        hr (np.ndarray): High-resolution image data read by ``PIL.Image``.

    Returns:
        Low-resolution image and high-resolution image with randomly adjusted contrast.

    """
    # Randomly adjust the contrast gain range.
    factor = random.uniform(0.5, 2)
    lr = F.adjust_contrast(lr, factor)
    hr = F.adjust_contrast(hr, factor)

    return lr, hr
