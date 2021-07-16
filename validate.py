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
import argparse
import os

import numpy as np
import skimage.color
import skimage.io
import skimage.metrics
import torch
import torchvision.transforms as transforms
from PIL import Image

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--lr-dir", type=str, default="./data/Set5/LRbicx4",
                    help="Path to lr datasets. (Default: `./data/Set5/LRbicx4`)")
parser.add_argument("--sr-dir", type=str, default="./sample/Set5",
                    help="Path to sr datasets. (Default: `./sample/Set5`)")
parser.add_argument("--hr-dir", type=str, default="./data/Set5/GTmod12",
                    help="Path to hr datasets. (Default: `./data/Set5/GTmod12`)")
parser.add_argument("--arch", metavar="ARCH", default="srcnn_x4",
                    choices=model_names,
                    help="model architecture: " +
                        "srcnn_x2 | srcnn_x3 | srcnn_x4"
                        " (Default: `srcnn_x4`)")
parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor. (Default: 4)")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--model-path", type=str, default="",
                    help="Path to weights.")
args = parser.parse_args()

# Set the operating device model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sr(model, lr_filename, sr_filename):
    r""" Turn low resolution into super resolution.
    
    Args:
        model (torch.nn.Module): Super-resolution model.
        lr_filename (str): Low resolution image address.
        sr_filename (srt): High resolution image address.
    """
    with torch.no_grad():
        image = Image.open(lr_filename).convert("YCbCr")
        new_image_size = (image.size[0] * args.scale, image.size[1] * args.scale)
        # Use BICUBIC to zoom the image to the specified size.
        image = image.resize(new_image_size, Image.BICUBIC)
        # Extract the Y channel input of the image and convert it to Tensor format.
        y, cb, cr = image.split()        
        image_tensor = transforms.ToTensor()(y)
        image_tensor = image_tensor.view(1, -1, y.size[1], y.size[0]).to(device)
        
        # The Y channel data is super-divided and converted to PIL format.
        out = model(image_tensor)
        out_y = out[0].cpu().numpy()
        out_y *= 255.0
        out_y = out_y.clip(0, 255)
        out_y = Image.fromarray(np.uint8(out_y[0]), mode="L")
        out_image = Image.merge("YCbCr", (out_y, cb, cr)).convert("RGB")
        # Save the super-resolution image to the specified location.
        out_image.save(sr_filename)


def image_similarity_measures(sr_filename, hr_filename):
    r""" Image similarity measures function.
    
    Args:
        lr_filename (str): Low resolution image address.
        sr_filename (srt): High resolution image address.

    Returns:
        PSNR value(float), SSIM value(float).
    """
    sr_image = skimage.io.imread(sr_filename)
    hr_image = skimage.io.imread(hr_filename)

    # Delete 4 pixels around the image to facilitate PSNR calculation.
    sr_image = sr_image[4:-4, 4:-4, ...]
    hr_image = hr_image[4:-4, 4:-4, ...]

    # Calculate the Y channel of the image. Use the Y channel to 
    # calculate PSNR and SSIM instead of using RGB three channels.
    sr_image = sr_image / 255.0
    hr_image = hr_image / 255.0
    sr_image = skimage.color.rgb2ycbcr(sr_image)[:, :, 0:1]
    hr_image = skimage.color.rgb2ycbcr(hr_image)[:, :, 0:1]
    # Because rgb2ycbcr() outputs a floating point type and the range is [0, 255], 
    # it needs to be renormalized to [0, 1].
    sr_image = sr_image / 255.0
    hr_image = hr_image / 255.0

    psnr = skimage.metrics.peak_signal_noise_ratio(sr_image, hr_image)
    ssim = skimage.metrics.structural_similarity(sr_image,
                                                 hr_image,
                                                 win_size=11,
                                                 gaussian_weights=True,
                                                 multichannel=True,
                                                 data_range=1.0,
                                                 K1=0.01,
                                                 K2=0.03,
                                                 sigma=1.5)

    return psnr, ssim


def main():
    # Initialize the image evaluation index.
    avg_psnr = 0.0
    avg_ssim = 0.0

    # Load the model and weights.
    if args.pretrained:
        print(f"=> Using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True, mode="eval")
    else:
        print(f"=> Creating model '{args.arch}'")
        model = models.__dict__[args.arch](mode="eval")

    if args.model_path != "":
        print(f"=> Loading weights from `{args.model_path}`.")
        model.load_state_dict(torch.load(args.model_path, torch.device("cpu")))

    # Switch model to specifal device.
    model = model.to(device)

    # Get the test image file index.
    filenames = os.listdir(args.lr_dir)

    for index in range(len(filenames)):
        lr_filename = os.path.join(args.lr_dir, filenames[index])
        sr_filename = os.path.join(args.sr_dir, filenames[index])
        hr_filename = os.path.join(args.hr_dir, filenames[index])

        # Process low-resolution images into super-resolution images.
        sr(model, lr_filename, sr_filename)

        # Test the image quality difference between the super-resolution image 
        # and the original high-resolution image.
        psnr, ssim = image_similarity_measures(sr_filename, hr_filename)
        avg_psnr += psnr
        avg_ssim += ssim

    # Calculate the average index value of the image quality of the test dataset.
    avg_psnr = avg_psnr / len(filenames)
    avg_ssim = avg_ssim / len(filenames)

    print(f"=> Avg PSNR: {avg_psnr:.2f}dB.")
    print(f"=> Avg SSIM: {avg_ssim:.4f}.")


if __name__ == "__main__":
    main()
