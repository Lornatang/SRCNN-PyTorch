# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from image_quality_assessment import PSNR, SSIM
from model import SRCNN


def main() -> None:
    # Initialize the super-resolution model
    model = SRCNN().to(device=config.device, memory_format=torch.channels_last)
    print("Build SRCNN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRCNN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the sharpness evaluation function
    psnr = PSNR(config.upscale_factor, False)
    ssim = SSIM(config.upscale_factor, False)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image and HR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        lr_image = imgproc.image_resize(lr_image, 1 / config.upscale_factor)
        lr_image = imgproc.image_resize(lr_image, config.upscale_factor)

        # Get Y channel image data
        lr_y_image = imgproc.bgr2ycbcr(lr_image, True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image, True)

        # Get Cb Cr image data from hr image
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, False)
        _, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, False, True).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, False, True).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_y_tensor = lr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr_y_tensor = hr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, False, True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_y_tensor, hr_y_tensor).item()
        ssim_metrics += ssim(sr_y_tensor, hr_y_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} dB\n"
          f"SSIM: {avg_ssim:4.4f} u")


if __name__ == "__main__":
    main()
