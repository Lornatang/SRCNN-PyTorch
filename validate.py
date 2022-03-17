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
"""File description: Realize the verification function after model training."""
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from model import SRCNN


def main() -> None:
    # Initialize the super-resolution model
    model = SRCNN().to(config.device)
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

    # Initialize the image evaluation index.
    total_psnr = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.hr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Make low-resolution image
        lr_image = cv2.imread(lr_image_path).astype(np.float32) / 255.0
        lr_image_height, lr_image_width = lr_image.shape[:2]
        lr_image_height_remainder = lr_image_height % config.upscale_factor
        lr_image_width_remainder = lr_image_width % config.upscale_factor
        lr_image = lr_image[:lr_image_height - lr_image_height_remainder, :lr_image_width - lr_image_width_remainder, ...]
        lr_image = imgproc.imresize(lr_image, 1 / config.upscale_factor)
        lr_image = imgproc.imresize(lr_image, config.upscale_factor)

        # Make high-resolution image
        hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.0
        hr_image_height, hr_image_width = hr_image.shape[:2]
        hr_image_height_remainder = hr_image_height % config.upscale_factor
        hr_image_width_remainder = hr_image_width % config.upscale_factor
        hr_image = hr_image[:hr_image_height - hr_image_height_remainder, :hr_image_width - hr_image_width_remainder, ...]

        # Convert BGR image to YCbCr image
        lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

        # Split YCbCr image data
        lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
        hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert Y image data convert to Y tensor data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        # Cal PSNR
        total_psnr += 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

    print(f"PSNR: {total_psnr / total_files:4.2f}dB.\n")


if __name__ == "__main__":
    main()
