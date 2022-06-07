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
import argparse

import cv2
import numpy as np
import torch

import config
import imgproc
from model import SRCNN


def main(args):
    # Initialize the model
    model = SRCNN()
    model = model.to(memory_format=torch.channels_last, device=config.device)
    print("Build SRCNN model successfully.")

    # Load the CRNN model weights
    checkpoint = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRCNN model weights `{args.weights_path}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    # Read LR image and HR image
    lr_image = cv2.imread(args.inputs_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    # Get Y channel image data
    lr_y_image = imgproc.bgr2ycbcr(lr_image, True)

    # Get Cb Cr image data from hr image
    lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, False)
    _, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_y_tensor = imgproc.image2tensor(lr_y_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_y_tensor = lr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Only reconstruct the Y channel image data.
    with torch.no_grad():
        sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

    # Save image
    sr_y_image = imgproc.tensor2image(sr_y_tensor, False, False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image, lr_cb_image, lr_cr_image])
    sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
    cv2.imwrite(args.output_path, sr_image * 255.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the SRCNN model generator super-resolution images.")
    parser.add_argument("--inputs_path", type=str, help="Low-resolution image path.")
    parser.add_argument("--output_path", type=str, help="Super-resolution image path.")
    parser.add_argument("--weights_path", type=str, help="Model weights file path.")
    args = parser.parse_args()

    main(args)
