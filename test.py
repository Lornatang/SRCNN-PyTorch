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
import argparse
import os
import random

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from sewar.full_ref import msssim
from sewar.full_ref import sam
from sewar.full_ref import vifp

from srcnn_pytorch import SRCNN
from srcnn_pytorch import cal_mse
from srcnn_pytorch import cal_niqe
from srcnn_pytorch import cal_psnr
from srcnn_pytorch import cal_rmse
from srcnn_pytorch import cal_ssim

parser = argparse.ArgumentParser(description="PyTorch Super Resolution CNN.")
parser.add_argument("--dataroot", type=str, default="./data/Set5",
                    help="The directory address where the image needs "
                         "to be processed. (default: `./data/Set5`).")
parser.add_argument("--weights", type=str, default="weights/srcnn_X4.pth",
                    help="Generator model name.  "
                         "(default:`weights/srcnn_X4.pth`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 3, 4],
                    help="Image scaling ratio. (default: `4`).")

args = parser.parse_args()
print(args)

try:
    os.makedirs("result")
except OSError:
    pass

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = SRCNN().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Evaluate algorithm performance
total_mse_value = 0.0
total_rmse_value = 0.0
total_psnr_value = 0.0
total_ssim_value = 0.0
total_ms_ssim_value = 0.0
total_niqe_value = 0.0
total_sam_value = 0.0
total_vif_value = 0.0
# Count the number of files in the directory
total_file = 0

dataroot = f"{args.dataroot}/X{args.scale_factor}/data"
target = f"{args.dataroot}/X{args.scale_factor}/target"
scale_factor = args.scale_factor

for filename in os.listdir(dataroot):
    image = Image.open(f"{dataroot}/{filename}").convert("YCbCr")
    image_width = int(image.size[0] * scale_factor)
    image_height = int(image.size[1] * scale_factor)
    image = image.resize((image_width, image_height), Image.BICUBIC)
    y, cb, cr = image.split()

    preprocess = transforms.ToTensor()
    inputs = preprocess(y).view(1, -1, y.size[1], y.size[0])

    inputs = inputs.to(device)

    out = model(inputs)
    out = out.cpu()
    out_image_y = out[0].detach().numpy()
    out_image_y *= 255.0
    out_image_y = out_image_y.clip(0, 255)
    out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")

    out_img = Image.merge("YCbCr", [out_image_y, cb, cr]).convert("RGB")
    # before converting the result in RGB
    out_img.save(f"result/{filename}")

    # Evaluate performance
    src_img = cv2.imread(f"result/{filename}")
    dst_img = cv2.imread(f"{target}/{filename}")

    total_mse_value += cal_mse(src_img, dst_img)
    total_rmse_value += cal_rmse(src_img, dst_img)
    total_psnr_value += cal_psnr(src_img, dst_img)
    total_ssim_value += cal_ssim(src_img, dst_img)
    total_ms_ssim_value += msssim(src_img, dst_img)
    total_niqe_value += cal_niqe(f"result/{filename}")
    total_sam_value += sam(src_img, dst_img)
    total_vif_value += vifp(src_img, dst_img)

    total_file += 1

print(f"Avg MSE: {total_mse_value / total_file:.2f}\n"
      f"Avg RMSE: {total_rmse_value / total_file:.2f}\n"
      f"Avg PSNR: {total_psnr_value / total_file:.2f}\n"
      f"Avg SSIM: {total_ssim_value / total_file:.4f}\n"
      f"Avg MS-SSIM: {total_ms_ssim_value / total_file:.4f}\n"
      f"Avg NIQE: {total_niqe_value / total_file:.2f}\n"
      f"Avg SAM: {total_sam_value / total_file:.4f}\n"
      f"Avg VIF: {total_vif_value / total_file:.4f}")
