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
import math
import os

import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm

from srcnn_pytorch import DatasetFromFolder
from srcnn_pytorch import SRCNN
from srcnn_pytorch import select_device

parser = argparse.ArgumentParser(description="Image Super-Resolution Using "
                                             "Deep Convolutional Networks.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--weights", default="./weights/SRCNN.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weights/SRCNN.pth``).")
parser.add_argument("--device", default="",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")

args = parser.parse_args()
print(args)

try:
    os.makedirs("weights")
except OSError:
    pass

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=1)

dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/val/input",
                            target_dir=f"{args.dataroot}/val/target")

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         pin_memory=True,
                                         num_workers=int(args.workers))
# Construct SRCNN model.
model = SRCNN().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set model eval mode
model.eval()

avg_psnr = 0.

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
with torch.no_grad():
    for iteration, (input, target) in progress_bar:
        # Set model gradients to zero
        lr = input.to(device)
        hr = target.to(device)

        sr = model(lr)

        mse = ((sr - hr) ** 2).data.mean()
        psnr_value = 10 * math.log10((hr.max() ** 2) / mse)
        avg_psnr += psnr_value

        progress_bar.set_description(f"[{iteration + 1}/{len(dataloader)}] "
                                     f"PSNR: {psnr_value:.2f}dB")

print(f"AVg PSNR: {avg_psnr / len(dataloader):.2f}dB.")
