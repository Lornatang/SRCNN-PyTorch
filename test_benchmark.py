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

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed

from srcnn_pytorch import DatasetFromFolder
from srcnn_pytorch import SRCNN
from srcnn_pytorch import progress_bar

parser = argparse.ArgumentParser(description="PyTorch Super Resolution CNN.")
parser.add_argument("--dataroot", type=str, default="./data/DIV2K",
                    help="Path to datasets. (default:`./data/DIV2K`)")
parser.add_argument("--image-size", type=int, default=32,
                    help="Size of the data crop (squared assumed). (default:32)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--scale-factor", type=int, required=True, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor.")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--weights", type=str, required=True,
                    help="Path to weights.")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = DatasetFromFolder(f"{args.dataroot}/val",
                            image_size=args.image_size,
                            scale_factor=args.scale_factor)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

model = SRCNN().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
criterion = nn.MSELoss().to(device)

# Test
model.eval()
avg_psnr = 0.
with torch.no_grad():
    for iteration, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.to(device), target.to(device)

        prediction = model(inputs)
        mse = criterion(prediction, target)
        psnr = 10 * math.log10(1 / mse.item())
        avg_psnr += psnr
        progress_bar(0, 1, iteration, len(dataloader), f"PSNR: {psnr:.2f} dB")

print(f"Average PSNR: {avg_psnr / len(dataloader):.2f} dB.")
