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
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda import amp

from srcnn_pytorch import DatasetFromFolder
from srcnn_pytorch import SRCNN
from srcnn_pytorch import progress_bar

parser = argparse.ArgumentParser(description="PyTorch Super Resolution CNN.")
parser.add_argument("--dataroot", type=str, default="./data/DIV2K",
                    help="Path to datasets. (default:`./data/DIV2K`)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="Number of total epochs to run. (default:200)")
parser.add_argument("--image-size", type=int, default=256,
                    help="Size of the data crop (squared assumed). (default:256)")
parser.add_argument("-b", "--batch-size", default=16, type=int,
                    metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate. (default:0.01)")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("-p", "--print-freq", default=5, type=int,
                    metavar="N", help="Print frequency. (default:5)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--weights", default="",
                    help="Path to weights (to continue training).")
parser.add_argument("--manualSeed", type=int, default=0,
                    help="Seed for initializing training. (default:0)")

args = parser.parse_args()
print(args)

try:
    os.makedirs("weights")
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = DatasetFromFolder(f"{args.dataroot}/train",
                                  image_size=args.image_size,
                                  scale_factor=args.scale_factor)
val_dataset = DatasetFromFolder(f"{args.dataroot}/val",
                                image_size=args.image_size,
                                scale_factor=args.scale_factor)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=int(args.workers))
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

model = SRCNN().to(device)
model.weight_init()

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device))

criterion = nn.MSELoss().to(device)
# we use Adam instead of SGD like in the paper, because it's faster
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

best_psnr = 0.

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler()

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.
    for iteration, (inputs, target) in enumerate(train_dataloader):
        optimizer.zero_grad()

        inputs, target = inputs.to(device), target.to(device)

        # Runs the forward pass with autocasting.
        with amp.autocast():
            output = model(inputs)
            loss = criterion(output, target)

        # Scales loss.  Calls backward() on scaled loss to
        # create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose
        # for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of
        # the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs,
        # optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        train_loss += loss.item()

        progress_bar(epoch, args.epochs, iteration, len(train_dataloader), f"Loss: {loss.item():.6f}")

    print(f"Training average loss: {train_loss / len(train_dataloader):.6f}")

    # Test
    model.eval()
    avg_psnr = 0.
    with torch.no_grad():
        for iteration, (inputs, target) in enumerate(val_dataloader):
            inputs, target = inputs.to(device), target.to(device)

            prediction = model(inputs)
            mse = criterion(prediction, target)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
    print(f"Average PSNR: {avg_psnr / len(val_dataloader):.2f} dB.")

    # Dynamic adjustment of learning rate.
    scheduler.step()

    # Save model
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"weights/srcnn_{args.scale_factor}x_epoch_{epoch + 1}.pth")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f"weights/srcnn_{args.scale_factor}x.pth")
