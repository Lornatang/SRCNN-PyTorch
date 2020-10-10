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
from tensorboardX import SummaryWriter

from srcnn_pytorch import DatasetFromFolder
from srcnn_pytorch import SRCNN
from srcnn_pytorch import progress_bar

parser = argparse.ArgumentParser(description="Image Super-Resolution Using Deep Convolutional Networks.")
parser.add_argument("--dataroot", type=str, default="./data/91-images",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--iters", default=1e8, type=int, metavar="N",
                    help="Number of total epochs to run. According to the 1e8 iterations in the original paper."
                         "(default:1e8)")
parser.add_argument("--src-size", type=int, default=33,
                    help="Size of the data image (squared assumed). (default:33)")
parser.add_argument("--dst-size", type=int, default=21,
                    help="Size of the data image (squared assumed). (default:21)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("-b", "--batch-size", default=16, type=int,
                    metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (default:0.0001)")
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

train_dataset = DatasetFromFolder(data_dir=f"{args.dataroot}/train/data",
                                  target_dir=f"{args.dataroot}/train/target",
                                  src_size=args.src_size,
                                  dst_size=args.dst_size,
                                  upscale_factor=args.upscale_factor)
val_dataset = DatasetFromFolder(data_dir=f"{args.dataroot}/val/data",
                                target_dir=f"{args.dataroot}/val/target",
                                src_size=args.src_size,
                                dst_size=args.dst_size,
                                upscale_factor=args.upscale_factor)

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

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device))

criterion = nn.MSELoss().to(device)
# we use Adam instead of SGD like in the paper, because it's faster
optimizer = optim.Adam([
    {"params": model.features.parameters()},
    {"params": model.map.parameters()},
    {"params": model.reconstruction.parameters(), "lr": args.lr * 0.1}
], lr=args.lr)

best_psnr = 0.
epochs = int(args.iters // len(train_dataloader))

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler()
# Start write training log
writer = SummaryWriter("logs")
print("Run `tensorboard --logdir=./logs` view training log.")

for epoch in range(epochs):
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
        writer.add_scalar("Train_loss", loss.item(), iteration + iteration * epoch)
        progress_bar(epoch, epochs, iteration, len(train_dataloader), f"Loss: {loss.item():.6f}")

    print(f"Training average loss: {train_loss / len(train_dataloader):.6f}")

    # Test
    model.eval()
    avg_psnr = 0.
    with torch.no_grad():
        for iteration, (inputs, target) in enumerate(val_dataloader):
            inputs, target = inputs.to(device), target.to(device)

            prediction = model(inputs)
            mse = criterion(prediction, target)
            psnr = 10 * math.log10(1. / mse.item())
            avg_psnr += psnr

            writer.add_scalar("Test_PSNR", psnr, iteration + iteration * epoch)
    print(f"Average PSNR: {avg_psnr / len(val_dataloader):.2f} dB.")

    # Save model
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f"weights/srcnn_{args.upscale_factor}x.pth")
