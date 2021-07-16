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
import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import models
from dataset import CustomDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="./data/T91/LRbicx4",
                    help="Path to datasets. (Default: `./data/T91/LRbicx4`)")
parser.add_argument("--epochs", default=4096, type=int, metavar="N",
                    help="Number of total epochs to run. "
                         "According to the 1e8 iters in the original paper."
                         "(Default: 4096)")
parser.add_argument("--batch-size", type=int, default=128, 
                    help="mini-batch size (Default: 128)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate. (Default: 0.0001)")
parser.add_argument("--arch", metavar="ARCH", default="srcnn_x4",
                    choices=model_names,
                    help="model architecture: " +
                        "srcnn_x2 | srcnn_x3 | srcnn_x4"
                        " (Default: `srcnn_x4`)")
parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--model-path", type=str, default="",
                    help="Path to weights.")
parser.add_argument("--seed", type=int, default=666,
                    help="Seed for initializing training. (Default: 666)")
args = parser.parse_args()

# Set random initialization seed, easy to reproduce.
if args.seed is None:
    args.seed = random.randint(1, 10000)
print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True

# Set the operating device model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset(args.dataroot)
dataloader = data.DataLoader(dataset, args.batch_size, True, pin_memory=True)

# Construct SRCNN model.
if args.pretrained:
    print(f"=> Using pre-trained model '{args.arch}'")
    model = models.__dict__[args.arch](pretrained=True)
else:
    print(f"=> Creating model '{args.arch}'")
    model = models.__dict__[args.arch]()

# Load the last training weights.
# How many iterations to continue training from. The default starts from 0.
start_epoch = 0
if args.model_path != "":
    model.load_state_dict(torch.load(args.model_path, torch.device("cpu")))
    start_epoch = "".join(list(filter(str.isdigit, args.model_path)))
    print(f"You loaded {args.model_path} for model. "
          f"Resume epoch from {start_epoch}.")

# Define the loss function.
criterion = nn.MSELoss().to(device)
# Define the optimizer.
optimizer = optim.SGD(params=[
    {"params": model.features.parameters()}, 
    {"params": model.map.parameters()}, 
    {"params": model.reconstruction.parameters(),
     "lr": args.lr * 0.1}], lr=args.lr)
# Define a mixed precision trainer.
scaler = amp.GradScaler()

# Use Tensorboard to record the Loss curve during training.
writer = SummaryWriter("sample/logs")


def main():
    num_batches = len(dataloader)
    for epoch in range(int(start_epoch), args.epochs):
        for index, data in enumerate(dataloader, 1):
            # Copy the data to the designated device.
            inputs, target = data[0].to(device), data[1].to(device)

            ##############################################
            # Turn on mixed precision training.
            ##############################################
            optimizer.zero_grad()

            with amp.autocast():
                output = model(inputs)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            print(f"Epoch[{epoch}/{args.epochs}]"
                  f"({index}/{num_batches}) Loss: {loss.item():.4f}.")

            # Write the loss value during training into Tensorboard.
            batches = index + epoch * num_batches + 1
            writer.add_scalar("Train/Loss", loss.item(), batches)
        torch.save(model.state_dict(), os.path.join("sample", f"{args.arch}_epoch{epoch}.pth"))

    # Save the model weights of the last iteration.
    torch.save(model.state_dict(), os.path.join("result", f"{args.arch}-last.pth"))


if __name__ == "__main__":
    main()
