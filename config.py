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

# ==============================================================================
# File description: Realize the parameter configuration function of data set, model, training and verification code.
# ==============================================================================
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import SRCNN

# =============================================================================
# General configuration
# =============================================================================
torch.manual_seed(0)             # Set random seed.
upscale_factor = 4               # How many times the size of the high-resolution image in the data set is than the low-resolution image.
device = torch.device("cuda:0")  # The first GPU is used for processing by default.
cudnn.benchmark = True           # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
mode = "train"                   # Run mode. Specific mode loads specific variables.
exp_name = "exp000"              # Experiment name.

# =============================================================================
# Training configuration
# =============================================================================
if mode == "train":
    # Dataset.
    train_dir     = f"data/T91/X{upscale_factor}/train"  # The address of the training data set.
    valid_dir     = f"data/T91/X{upscale_factor}/valid"  # Verify the address of the data set.
    lr_image_size = 32                                   # Low-resolution image size in the training data set.
    hr_image_size = 20                                   # High resolution image size in the training data set.
    batch_size    = 128                                  # Training data batch size.

    # Model.
    model         = SRCNN(mode).to(device)               # Load the generative model.

    # Interrupt training.
    start_epoch   = 0                                    # The initial number of iterations during network training. When set to 0, it means incremental training.
    resume        = False                                # Set to `True` to continue training from the previous training progress.
    resume_weight = ""                                   # Resume the model weight during training.

    # Total number of iterations.
    epochs        = 30000                                # The total number of network training cycles.

    # Loss function.
    criterion = nn.MSELoss().to(device)                  # Pixel loss.

    # Optimizer.
    optimizer = optim.SGD(params=[{"params": model.features.parameters(), "lr": 0.0001},
                                  {"params": model.map.parameters(), "lr": 0.0001},
                                  {"params": model.reconstruction.parameters(), "lr": 0.00001}],
                          lr=0.0001)                     # Learning rate during network training.

    # Training log.
    writer = SummaryWriter(os.path.join("samples", "logs", exp_name))

    # Additional variables.
    exp_dir1 = os.path.join("samples", exp_name)
    exp_dir2 = os.path.join("results", exp_name)

# =============================================================================
# Verify configuration
# =============================================================================
if mode == "valid":
    exp_dir = os.path.join("results", "test", exp_name)  # Additional variables.
    model = SRCNN(mode).to(device)                       # Load the super-resolution model.
    model_path = f"results/{exp_name}/best.pth"          # Model weight address.
    sr_dir = f"results/test/{exp_name}"                  # Super resolution image address.
    hr_dir = f"C:/dataset/Set5/GTmod12"                  # High resolution image address.





