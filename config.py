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
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn

# ==============================================================================
# General configuration
# ==============================================================================
torch.manual_seed(0)
device = torch.device("cuda", 0)
cudnn.benchmark = True
upscale_factor = 2
mode = "train"
exp_name = "train_srcnn_x2"

# ==============================================================================
# Training configuration
# ==============================================================================
if mode == "train":
    # Dataset
    train_image_dir = f"data/T91/SRCNN/train"
    valid_image_dir = f"data/T91/SRCNN/valid"

    image_size = 33
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = False
    strict = True
    start_epoch = 0
    resume_weight = ""

    # Total number of epochs. SGD: 72500. Adam: 4500
    epochs = 72500

    # SGD optimizer parameter (less training and low PSNR)
    model_optimizer_name = "sgd"
    model_lr = 1e-4
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # Adam optimizer parameter (faster training and better PSNR)
    # model_optimizer_name = "adam"
    # model_lr = 1e-4
    # model_betas = (0.9, 0.999)

    print_frequency = 1000

# ==============================================================================
# Verify configuration
# ==============================================================================
if mode == "valid":
    # Test data address
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/last.pth"
