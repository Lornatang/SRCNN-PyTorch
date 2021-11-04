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
from torch.backends import cudnn as cudnn

# ==============================================================================
# General configuration
# ==============================================================================
torch.manual_seed(0)
device = torch.device("cuda", 0)
cudnn.benchmark = True
upscale_factor = 2
mode = "train"
exp_name = "exp001"

# ==============================================================================
# Training configuration
# ==============================================================================
if mode == "train":
    # Dataset
    # Image format
    train_image_dir = f"data/T91/X{upscale_factor}/train"
    valid_image_dir = f"data/T91/X{upscale_factor}/valid"
    # LMDB format
    train_lr_lmdb_path = f"data/train_lmdb/LR/T91_X{upscale_factor}_lmdb"
    train_hr_lmdb_path = f"data/train_lmdb/HR/T91_X{upscale_factor}_lmdb"
    valid_lr_lmdb_path = f"data/valid_lmdb/LR/T91_X{upscale_factor}_lmdb"
    valid_hr_lmdb_path = f"data/valid_lmdb/HR/T91_X{upscale_factor}_lmdb"

    batch_size = 16

    # Incremental training and migration training
    resume = False
    strict = False
    start_epoch = 0
    resume_weight = ""

    # Total number of epochs
    epochs = 100

    # Model optimizer parameter
    model_optimizer_name = "sgd"
    model_lr = 1e-4
    model_momentum = 0.9
    model_weight_decay = 1e-7
    model_nesterov = False

    # Modify optimizer parameter (faster training and better PSNR)
    # model_optimizer_name = "adam"
    # model_lr = 1e-4
    # model_betas = (0.9, 0.999)

# ==============================================================================
# Verify configuration
# ==============================================================================
if mode == "valid":
    # Test data address
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/srcnn_best.pth"
