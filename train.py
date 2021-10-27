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
# ============================================================================
"""File description: Realize the model training function."""
import os

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import LMDBDataset
from model import SRCNN


def load_dataset() -> [DataLoader, DataLoader]:
    train_datasets = LMDBDataset(config.train_lr_lmdb_path, config.train_hr_lmdb_path)
    valid_datasets = LMDBDataset(config.valid_lr_lmdb_path, config.valid_hr_lmdb_path)
    train_dataloader = DataLoader(train_datasets, config.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_datasets, config.batch_size, shuffle=False, pin_memory=True)

    return train_dataloader, valid_dataloader


def build_model() -> nn.Module:
    model = SRCNN(mode="train").to(config.device, non_blocking=True)

    return model


def define_loss() -> nn.MSELoss:
    criterion = nn.MSELoss().to(config.device, non_blocking=True)

    return criterion


def define_optimizer(model) -> optim.SGD:
    if config.model_optimizer_name == "sgd":
        optimizer = optim.SGD([{"params": model.features.parameters()},
                               {"params": model.map.parameters()},
                               {"params": model.reconstruction.parameters(), "lr": config.model_lr * 0.1}],
                              lr=config.model_lr)
    else:
        optimizer = optim.SGD([{"params": model.features.parameters()},
                               {"params": model.map.parameters()},
                               {"params": model.reconstruction.parameters(), "lr": config.model_lr * 0.1}],
                              lr=config.model_lr)

    return optimizer


def resume_checkpoint(model):
    if config.resume:
        if config.resume_weight != "":
            model.load_state_dict(torch.load(config.resume_weight), strict=config.strict)


def train(model, train_dataloader, criterion, optimizer, epoch, scaler, writer) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_dataloader)
    # Put the generator in training mode
    model.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)

        # Initialize the generator gradient
        model.zero_grad()

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # In this Epoch, every one hundred iterations and the last iteration print the loss function
        # and write it to Tensorboard at the same time
        if (index + 1) % 100 == 0 or (index + 1) == batches:
            iters = index + epoch * batches + 1
            writer.add_scalar("Train/MSE_Loss", loss.item(), iters)
            print(f"Epoch[{epoch + 1:05d}/{config.epochs:05d}]({index + 1:05d}/{batches:05d}) MSE loss: {loss.item():.6f} .")


def validate(model, valid_dataloader, criterion, epoch, writer) -> float:
    # Calculate how many iterations there are under Epoch.
    batches = len(valid_dataloader)
    # Put the generator in verification mode.
    model.eval()
    # Initialize the evaluation index.
    total_psnr_metric = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.to(config.device, non_blocking=True)
            hr = hr.to(config.device, non_blocking=True)
            # Calculate the PSNR evaluation index.
            sr = model(lr)
            psnr = 10 * torch.log10(1 / criterion(sr, hr)).item()
            total_psnr_metric += psnr

        avg_psnr = total_psnr_metric / batches
        # Write the value of each round of verification indicators into Tensorboard.
        writer.add_scalar("Valid/PSNR", avg_psnr, epoch + 1)
        # Print evaluation indicators.
        print(f"Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr:.2f}.\n")

    return avg_psnr


def main() -> None:
    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    print("Load train dataset and valid dataset...")
    train_dataloader, valid_dataloader = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    print("Build SR model...")
    model = build_model()
    print("Build SR model successfully.")

    print("Define all loss functions...")
    criterion = define_loss()
    print("Define all loss functions successfully.")

    print("Define all optimizer functions...")
    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    print("Check whether the training weight is restored...")
    resume_checkpoint(model)
    print("Check whether the training weight is restored successfully.")

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    print("Start train SRCNN model.")
    for epoch in range(config.start_epoch, config.epochs):
        train(model, train_dataloader, criterion, optimizer, epoch, scaler, writer)

        psnr = validate(model, valid_dataloader, criterion, epoch, writer)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save(model.state_dict(), os.path.join(samples_dir, f"srcnn_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(results_dir, "srcnn_best.pth"))

    # Save the generator weight under the last Epoch in this stage
    torch.save(model.state_dict(), os.path.join(results_dir, "srcnn_last.pth"))
    print("End train SRCNN model.")


if __name__ == "__main__":
    main()
