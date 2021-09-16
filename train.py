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
from torch.cuda import amp
from torch.utils.data import DataLoader

from config import *
from dataset import BaseDataset


def train(train_dataloader, epoch, scaler) -> None:
    """Training a super-resolution model based on the MSE loss function.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
        epoch (int): number of training cycles.
        scaler (amp.GradScaler): Gradient scaler.
    """
    # The number of training steps per cycle.
    batches = len(train_dataloader)
    # Set the module to training mode.
    model.train()
    for index, (lr, hr) in enumerate(train_dataloader):
        lr = lr.to(device)
        hr = hr.to(device)

        # Initialize the model gradient.
        model.zero_grad()
        # Mixed precision training.
        with amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Write the loss during training to Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train/MSE_Loss", loss.item(), iters)
        # Print loss function.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {loss.item():.6f}.")


def validate(valid_dataloader, epoch) -> float:
    """Test the performance of the super-resolution model.

    Args:
        valid_dataloader (torch.utils.data.DataLoader): loader for validating data set.
        epoch (int): number of training cycles.

    Returns:
        PSNR value(float).
    """
    # The number of verification steps per cycle.
    batches = len(valid_dataloader)
    # Set the module to verification mode.
    model.eval()
    # Initialize the evaluation index.
    total_psnr_value = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            # Calculate the PSNR value.
            sr = model(lr)
            psnr_value = 10 * torch.log10(1 / criterion(sr, hr)).item()
            total_psnr_value += psnr_value

        avg_psnr_value = total_psnr_value / batches
        # Write the PSNR indicator value into Tensorboard.
        writer.add_scalar("Valid/MSE_metrics", avg_psnr_value, epoch + 1)
        # Print evaluation indicators.
        print(f"Valid Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.2f}.\n")

    return avg_psnr_value


def main() -> None:
    # Create a super-resolution experiment result folder.
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)

    # Load the dataset.
    train_dataset = BaseDataset(train_dir)
    valid_dataset = BaseDataset(valid_dir)
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)

    # Check whether the training progress of the last abnormal end is restored, for example, the power is cut off in the middle of the training.
    if resume:
        print("Resuming...")
        model.load_state_dict(torch.load(resume_weight, map_location=device))

    # Create GradScalar function.
    scaler = amp.GradScaler()

    # Initialize the evaluation index.
    best_mse_metrics = 0.0

    # Train the super-resolution model based on the MSE loss function.
    for epoch in range(start_epoch, epochs):
        train(train_dataloader, epoch, scaler)
        psnr_value = validate(valid_dataloader, epoch)
        # Automatically search and save the optimal model weight.
        is_best = psnr_value > best_mse_metrics
        best_mse_metrics = max(psnr_value, best_mse_metrics)
        torch.save(model.state_dict(), os.path.join(exp_dir1, f"epoch_{epoch + 1}.pth"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(exp_dir2, "best.pth"))

    # Save the model weight of the last cycle in the process of training the super-resolution model.
    torch.save(model.state_dict(), os.path.join(exp_dir2, "last.pth"))


if __name__ == "__main__":
    main()
