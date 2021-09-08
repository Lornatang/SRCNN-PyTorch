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

# ================================================ ===========================
# File description: Realize the model training function.
# ================================================ ===========================
from torch.utils.data import DataLoader

from config import *
from dataset import BaseDataset


def train(train_dataloader, epoch) -> None:
    """Train the model.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The loader of the training data set.
        epoch (int): number of training cycles.

    """
    # Calculate how many iterations there are under Epoch.
    batches = len(train_dataloader)
    # Put the model in training mode.
    model.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # Copy the data to the specified device.
        lr = lr.to(device)
        hr = hr.to(device)
        # Initialize the model gradient.
        model.zero_grad()
        # Generate super-resolution images.
        sr = model(lr)
        # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        pixel_loss = criterion(sr, hr)
        # Update model weights.
        pixel_loss.backward()
        optimizer.step()
        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train/Loss", pixel_loss.item(), iters)
        # Print the loss function every ten iterations and the last iteration in this Epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Train Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {pixel_loss.item():.6f}.")


def validate(valid_dataloader, epoch) -> float:
    """Verify the model.

    Args:
        valid_dataloader (torch.utils.data.DataLoader): loader for validating data set.
        epoch (int): number of training cycles.

    Returns:
        PSNR value(float).

    """
    # Calculate how many iterations there are under Epoch.
    batches = len(valid_dataloader)
    # Put the model in verification mode.
    model.eval()
    # Initialize the evaluation index.
    total_psnr_value = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            # Copy the data to the specified device.
            lr = lr.to(device)
            hr = hr.to(device)
            # Generate super-resolution images.
            sr = model(lr)
            # Calculate the PSNR indicator.
            mse_loss = ((sr - hr) ** 2).data.mean()
            psnr_value = 10 * torch.log10(1 / mse_loss).item()
            total_psnr_value += psnr_value

        avg_psnr_value = total_psnr_value / batches
        # Write the value of each round of verification indicators into Tensorboard.
        writer.add_scalar("Valid/PSNR", avg_psnr_value, epoch + 1)
        # Print evaluation indicators.
        print(f"Valid Epoch[{epoch + 1:05d}] avg PSNR: {avg_psnr_value:.2f}.\n")

    return avg_psnr_value


def main() -> None:
    # Create a super-resolution experiment result folder.
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)

    # Load the data set.
    train_dataset = BaseDataset(train_dir)
    valid_dataset = BaseDataset(valid_dir)
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    # Check whether the training progress of the last abnormal end is restored, for example, 
    # the power is cut off in the middle of the training.
    if resume:
        print("Resuming...")
        model.load_state_dict(torch.load(resume_weight))

    # Initialize the evaluation index of the training phase
    best_psnr_value = 0.0
    for epoch in range(start_epoch, epochs):
        # Train the super-score model under Epoch every time.
        train(train_dataloader, epoch)
        # Verify the super-score model under Epoch every time.
        psnr_value = validate(valid_dataloader, epoch)
        # Determine whether the performance of the super-score model under Epoch is the best.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the weights of the super-score model under Epoch. If the performance of the super-score model 
        # under Epoch is best, another model file named `best.pth` will be saved in the `results` directory.
        torch.save(model.state_dict(), os.path.join(exp_dir1, f"epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(exp_dir2, "best.pth"))

    # Save the weight of the last sr model under Epoch.
    torch.save(model.state_dict(), os.path.join(exp_dir2, "last.pth"))


if __name__ == "__main__":
    main()
