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
import shutil

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode as IMode
from tqdm import tqdm


def main(args):
    lr_image_dir = f"{args.output_dir}/x{args.upscale_factor}/train/inputs"
    hr_image_dir = f"{args.output_dir}/x{args.upscale_factor}/train/target"

    if os.path.exists(lr_image_dir):
        shutil.rmtree(lr_image_dir)
    if os.path.exists(hr_image_dir):
        shutil.rmtree(hr_image_dir)

    os.makedirs(lr_image_dir)
    os.makedirs(hr_image_dir)

    image_file_names = os.listdir(args.inputs_dir)
    process_bar = tqdm(image_file_names, total=len(image_file_names))

    for image_file_name in process_bar:
        # Use PIL to read high-resolution image
        hr_image = Image.open(f"{args.inputs_dir}/{image_file_name}")

        # Get HR image size
        hr_image_width = (hr_image.width // args.upscale_factor) * args.upscale_factor
        hr_image_height = (hr_image.height // args.upscale_factor) * args.upscale_factor

        # Get HR image size
        lr_image_width = hr_image_width // args.upscale_factor
        lr_image_height = hr_image_height // args.upscale_factor

        # Process HR image
        hr_image = transforms.Resize([hr_image_height, hr_image_width], IMode.BICUBIC)(hr_image)

        # Process LR image
        lr_image = transforms.Resize([lr_image_height, lr_image_width], IMode.BICUBIC)(hr_image)
        lr_image = transforms.Resize([hr_image_height, hr_image_width], IMode.BICUBIC)(lr_image)

        for pos_x in range(0, hr_image.size[0] - args.image_size + 1, args.step):
            for pos_y in range(0, hr_image.size[1] - args.image_size + 1, args.step):
                lr_crop_image = lr_image.crop([pos_x, pos_y, pos_x + args.image_size, pos_y + args.image_size])
                hr_crop_image = hr_image.crop([pos_x + 6, pos_y + 6, pos_x + args.image_size - 6, pos_y + args.image_size - 6])

                # Save all images
                lr_crop_image.save(f"{lr_image_dir}/{image_file_name.split('.')[0]}_{pos_x}_{pos_y}.bmp")
                hr_crop_image.save(f"{hr_image_dir}/{image_file_name.split('.')[0]}_{pos_x}_{pos_y}.bmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts (Use SRCNN functions).")
    parser.add_argument("--inputs_dir", type=str, default="T91/original", help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str, default="T91", help="Path to generator image directory.")
    parser.add_argument("--image_size", type=int, default=32, help="Low-resolution image size from raw image.  (Default: 32)")
    parser.add_argument("--step", type=int, default=14, help="Crop image similar to sliding window.  (Default: 14)")
    parser.add_argument("--upscale_factor", type=int, default=2, help="Make several times upsampling datasets.  (Default: 2)")
    args = parser.parse_args()

    main(args)
