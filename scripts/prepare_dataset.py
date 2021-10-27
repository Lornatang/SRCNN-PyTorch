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
    lr_image_dir = f"{args.output_image_dir}/X{args.upscale_factor}/inputs"
    hr_image_dir = f"{args.output_image_dir}/X{args.upscale_factor}/target"

    if os.path.exists(lr_image_dir):
        shutil.rmtree(lr_image_dir)
    os.makedirs(lr_image_dir)
    if os.path.exists(hr_image_dir):
        shutil.rmtree(hr_image_dir)
    os.makedirs(hr_image_dir)

    image_file_names = os.listdir(args.inputs_image_dir)
    process_bar = tqdm(image_file_names, total=len(image_file_names))

    for image_file_name in process_bar:
        # Use PIL to read high-resolution image
        hr_image = Image.open(f"{args.inputs_image_dir}/{image_file_name}")

        # Get all image size
        hr_image_width = (hr_image.width // args.upscale_factor) * args.upscale_factor
        hr_image_height = (hr_image.height // args.upscale_factor) * args.upscale_factor

        # Process HR  image
        hr_image = transforms.Resize([hr_image_height, hr_image_width], IMode.BICUBIC)(hr_image)

        for pos_x in range(0, hr_image.size[0] - args.lr_image_size + 1, args.step):
            for pos_y in range(0, hr_image.size[1] - args.lr_image_size + 1, args.step):
                hr_crop_image = hr_image.crop([pos_x, pos_y, pos_x + args.lr_image_size, pos_y + args.lr_image_size])

                # Scale and intercept the image according to the method in the SRCNN paper
                # HR: 32 -> 20
                hr_sub_image = transforms.CenterCrop([args.hr_image_size, args.hr_image_size])(hr_crop_image)

                lr_crop_image_height = args.lr_image_size // args.upscale_factor
                lr_crop_image_width = args.lr_image_size // args.upscale_factor
                # LR: 32 -> 8
                lr_crop_image = transforms.Resize([lr_crop_image_height, lr_crop_image_width], IMode.BICUBIC)(hr_crop_image)
                # LR: 8 -> 32
                lr_sub_image = transforms.Resize([args.lr_image_size, args.lr_image_size], IMode.BICUBIC)(lr_crop_image)

                # Save all images
                lr_sub_image.save(f"{lr_image_dir}/{image_file_name.split('.')[0]}_{pos_x}_{pos_y}.bmp")
                hr_sub_image.save(f"{hr_image_dir}/{image_file_name.split('.')[0]}_{pos_x}_{pos_y}.bmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts (Use SRCNN functions).")
    parser.add_argument("--inputs_image_dir", type=str, required=True, help="Path to input image directory.")
    parser.add_argument("--output_image_dir", type=str, required=True, help="Path to generator image directory.")
    parser.add_argument("--lr_image_size", type=int, default=32, help="Low-resolution image size from High-resolution image.  (Default: 32)")
    parser.add_argument("--hr_image_size", type=int, default=20, help="High-resolution image size from raw image.  (Default: 20)")
    parser.add_argument("--step", type=int, default=14, help="Crop image similar to sliding window.  (Default: 14)")
    parser.add_argument("--upscale_factor", type=int, default=4, help="Make several times upsampling datasets.  (Default: 4)")
    args = parser.parse_args()

    main(args)
