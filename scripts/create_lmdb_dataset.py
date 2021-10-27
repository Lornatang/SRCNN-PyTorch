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

import cv2
import lmdb
from tqdm import tqdm


def main(args):
    if os.path.exists(args.lr_lmdb_path):
        shutil.rmtree(args.lr_lmdb_path)
    if os.path.exists(args.hr_lmdb_path):
        shutil.rmtree(args.hr_lmdb_path)

    os.makedirs(args.lr_lmdb_path)
    os.makedirs(args.hr_lmdb_path)

    image_file_names = os.listdir(args.lr_image_dir)
    total_image_number = len(image_file_names)

    # Determine the LMDB database file size according to the image size
    lr_image = cv2.imread(os.path.abspath(f"{args.lr_image_dir}/{image_file_names[0]}"))
    hr_image = cv2.imread(os.path.abspath(f"{args.lr_image_dir}/{image_file_names[0]}"))
    lr_image_lmdb_map_size = lr_image.shape[0] * lr_image.shape[1] * lr_image.shape[2] * total_image_number * 1.5
    hr_image_lmdb_map_size = hr_image.shape[0] * hr_image.shape[1] * hr_image.shape[2] * total_image_number * 1.5

    # Open LMDB write environment
    lr_lmdb_env = lmdb.open(args.lr_lmdb_path, map_size=int(lr_image_lmdb_map_size))
    hr_lmdb_env = lmdb.open(args.hr_lmdb_path, map_size=int(hr_image_lmdb_map_size))

    # Easy to read and visualize with DataLoader
    total_sub_image_number = 1
    process_bar = tqdm(image_file_names, total=total_image_number)

    # Start over to write the file
    lr_content = lr_lmdb_env.begin(write=True)
    hr_content = hr_lmdb_env.begin(write=True)

    for image_file_name in process_bar:
        # Use OpenCV to read low-resolution and high-resolution images
        lr_image = cv2.imread(f"{args.lr_image_dir}/{image_file_name}")
        hr_image = cv2.imread(f"{args.hr_image_dir}/{image_file_name}")
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Label from int to ascii
        image_key_bytes = str(total_sub_image_number).encode("ascii")

        # LR bytes and HR bytes
        _, lr_image_encode = cv2.imencode(".bmp", lr_image)
        _, hr_image_encode = cv2.imencode(".bmp", hr_image)
        lr_image_bytes = lr_image_encode.tobytes()
        hr_image_bytes = hr_image_encode.tobytes()

        process_bar.set_description(f"Write {total_sub_image_number} images to lmdb dataset.")
        total_sub_image_number += 1

        lr_content.put(image_key_bytes, lr_image_bytes)
        hr_content.put(image_key_bytes, hr_image_bytes)

    # Submit image data to LMDB database
    lr_content.commit()
    hr_content.commit()
    # Close all events
    process_bar.close()
    lr_lmdb_env.close()
    hr_lmdb_env.close()
    print("Writing lmdb database successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LMDB database scripts.")
    parser.add_argument("--lr_image_dir", type=str, required=True, help="Path to low-resolution image directory.")
    parser.add_argument("--hr_image_dir", type=str, required=True, help="Path to high-resolution image directory.")
    parser.add_argument("--lr_lmdb_path", type=str, required=True, help="Path to low-resolution lmdb database.")
    parser.add_argument("--hr_lmdb_path", type=str, required=True, help="Path to high-resolution lmdb database.")
    args = parser.parse_args()

    main(args)
