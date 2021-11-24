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
    if os.path.exists(args.lmdb_path):
        shutil.rmtree(args.lmdb_path)

    os.makedirs(args.lmdb_path)

    image_file_names = os.listdir(args.image_dir)
    total_image_number = len(image_file_names)

    # Determine the LMDB database file size according to the image size
    image = cv2.imread(os.path.abspath(f"{args.image_dir}/{image_file_names[0]}"))
    _, image_byte = cv2.imencode(f".{image_file_names[0].split('.')[-1]}", image)
    lmdb_map_size = image_byte.nbytes * total_image_number * 3.0

    # Open LMDB write environment
    lmdb_env = lmdb.open(args.lmdb_path, map_size=int(lmdb_map_size))

    # Start over to write the file
    content = lmdb_env.begin(write=True)

    # Easy to read and visualize with DataLoader
    total_sub_image_number = 1
    process_bar = tqdm(image_file_names, total=total_image_number)

    for file_name in process_bar:
        # Use OpenCV to read low-resolution and high-resolution images
        image = cv2.imread(f"{args.image_dir}/{file_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process HR to LR image
        if args.upscale_factor > 1:
            image = cv2.resize(image, [image.shape[0] // args.upscale_factor, image.shape[1] // args.upscale_factor], interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(image, [image.shape[0] * args.upscale_factor, image.shape[1] * args.upscale_factor], interpolation=cv2.INTER_CUBIC)

        # Label from int to ascii
        image_key_bytes = str(total_sub_image_number).encode("ascii")

        # LR bytes and HR bytes
        _, image_encode = cv2.imencode(f".{file_name.split('.')[-1]}", image)
        image_bytes = image_encode.tobytes()

        process_bar.set_description(f"Write {total_sub_image_number} images to lmdb dataset.")
        total_sub_image_number += 1

        content.put(image_key_bytes, image_bytes)

    # Submit image data to LMDB database
    content.commit()
    # Close all events
    process_bar.close()
    lmdb_env.close()
    print("Writing lmdb database successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LMDB database scripts.")
    parser.add_argument("--image_dir", type=str, default="T91/SRCNN/train", help="Path to image directory. (Default: ``T91/SRCNN/train``)")
    parser.add_argument("--lmdb_path", type=str, default="train_lmdb/SRCNN/T91_HR_lmdb", help="Path to lmdb database. (Default: ``train_lmdb/SRCNN/T91_HR_lmdb``)")
    parser.add_argument("--upscale_factor", type=int, default=1, help="Image zoom factor. (Default: 1)")
    args = parser.parse_args()

    main(args)
