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
import random
import shutil
import os

upscale_factor = 4
valid_samples_ratio = 0.1

train_lr_image_dir = f"T91/X{upscale_factor}/train/inputs"
train_hr_image_dir = f"T91/X{upscale_factor}/train/target"
valid_lr_image_dir = f"T91/X{upscale_factor}/valid/inputs"
valid_hr_image_dir = f"T91/X{upscale_factor}/valid/target"

if not os.path.exists(valid_lr_image_dir):
    os.makedirs(valid_lr_image_dir)
if not os.path.exists(valid_hr_image_dir):
    os.makedirs(valid_hr_image_dir)

train_files = os.listdir(train_lr_image_dir)
valid_files = random.sample(train_files, int(len(train_files) * valid_samples_ratio))

for file_name in valid_files:
    train_lr_image_path = f"{train_lr_image_dir}/{file_name}"
    train_hr_image_path = f"{train_hr_image_dir}/{file_name}"
    valid_lr_image_path = f"{valid_lr_image_dir}/{file_name}"
    valid_hr_image_path = f"{valid_hr_image_dir}/{file_name}"

    shutil.copyfile(train_lr_image_path, valid_lr_image_path)
    shutil.copyfile(train_hr_image_path, valid_hr_image_path)
