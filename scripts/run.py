import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/T91/original --output_dir ../data/T91/SRCNN/train --image_size 33 --step 14 --num_workers 10")
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/T91/SRCNN/train --valid_images_dir ../data/T91/SRCNN/valid --valid_samples_ratio 0.1")
