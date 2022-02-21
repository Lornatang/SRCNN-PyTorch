import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/T91/original --output_dir ../data/T91/SRCNN/train --image_size 33 --step 14 --num_workers 10")
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/T91/SRCNN/train --valid_images_dir ../data/T91/SRCNN/valid --valid_samples_ratio 0.1")

# Create LMDB database file
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_LRbicx3_lmdb --upscale_factor 3")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_LRbicx4_lmdb --upscale_factor 4")

os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_LRbicx3_lmdb --upscale_factor 3")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_LRbicx4_lmdb --upscale_factor 4")
