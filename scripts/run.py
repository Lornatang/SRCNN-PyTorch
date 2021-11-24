import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --inputs_dir ../data/T91/original --output_dir ../data/T91/SRCNN/")

# Split train and valid
os.system("python3 ./split_train_valid_dataset.py --inputs_dir ../data/T91/SRCNN")

# Create LMDB database file
os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_LRbicx2_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_LRbicx3_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/train --lmdb_path ../data/train_lmdb/SRCNN/T91_LRbicx4_lmdb --upscale_factor 4")

os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_LRbicx2_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_LRbicx3_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_dataset.py --image_dir ../data/T91/SRCNN/valid --lmdb_path ../data/valid_lmdb/SRCNN/T91_LRbicx4_lmdb --upscale_factor 4")

