#!/bin/bash

echo "Start downloading pre training model..."
wget https://github.com/Lornatang/SRCNN-PyTorch/releases/download/1.1/srcnn_2x.pth
wget https://github.com/Lornatang/SRCNN-PyTorch/releases/download/1.1/srcnn_3x.pth
wget https://github.com/Lornatang/SRCNN-PyTorch/releases/download/1.1/srcnn_4x.pth
echo "All pre training models have been downloaded!"
