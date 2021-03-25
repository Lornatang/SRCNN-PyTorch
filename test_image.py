# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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

import numpy as np
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

from srcnn_pytorch import SRCNN
from srcnn_pytorch import select_device

parser = argparse.ArgumentParser(description="Image Super-Resolution Using "
                                             "Deep Convolutional Networks.")
parser.add_argument("--file", type=str, default="./assets/baby.png",
                    help="Test low resolution image name. "
                         "(default:`./assets/baby.png`)")
parser.add_argument("--weights", type=str, default="./weights/SRCNN.pth",
                    help="Generator model name. (default:`./weights/SRCNN.pth`)")

args = parser.parse_args()
print(args)

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=1)

# create model
model = SRCNN().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Open image
image = Image.open(args.file).convert("YCbCr")
image = np.array(image).astype(np.float32)、

# RGB convert to YCbCr
y = 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
cb = 128. + (-37.945 * image[:, :, 0] - 74.494 * image[:, :, 1] + 112.439 * image[:, :, 2]) / 256.
cr = 128. + (112.439 * image[:, :, 0] - 94.154 * image[:, :, 1] - 18.285 * image[:, :, 2]) / 256.
ycbcr = np.array([y, cb, cr]).transpose([1, 2, 0])

inputs = ycbcr[..., 0]
inputs /= 255.
inputs = torch.from_numpy(inputs).to(device)
inputs = inputs.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out = model(inputs).clamp(0.0, 1.0)

out_image = out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

out_image = np.array([out_image, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])

# YCbCr convert to RGB
if len(out_image.shape) == 4:
    out_image = out_image.squeeze(0)
y = 16. + (64.738 * out_image[0, :, :] + 129.057 * out_image[1, :, :] + 25.064 * out_image[2, :, :]) / 256.
cb = 128. + (-37.945 * out_image[0, :, :] - 74.494 * out_image[1, :, :] + 112.439 * out_image[2, :, :]) / 256.
cr = 128. + (112.439 * out_image[0, :, :] - 94.154 * out_image[1, :, :] - 18.285 * out_image[2, :, :]) / 256.
out_image = torch.cat([y, cb, cr], 0).permute(1, 2, 0)

out_image = np.clip(out_image, 0.0, 255.0).astype(np.uint8)
out_image = Image.fromarray(out_image)
out_img.save(f"srcnn.png")
