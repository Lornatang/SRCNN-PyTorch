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

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

from srcnn_pytorch import SRCNN

parser = argparse.ArgumentParser(description="SRCNN algorithm is applied to video files.")
parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--weights", type=str, default="weights/srcnn_4x.pth",
                    help="Generator model name. (default:`weights/srcnn_4x.pth`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--scale-factor", default=4, type=int, choices=[2, 3, 4],
                    help="Super resolution upscale factor. (default:4)")
parser.add_argument("--view", default=False, type=bool,
                    help="Super resolution real time to show.")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = SRCNN().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Open video file
video_name = args.file
videoCapture = cv2.VideoCapture(video_name)

# Prepare to write the processed image into the video.
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale_factor),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * args.scale_factor)
videoWriter = cv2.VideoWriter(f"srcnn_{args.scale_factor}x_{video_name.split('/')[-1]}",
                              cv2.VideoWriter_fourcc(*"MPEG"), fps, size)

# read frame
success, frame = videoCapture.read()

while success:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('YCbCr')
    y, cb, cr = img.split()
    preprocess = transforms.ToTensor()
    inputs = preprocess(y).view(1, -1, y.size[1], y.size[0])

    inputs = inputs.to(device)

    prediction = model(inputs)
    prediction = prediction.cpu()
    out_image_y = prediction[0].detach().numpy()
    out_image_y *= 255.0
    out_image_y = out_image_y.clip(0, 255)
    out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")

    out_image_cb = cb.resize(out_image_y.size, Image.BICUBIC)
    out_image_cr = cr.resize(out_image_y.size, Image.BICUBIC)
    out_image = Image.merge("YCbCr", [out_image_y, out_image_cb, out_image_cr]).convert("RGB")
    # before converting the result in RGB
    out_image = cv2.cvtColor(np.asarray(out_image), cv2.COLOR_RGB2BGR)

    if args.view:
        # display video
        cv2.imshow("LR Video ", frame)
        cv2.imshow("SRCNN Video ", out_image)
        cv2.waitKey(1)
    else:
        # save video
        videoWriter.write(out_image)
    # next frame
    success, frame = videoCapture.read()
