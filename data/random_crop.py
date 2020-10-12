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
"""The method of randomly cutting out the number of pictures of specified size 
from the original image"""
import glob
import os

import cv2
import numpy as np


def crop_bbox(filename: str, size: int, numbers: int,
              delete: bool = True) -> None:
    r"""

    Args:
        filename (str): Image path to crop.
        size (int): Size of each clip.
        numbers (int): How many small img need to be cut out from each img?.
        delete (bool, optional): Do you want to delete the clipped image
            after clipping. (Default: ``True``)
    """
    img = cv2.imread(filename)
    # get shape
    h, w, _ = img.shape

    # each crop
    for i in range(numbers):
        # get left top w of crop bounding box
        w1 = np.random.randint(w - size)
        # get left top h of crop bounding box
        h1 = np.random.randint(h - size)
        # get right bottom w of crop bounding box
        w2 = w1 + size
        # get right bottom h of crop bounding box
        h2 = h1 + size

        # crop bounding box
        cv2.imwrite(f"{filename.split('/')[-1].split('.')[0]}_{i}.png",
                    img[w1:w2, h1:h2])

    if delete:
        os.remove(filename)


if __name__ == "__main__":
    for filename in glob.glob("./*.png"):
        crop_bbox(filename, 33, 273)
