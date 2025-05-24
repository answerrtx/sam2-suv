#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo script for performing OmniGlue inference."""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 构造出 omniglue/src 路径
"""omniglue_src = os.path.join(current_dir, 'src', 'omniglue')
if omniglue_src not in sys.path:
    sys.path.insert(0, omniglue_src)"""
#print(sys.path)
from omniglue_onnx import omniglue
from omniglue_onnx.omniglue import utils
from PIL import Image
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def match(image0_fp, image1_fp):

  for im_fp in [image0_fp, image1_fp]:
    if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
      raise ValueError(f"Image filepath '{im_fp}' doesn't exist or is not a file.")


  # Load images.
  print("> Loading images...")
  image0 = np.array(Image.open(image0_fp).convert("RGB"))
  image1 = np.array(Image.open(image1_fp).convert("RGB"))

  # Load models.
  print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
  start = time.time()
  og = omniglue.OmniGlue(
      og_export="./checkpoints/omniglue.onnx",
      sp_export="./checkpoints/sp_v6.onnx",
      dino_export="./checkpoints/dinov2_vitb14_pretrain.pth",
  )
  print(f"> \tTook {time.time() - start} seconds.")

  # Perform inference.
  print("> Finding matches...")
  start = time.time()
  match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)
  num_matches = match_kp0.shape[0]
  print(f"> \tFound {num_matches} matches.")
  print(f"> \tTook {time.time() - start} seconds.")

  # Filter by confidence (0.02).
  print("> Filtering matches...")
  match_threshold = 0.01  # Choose any value [0.0, 1.0).
  keep_idx = []
  for i in range(match_kp0.shape[0]):
    if match_confidences[i] > match_threshold:
      keep_idx.append(i)
  num_filtered_matches = len(keep_idx)
  match_kp0 = match_kp0[keep_idx]
  match_kp1 = match_kp1[keep_idx]
  match_confidences = match_confidences[keep_idx]
  print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")

  point_list0 = match_kp0
  point_list1 = match_kp1
  # Visualize.
  prompt_save = 'match_res'
  with open(prompt_save + '.txt', 'w') as f:
    for ii in match_kp0:
        f.write(str(ii[0]) + ',' + str(ii[1]) + '\n')
  print("> Visualizing matches...")
  viz = utils.visualize_matches(
      image0,
      image1,
      match_kp0,
      match_kp1,
      np.eye(num_filtered_matches),
      show_keypoints=True,
      highlight_unmatched=True,
      title=f"{num_filtered_matches} matches",
      line_width=1,
  )
  plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
  plt.axis("on")
  plt.imshow(viz)
  plt.imsave("./demo_output.png", viz)

  plt.show()
  #print("> \tSaved visualization to ./demo_output.png")


if __name__ == "__main__":
  path0 = "./Datasets/SPUS/DIFF/S1704/imgs/00113.jpg"
  mask0 = "./Datasets/SPUS/DIFF/S1704/mask/00113_mask.png"
  path1 = "./Datasets/SPUS/DIFF/S1602/imgs/00004.jpg"
  mask1 = "./Datasets/SPUS/DIFF/S1602/mask/00004_mask.png"
  #path0 = "./Datasets/MSKUSO/hp8/imgs/00114.jpg"
  #path1 = "./Datasets/MSKUSO/support/imgs/hp_00011.jpg"

  match(path0, path1)
  
  #raise ValueError("Incorrect command line usage - usage: python demo.py <img1_fp> <img2_fp>")
