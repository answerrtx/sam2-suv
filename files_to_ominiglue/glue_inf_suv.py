# run this script with CUDA_VISIBLE_DEVICES=1 python glue_inf_us.py 

#!/usr/bin/env python3

# only match with rois with iou > 0.7 

#CUDA_VISIBLE_DEVICES=1 python demo.py ./res/demo1.jpg ./res/demo2.jpg
# save to img_id.txt
# cur_id, match_id, # of ROI(0-based),  # of matched points, p1, p2, p3, p...
"""Demo script for performing OmniGlue inference."""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import omniglue
from omniglue import utils
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

def compute_iou(box0, box1):
    # box0 和 box1 都是 (x_min, y_min, x_max, y_max) 格式
    x_min0, y_min0, x_max0, y_max0 = box0
    x_min1, y_min1, x_max1, y_max1 = box1
    
    # 计算交集坐标
    inter_x_min = max(x_min0, x_min1)
    inter_y_min = max(y_min0, y_min1)
    inter_x_max = min(x_max0, x_max1)
    inter_y_max = min(y_max0, y_max1)
    
    # 交集宽度和高度
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    
    # 交集面积
    inter_area = inter_width * inter_height
    
    # 计算每个矩形的面积
    area0 = (x_max0 - x_min0) * (y_max0 - y_min0)
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    
    # 计算并集面积
    union_area = area0 + area1 - inter_area
    
    # 计算 IoU
    if union_area == 0:
        return 0  # 防止除以 0
    iou = inter_area / union_area
    return iou

def crop_image(image, coords):
    crops = []
    for coord in coords:
        x_min, y_min, x_max, y_max = map(int, coord)    # Convert to int
        cropped_image = image[y_min:y_max, x_min:x_max]    # Crop the image
        crops.append((cropped_image, (x_min, y_min)))    # 保存裁剪的图像和左上角坐标
    return crops

def read_crop_coordinates(filename):
        coordinates = []
        with open(filename, 'r') as file:
                for line in file:
                        coords = list(map(float, line.strip().split()))
                        coordinates.append(coords)
        return coordinates

def lookup(image0_fp, image1_fp, anno0, anno1, vis_name, prompt_save):
    for im_fp in [image0_fp, image1_fp]:
        if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
            raise ValueError(f"Image filepath '{im_fp}' doesn't exist or is not a file.")
    coords0 = read_crop_coordinates(anno0)
    coords1 = read_crop_coordinates(anno1)

    # Load images.
    print("> Loading images...")
    ima0 = np.array(Image.open(image0_fp).convert("RGB"))
    ima1 = np.array(Image.open(image1_fp).convert("RGB"))

    num_roi = 0
    thresh= 0.7
    num_c0 = len(coords0)
    num_c1 = len(coords1)
    prompt_list0 = []
    prompt_list1 = []
    for p_roi1 in range(num_c0):
        
        x_min0, y_min0, x_max0, y_max0 = map(int, coords0[p_roi1]) 
        cd0 = [x_min0, y_min0, x_max0, y_max0]
        for p_roi2 in range(num_c1):
            num_roi += 1
            x_min1, y_min1, x_max1, y_max1 = map(int, coords1[p_roi2]) 
            cd1 = [x_min1, y_min1, x_max1, y_max1]
            if(compute_iou(cd0, cd1) < thresh):
                continue
            else: print("!!!", p_roi1, p_roi2)
            #print(cd0)
            # 获取裁剪的图片和左上角的坐标
            roi0, offset0 = crop_image(ima0, [cd0])[0]
            roi1, offset1 = crop_image(ima1, [cd1])[0]
            # Load models.
            #global og
            # Perform inference.
            print("> Finding matches...")
            start = time.time()

            match_kp0, match_kp1, match_confidences = og.FindMatches(roi0, roi1)
            num_matches = match_kp0.shape[0]
            print(f"> \tFound {num_matches} matches.")
            print(f"> \tTook {time.time() - start} seconds.")

            # Filter by confidence (0.02).
            print("> Filtering matches...")
            match_threshold = 0.02    # Choose any value [0.0, 1.0).
            keep_idx = []
            for i in range(match_kp0.shape[0]):
                if match_confidences[i] > match_threshold:
                    keep_idx.append(i)
            num_filtered_matches = len(keep_idx)
            if num_filtered_matches==0: continue

            match_kp0 = match_kp0[keep_idx]
            match_kp1 = match_kp1[keep_idx]
            match_confidences = match_confidences[keep_idx]
            print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")

            # 还原匹配点的坐标到原图像坐标系
            x_offset0, y_offset0 = offset0
            x_offset1, y_offset1 = offset1

            match_kp0_original = match_kp0 + [x_offset0, y_offset0]  # 还原到原图
            match_kp1_original = match_kp1 + [x_offset1, y_offset1]  # 还原到原图

            #print(match_kp0.shape)
            #print(match_kp0[0][0], "---" , match_kp0[0][1])
            #print(match_kp0_original.shape, type(match_kp0_original))

            # prompt_list0 是每个roi1找到的match的的点的坐标
            prompt_list0.append(match_kp0_original)
            prompt_list1.append(match_kp1_original)
            with open(prompt_save + '.txt', 'w') as f:

                f.write(str(p_roi1) + ' ')
                f.write(str(len(match_kp0_original)) + '\n')
                
                for ii in match_kp0_original:
                    f.write(str(ii[0]) + ',' + str(ii[1]) + '\n')
                # 第p_roi1个roi里面找到的match points


    if prompt_list0:
        print("> Visualizing matches...")

        #print(prompt_list0, vis_name,'!!')

                
        kp0s = np.vstack(prompt_list0)
        kp1s = np.vstack(prompt_list1)
        # Visualize.
        viz = utils.visualize_matches(
                ima0,
                ima1,
                kp0s,
                kp1s,
                np.eye(len(kp0s)),
                show_keypoints=True,
                highlight_unmatched=True,
                title=f"{num_filtered_matches} matches",
                line_width=1,
        )
        plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
        plt.axis("off")
        plt.imshow(viz)
        plt.imsave(vis_name, viz)
        print("> \tSaved visualization")


def main(argv):
    FOLDER= argv[1]
    COORD_FOLDER = FOLDER+'_ROIS_AF'
    VIS_FOLDER = FOLDER + '_MATCHES'
    print(FOLDER)
    os.makedirs(VIS_FOLDER, exist_ok=True)
    file_list = []
    anno_list = []
    name_list = []
    for file_name in os.listdir(FOLDER):
        # 构建文件的完整路径并添加到列表中
        name_list.append(file_name.split('.')[0])
        file_list.append(os.path.join(FOLDER, file_name))
        anno_list.append(os.path.join(COORD_FOLDER, file_name.replace('.jpg', '.txt')))
        
    sorted_file_list = sorted(file_list)#, key=lambda x: int(x[-9:-4]))
    sorted_anno_list = sorted(anno_list)#, key=lambda x: int(x[-9:-4]))
    name_list = sorted(name_list)
    print(sorted_file_list[0])
    print(sorted_anno_list[0])
    file_num = len(sorted_file_list)

    print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
    start = time.time()
    global og 
    og = omniglue.OmniGlue(
            og_export="./models/og_export",
            sp_export="./models/sp_v6",
            dino_export="./models/dinov2_vitb14_pretrain.pth",
    )
    print(f"> \tTook {time.time() - start} seconds.")


    for i in range(file_num):
        #if i >2:
        #    break
        cur = i
        step = 1
        #while(1):
        #    if step > min(file_num / 2, 32): break
        if (cur + step) > file_num - 1: break
        #        step *= 2 
        #    else: 
        image0_fp = sorted_file_list[cur]
        image1_fp = sorted_file_list[cur + step]
        anno0 = sorted_anno_list[cur]
        print(anno0)

        anno1 = sorted_anno_list[cur + step]
        vis_save = os.path.join(VIS_FOLDER, name_list[cur] + '_' + name_list[cur + step] + '.jpg')
        prompt_save = vis_save.replace('.jpg', '')
        lookup(image0_fp, image1_fp, anno0, anno1, vis_save, prompt_save)

if __name__ == "__main__":
    main(sys.argv)
