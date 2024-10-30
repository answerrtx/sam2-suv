import os
import cv2

def read_bboxes_from_file(txt_file):
    """从txt文件中读取bbox，格式为x_min, y_min, x_max, y_max"""
    bboxes = []
    with open(txt_file, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))
            bboxes.append(coords)  # 每个坐标为 (x_min, y_min, x_max, y_max)
    return bboxes

def get_combined_bbox(bboxes):
    """合并所有的bbox，返回最左上和最右下的box"""
    x_min = min(bbox[0] for bbox in bboxes)
    y_min = min(bbox[1] for bbox in bboxes)
    x_max = max(bbox[2] for bbox in bboxes)
    y_max = max(bbox[3] for bbox in bboxes)
    return [x_min, y_min, x_max, y_max]

def draw_bbox_on_image(image, bbox):
    """在图像上绘制bbox"""
    x_min, y_min, x_max, y_max = map(int, bbox)  # 转换为整数
    return cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

def process_files(txt_directory, image_directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 遍历所有的txt文件
    txt_files = sorted([f for f in os.listdir(txt_directory) if f.endswith('.txt')])
    
    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]  # 提取不带扩展名的文件名，例如"00000"
        txt_path = os.path.join(txt_directory, txt_file)
        
        # 读取txt文件中的所有bboxes
        bboxes = read_bboxes_from_file(txt_path)
        
        # 合并bboxes
        combined_bbox = get_combined_bbox(bboxes)
        
        # 将新的bbox写入新的txt文件
        output_txt_path = os.path.join(output_directory, f"{base_name}.txt")
        with open(output_txt_path, 'w') as f_out:
            f_out.write(f"{combined_bbox[0]} {combined_bbox[1]} {combined_bbox[2]} {combined_bbox[3]}\n")
        
        # 读取对应的图片
        image_path = os.path.join(image_directory, f"{base_name}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image {image_path}, skipping...")
            continue
        
        # 在图片上画出合并后的bbox
        image_with_bbox = draw_bbox_on_image(image, combined_bbox)
        
        # 保存带有bbox的图片
        output_image_path = os.path.join(output_directory, f"{base_name}.jpg")
        cv2.imwrite(output_image_path, image_with_bbox)
        print(f"Processed: {txt_file} and saved new bbox and image to {output_directory}")

# 定义文件夹路径
txt_directory = './Datasets/CLU/MS011_ROIS'
image_directory = './Datasets/CLU/MS011'
output_directory = './Datasets/CLU/MS011_ROIS_AF'

# 执行处理操作
process_files(txt_directory, image_directory, output_directory)
