import os
import cv2

def rename_and_crop_files(directory, output_directory):
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 获取文件夹中所有的文件名，并排序
    files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    sorted_file_list = sorted(files, key=lambda x: int(x[-9:-4]))

    # 遍历文件并重命名
    for idx, filename in enumerate(sorted_file_list):
        # 构造旧文件的完整路径
        old_path = os.path.join(directory, filename)
        
        # 读取图像
        image = cv2.imread(old_path)
        if image is None:
            print(f"Error reading image {filename}, skipping...")
            continue
        
        # 裁剪图片，仅保留height从65到410的内容
        cropped_image = image[65:410, :]  # 保留 65 到 410 行，所有列 for SRS
        #cropped_image = image[65:415, :] # for CLU
        # 创建新的文件名，格式为五位数字从0开始，扩展名保持为.jpg
        new_filename = f"{idx:05}.jpg"
        new_path = os.path.join(output_directory, new_filename)
        
        # 将裁剪后的图像保存到输出目录
        cv2.imwrite(new_path, cropped_image)
        print(f"Renamed and cropped: {filename} -> {new_filename}")

# 定义要处理的文件夹路径
directory = './Datasets/SRS/SRS319'
output_directory = './Datasets/SRS/MSRS319'

# 执行重命名和裁剪操作
rename_and_crop_files(directory, output_directory)
