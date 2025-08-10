import os
import cv2
import random

def rename_and_crop_files(directory, output_directory):
    # 获取文件夹中所有的文件名，并排序
    files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    sorted_file_list = sorted(files, key=lambda x: int(x[-9:-4]))

    cnt = 0  # 用于跟踪当前的文件夹计数
    idx = 0  # 用于文件命名的计数器
    current_output_directory = f"{output_directory}_{cnt}"
    
    # 确保初始输出目录存在，如果不存在则创建
    if not os.path.exists(current_output_directory):
        os.makedirs(current_output_directory)

    # 生成一个随机数字，范围在200到400之间
    threshold = random.randint(200, 400)
    cntt = 0
    # 遍历文件并重命名
    with open(output_directory+'.txt', 'w') as f:
        for filename in sorted_file_list:
            cntt += 1
            # 构造旧文件的完整路径
            old_path = os.path.join(directory, filename)
            
            # 读取图像
            image = cv2.imread(old_path)
            if image is None:
                print(f"Error reading image {filename}, skipping...")
                continue
            
            # 裁剪图片，仅保留height从65到415的内容
            cropped_image = image[65:415, :]  # 适用于 CLU 的裁剪
            
            # 创建新的文件名，格式为五位数字从0开始，扩展名保持为.jpg
            new_filename = f"{idx:05}.jpg"
            new_path = os.path.join(current_output_directory, new_filename)
            
            # 将裁剪后的图像保存到输出目录
            cv2.imwrite(new_path, cropped_image)
            f.write(f"Renamed and cropped: {filename} -> {new_filename} at {current_output_directory}: {cntt}\n")

            # 更新 idx 计数器
            idx += 1
            
            # 当 idx 达到随机生成的数字时，重置 idx 并更新 output_directory
            if idx >= threshold:
                # 重置 idx 并生成新的文件夹
                cnt += 1
                idx = 0
                current_output_directory = f"{output_directory}_{cnt}"
                threshold = random.randint(100, 300)  # 生成新的随机阈值

                # 确保新输出目录存在
                if not os.path.exists(current_output_directory):
                    os.makedirs(current_output_directory)

for i in range(1,5):
    print(i)
    # 定义要处理的文件夹路径
    directory = './Datasets/CLU/S0'+str(i)
    output_directory = './Datasets/CLU/MS0'+str(i)

    # 执行重命名和裁剪操作
    rename_and_crop_files(directory, output_directory)
'''for i in range(5,10):
    print(i)
    # 定义要处理的文件夹路径
    directory = './Datasets/CLU/S0'+str(i)
    output_directory = './Datasets/CLU/MS0'+str(i)

    # 执行重命名和裁剪操作
    rename_and_crop_files(directory, output_directory)

for i in range(11,18):
    print(i)
    # 定义要处理的文件夹路径
    directory = './Datasets/CLU/S'+str(i)
    output_directory = './Datasets/CLU/MS'+str(i)

    # 执行重命名和裁剪操作
    rename_and_crop_files(directory, output_directory)'''