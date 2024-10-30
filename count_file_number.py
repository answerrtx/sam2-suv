import os

def count_images_in_folders(directory, image_extensions=('jpg', 'jpeg', 'png')):
    # 遍历目录中的每个文件夹和文件
    for root, dirs, files in os.walk(directory):
        image_count = 0
        if root.find('MS') < 0:
            continue
        for file in files:
            # 检查文件扩展名是否是图片类型
            if file.lower().endswith(image_extensions):
                image_count += 1
        
        # 输出每个文件夹的图片数量
        print(f"Folder: {root}, Images: {image_count}")

# 指定要统计的根目录
directory = './Datasets/CLU'

# 执行统计操作
count_images_in_folders(directory)
