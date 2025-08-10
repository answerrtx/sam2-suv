import os

def rename_folders(directory):
    # 遍历指定目录中的所有文件和文件夹
    for folder_name in os.listdir(directory):
        # 构造完整的路径
        full_path = os.path.join(directory, folder_name)
        
        # 检查是否为文件夹
        if os.path.isdir(full_path):
            # 如果文件夹名包含下划线，则进行修改
            new_folder_name = folder_name.replace('_', '')
            new_full_path = os.path.join(directory, new_folder_name)
            
            # 进行重命名
            os.rename(full_path, new_full_path)
            print(f"Renamed: {folder_name} -> {new_folder_name}")

# 指定要处理的文件夹路径
directory = './Datasets/CLU'

# 执行重命名操作
rename_folders(directory)
