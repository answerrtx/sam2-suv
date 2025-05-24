import os

def rename_files(directory):
    # 获取文件夹中所有的文件名，并排序
    files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    sorted_file_list = sorted(files, key=lambda x: int(x[-9:-4]))

    # 遍历文件并重命名
    for idx, filename in enumerate(files):
        # 构造旧文件的完整路径
        old_path = os.path.join(directory, filename)
        
        # 创建新的文件名，格式为五位数字从0开始，扩展名保持为.jpg
        new_filename = f"{idx:05}.jpg"
        new_path = os.path.join(directory, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

# 定义要处理的文件夹路径
directory = './Datasets/SRS/SRS323'

# 执行重命名操作
rename_files(directory)
