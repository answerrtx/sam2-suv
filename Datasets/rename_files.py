import os

def rename_images(folder):
    # 获取所有jpg文件并排序
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])
    
    for idx, filename in enumerate(files):
        old_path = os.path.join(folder, filename)
        new_filename = f"{idx:05d}.jpg"
        new_path = os.path.join(folder, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")

if __name__ == "__main__":
    folder_path = "./SPUS_09_247_394"  # 这里替换为你的图片文件夹路径
    rename_images(folder_path)
