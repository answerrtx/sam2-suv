import pydicom
import numpy as np
from PIL import Image
import os

def dicom_volume_to_images(dicom_path, output_path, output_format="PNG"):
    # 确保输出格式正确
    if output_format.upper() == "JPG":
        output_format = "JPEG"  # Pillow 需要 "JPEG" 格式

    # 读取 DICOM 文件
    dicom_data = pydicom.dcmread(dicom_path)
    
    # 提取3D像素数据
    pixel_array = dicom_data.pixel_array
    
    # 检查是否为3D卷数据
    if pixel_array.ndim != 3:
        print("This DICOM is not a 3D volume.")
        return

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    
    # 定义裁剪的高度范围
    crop_top, crop_bottom = 65, 420

    # 遍历每个切片并保存
    for i in range(pixel_array.shape[0]):
        slice_array = pixel_array[i, crop_top:crop_bottom, :]  # 裁剪高度为65到415的部分
        
        # 归一化数据到 0-255 之间
        normalized_array = ((slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array)) * 255).astype(np.uint8)
        
        # 转换为 PIL 图像
        image = Image.fromarray(normalized_array)
        
        # 设置输出文件名
        file_name = f"slice_{i:05}.{output_format.lower()}"
        output_file = os.path.join(output_path, file_name)
        
        # 保存图像
        image.save(output_file, format=output_format)
        print(f"Slice {i} saved at {output_file}")



output_format = "JPEG"                        # 输出格式，可以设置为 "PNG" 或 "JPG"

dicom_volume_to_images("./Datasets/CLU/dicom/s01_FINE.dcm", "./Datasets/CLU/S01", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S02.dcm", "./Datasets/CLU/S02", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/s03_.dcm", "./Datasets/CLU/S03", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/s04.dcm", "./Datasets/CLU/S04", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S05_2.dcm", "./Datasets/CLU/S05", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S06_2.dcm", "./Datasets/CLU/S06", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S07.dcm", "./Datasets/CLU/S07", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S08_2.dcm", "./Datasets/CLU/S08", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S09.dcm", "./Datasets/CLU/S09", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S10.dcm", "./Datasets/CLU/S10", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S11_2.dcm", "./Datasets/CLU/S11", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S12.dcm", "./Datasets/CLU/S12", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S13.dcm", "./Datasets/CLU/S13", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S14.dcm", "./Datasets/CLU/S14", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S15_2.dcm", "./Datasets/CLU/S15", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S16_2.dcm", "./Datasets/CLU/S16", output_format)
dicom_volume_to_images("./Datasets/CLU/dicom/S17.dcm", "./Datasets/CLU/S17", output_format)

