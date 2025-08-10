import cv2
import os

# 设置图片文件夹路径和输出视频路径
image_folder = 'dev_test/SPUS_09_247_394'  # 替换为你图片所在的文件夹
output_video = 'SPUS_09_247_394_2.mp4'
fps = 24  # 设置帧率（可以改成 15、25、60 等）

# 获取所有图片文件名并排序
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images.sort()  # 确保帧顺序正确

# 读取第一张图来获取尺寸
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# 定义视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID' / 'avc1' / 'H264'
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 写入每一帧
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print(f"✅ Video saved to {output_video}")
