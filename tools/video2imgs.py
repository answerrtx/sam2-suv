import cv2
import os

# 定义函数来拆分MP4视频成图片
def split_video_to_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束
            break

        # 生成图片文件名
        frame_filename = os.path.join(output_folder, f"{frame_count:05d}.jpg")

        # 保存帧为图片
        cv2.imwrite(frame_filename, frame)
        print(f"保存帧 {frame_count} 为 {frame_filename}")
        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"视频已拆分为 {frame_count} 张图片，保存在 {output_folder} 文件夹中。")

# 示例：使用该函数拆分视频
video_path = 'usseg/Datasets/YTU/hnd2.mp4'  # 输入视频文件路径
output_folder = 'usseg/Datasets/YTU/hnd2/'  # 保存图片的文件夹

split_video_to_frames(video_path, output_folder)
