import cv2
import os

# 定义函数来以固定FPS拆分MP4视频成图片
def split_video_to_frames(video_path, output_folder, desired_fps):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取原始视频的帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / desired_fps)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束
            break

        # 只保存间隔符合的帧
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"保存帧 {frame_count} 为 {frame_filename}")
            saved_frame_count += 1

        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"视频已以 {desired_fps} FPS 拆分为 {saved_frame_count} 张图片，保存在 {output_folder} 文件夹中。")

# 示例：使用该函数拆分视频，指定所需的输出FPS
video_path = './Datasets/YTU/MP4/sp7.mp4'  # 输入视频文件路径
output_folder = './Datasets/YTU/sp7/'  # 保存图片的文件夹
desired_fps = 15  # 目标帧率

split_video_to_frames(video_path, output_folder, desired_fps)
