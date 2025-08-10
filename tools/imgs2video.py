import os
import subprocess

def images_to_mp4(input_folder, output_file, frame_rate=30):
    """
    将文件夹中的图片合成一个 MP4 视频，并确保视频宽高为 2 的倍数。

    :param input_folder: 图片所在的文件夹路径
    :param output_file: 输出的 MP4 文件路径
    :param frame_rate: 视频的帧率，默认为 30
    """
    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        print("输入文件夹不存在。")
        return

    # 检查文件夹中是否有图片
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith('.jpg')
    ])
    if not image_files:
        print("输入文件夹中没有找到符合格式的图片文件。")
        return

    # 使用 ffmpeg 合成视频，并调整分辨率
    input_pattern = os.path.join(input_folder, "%05d.jpg")  # 对应 00000.jpg, 00001.jpg 格式
    command = [
        "ffmpeg",
        "-framerate", str(frame_rate),
        "-i", input_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 确保宽高为 2 的倍数
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"成功生成视频：{output_file}")
    except subprocess.CalledProcessError as e:
        print(f"生成视频失败，错误信息: {e}")

# 使用示例
input_folder = "./Datasets/SPC-mini/ss1704/imgs"  # 替换为图片文件夹路径
output_file = "./Datasets/SPC-mini/ss1704/output_video.mp4"  # 确保指定扩展名
frame_rate = 30  # 可调整帧率

images_to_mp4(input_folder, output_file, frame_rate)
