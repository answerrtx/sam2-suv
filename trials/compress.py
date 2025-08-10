import os
import cv2

def images_to_video(image_folder, output_video_path, fps=30):
    """
    Compress images in a folder into an MP4 video.
    
    :param image_folder: Path to the folder containing images.
    :param output_video_path: Path to save the output video (e.g., "output.mp4").
    :param fps: Frames per second for the video.
    """
    # Get all image files in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    # Read the first image to get the frame size
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # Define the video codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error reading image {image_file}, skipping...")
            continue
        
        # Resize image to match the first frame's size if necessary
        if (image.shape[1], image.shape[0]) != (width, height):
            image = cv2.resize(image, (width, height))
        
        video_writer.write(image)
        print(f"Added {image_file} to the video.")
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")

# Folder containing images
image_folder = './Datasets/YTU/hnd4/imgs'  # Change this to the folder path containing your images

# Output video file
output_video_path = './hnd4.mp4'

# Create video
images_to_video(image_folder, output_video_path, fps=30)
