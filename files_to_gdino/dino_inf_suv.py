import os
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

print("?")
# Load the model
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "./weights/groundingdino_swint_ogc.pth")
print("load model!")
# Define the folder containing images
FOLDER_PATH = "./Datasets/CLU/MS011"#"./Datasets/SRS/MSRS323"
OUTPUT_FOLDER = "./Datasets/CLU/MS011_ROIS"#"./Datasets/SRS/MSRS323_ROIS"
OUTPUT_FOLDER_VIS = "./Datasets/CLU/MS011_ROIS"#"./Datasets/SRS/MSRS323_ROIS"
TEXT_PROMPT = "white."
BOX_THRESHOLD = 0.02
TEXT_THRESHOLD = 0.15

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER_VIS, exist_ok=True)
cnt = 1
# Loop through all files in the folder
for filename in os.listdir(FOLDER_PATH):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for image file types
        try:
            #if cnt>5:
            #    break
            cnt+=1
            # Load the image
            IMAGE_PATH = os.path.join(FOLDER_PATH, filename)
            image_source, image = load_image(IMAGE_PATH)

            # Define margins
            left_margin = 0
            right_margin = 0
            top_margin = 0
            bottom_margin = 0

            # Ensure that margins do not exceed the image dimensions
            height, width = image.shape[1], image.shape[2]

            if (height > top_margin + bottom_margin) and (width > left_margin + right_margin):
                # Crop the image with the defined margins
                cropped_image = image[:, 
                                    top_margin:height - bottom_margin,  # Crop top and bottom
                                    left_margin:width - right_margin    # Crop left and right
                                    ]
            else:
                print(f"Skipping {filename}: Image dimensions are too small for the specified margins.")
                continue  # Skip this image

            # Predict boxes, logits, and phrases using the cropped image
            boxes, logits, phrases = predict(
                model=model,
                image=cropped_image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            # Annotate the original image based on the predictions
            filter_xyxy, annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            #print(filter_xyxy)
            # Save the annotated image
            output_path = os.path.join(OUTPUT_FOLDER_VIS, f"annotated_{filename}")
            cv2.imwrite(output_path, annotated_frame)
            output_filename = os.path.splitext(filename)[0] + '.txt'
            full_output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            # Write the array to the text file
            with open(full_output_path, 'w') as f:
                for sub_array in filter_xyxy:
                    f.write(' '.join(map(str, sub_array)) + '\n')
                print(f"Saved annotated image: {output_path}")
        except Exception:
            print(filename)