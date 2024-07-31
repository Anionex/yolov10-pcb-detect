"""
生成当前目录下图片翻转90、180、270度的图片，存储到当前目录下   
"""
import os
import cv2

def rotate_and_save_images(input_dir):
    """
    Rotates images in the input directory by 90, 180, and 270 degrees, and saves them with new names.

    :param input_dir: Directory containing the input images
    """
    # Define the rotation angles
    angles = [90, 180, 270]

    # Iterate over all files in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Skip if the file is not a valid image
        if image is None:
            continue
        
        # Perform rotations and save the images
        for angle in angles:
            # Rotate the image
            if angle == 90:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Generate the filename for the rotated image
            rotated_image_name = f"{os.path.splitext(image_name)[0]}_{angle}.png"
            rotated_image_path = os.path.join(input_dir, rotated_image_name)
            
            # Save the rotated image
            cv2.imwrite(rotated_image_path, rotated_image)
            print(f"Saved: {rotated_image_path}")

# User-defined parameter
input_directory = '.'  # Current directory

# Run the function
rotate_and_save_images(input_directory)
