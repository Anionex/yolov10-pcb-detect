import os
import cv2

def slide_and_save_images(input_dir, output_dir, window_size, step_size):
    """
    将输入目录中的图像分割为指定大小和步长的小窗口，并将它们保存到输出目录中。

    :param input_dir: 包含输入图像的目录
    :param output_dir: 保存裁剪图像的目录
    :param window_size: 窗口的大小（宽度和高度）
    :param step_size: 滑动窗口的步长
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Skip if the file is not a valid image
        if image is None:
            continue
        
        height, width = image.shape[:2]

        # Slide the window across the image
        for y in range(0, height - window_size + 1, step_size):
            for x in range(0, width - window_size + 1, step_size):
                # Crop the window from the image
                cropped_image = image[y:y + window_size, x:x + window_size]
                
                # Generate the filename for the cropped image
                cropped_image_name = f"{os.path.splitext(image_name)[0]}_{x}_{y}.png"
                cropped_image_path = os.path.join(output_dir, cropped_image_name)
                
                # Save the cropped image
                cv2.imwrite(cropped_image_path, cropped_image)
                
                print(f"Saved: {cropped_image_path}")

# User-defined parameters
input_directory = 'PCB_USED'  # Directory containing the images
output_directory = 'pcb_uesd_output_images'  # Directory to save the cropped images
window_size = 600  # Size of the sliding window (both width and height)
step_size = 480  # Step size for the sliding window

# Run the function
slide_and_save_images(input_directory, output_directory, window_size, step_size)
