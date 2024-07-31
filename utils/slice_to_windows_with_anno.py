import os
import cv2

def read_annotations(labels_input_dir, image_name):
    """
    Read the annotations from the label file corresponding to the image.

    Args:
        labels_input_dir (str): Path to the directory containing the input labels.
        image_name (str): Name of the image file.

    Returns:
        list: List of annotations read from the label file.
    """

    # Generate the path to the label file
    label_file_name = os.path.splitext(image_name)[0] + ".txt"
    label_file_path = os.path.join(labels_input_dir, label_file_name)

    # Read the label file
    with open(label_file_path, "r") as file:
        lines = file.readlines()

    # Parse the annotations
    annotations = []
    for line in lines:
        parts = line.strip().split(" ")
        cls, x, y, w, h = map(float, parts)
        annotations.append((cls, x, y, w, h))

    return annotations

def write_annotations(file_path, annotations):
    """
    Write YOLO format annotations to a file.

    Args:
        file_path (str): Path to the output label file.
        annotations (list): List of annotations to be written.
    """
    with open(file_path, 'w') as file:
        for ann in annotations:
            file.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

def convert_annotations(annotations, x_offset, y_offset, window_size, img_width, img_height):
    """
    Convert annotations from the large image to the cropped window.

    Args:
        annotations (list): List of annotations from the large image.
        x_offset (int): X-coordinate of the top-left corner of the window.
        y_offset (int): Y-coordinate of the top-left corner of the window.
        window_size (int): Size of the sliding window (both width and height).
        img_width (int): Width of the large image.
        img_height (int): Height of the large image.

    Returns:
        list: List of converted annotations for the cropped window.
    """
    new_annotations = []
    for cls, x, y, w, h in annotations:
        # Convert normalized coordinates to absolute coordinates
        x_center = x * img_width
        y_center = y * img_height
        bbox_width = w * img_width
        bbox_height = h * img_height

        # 都是相对于大图而言的，标注大图中的像素而不是百分比位置
        xmin = x_center - bbox_width / 2
        ymin = y_center - bbox_height / 2
        xmax = x_center + bbox_width / 2
        ymax = y_center + bbox_height / 2
        
        # 检查是否中心在窗口内
        if x_offset <= x_center < x_offset + window_size and y_offset <= y_center < y_offset + window_size:
            # Convert coordinates to the cropped window
            # 确保新的bbox在窗口内
            if xmin < x_offset: 
                xmin = x_offset
            if ymin < y_offset:
                ymin = y_offset
            if xmax > x_offset + window_size:
                xmax = x_offset + window_size
            if ymax > y_offset + window_size:
                ymax = y_offset + window_size
            # 计算新的bbox的中心点和宽高，相对于窗口而言，但是不是百分比
            new_x_center = (xmin + xmax) / 2 - x_offset
            new_y_center = (ymin + ymax) / 2 - y_offset
            new_bbox_width = xmax - xmin
            new_bbox_height = ymax - ymin
            
            # 换算成百分比
            new_x_center /= window_size
            new_y_center /= window_size
            new_bbox_width /= window_size
            new_bbox_height /= window_size
            

            new_annotations.append((cls, new_x_center, new_y_center, new_bbox_width, new_bbox_height))

    return new_annotations

def slide_and_save_images_and_labels(images_input_dir, images_output_dir, labels_input_dir, labels_output_dir, window_size, step_size, image_ext=".jpg"):
    """
    Slide a window across each image in the input directory, crop the window, and save it as a separate image.

    Args:
        images_input_dir (str): Path to the directory containing the input images.
        images_output_dir (str): Path to the directory where the cropped images will be saved.
        labels_input_dir (str): Path to the directory containing the input labels.
        labels_output_dir (str): Path to the directory where the cropped labels will be saved.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for sliding the window.

    Returns:
        None
    """

    # Ensure output directory exists
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    
    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)

    # Iterate over all files in the input directory
    for image_name in os.listdir(images_input_dir):
        image_path = os.path.join(images_input_dir, image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Skip if the file is not a valid image
        if image is None:
            continue
        
        annotations = read_annotations(labels_input_dir, image_name)
        
        height, width = image.shape[:2]

        # Slide the window across the image
        for y in range(0, height - window_size + 1, step_size):
            for x in range(0, width - window_size + 1, step_size):
                # Crop the window from the image
                cropped_image = image[y:y + window_size, x:x + window_size]
                
                # Generate the filename for the cropped image
                cropped_image_name = f"{os.path.splitext(image_name)[0]}_{x}_{y}{image_ext}"
                cropped_image_path = os.path.join(images_output_dir, cropped_image_name)
                
                # Save the cropped image
                cv2.imwrite(cropped_image_path, cropped_image)
                
                # Convert annotations for the cropped window
                window_annotations = convert_annotations(
                    annotations, x, y, window_size, width, height)
                
                # Generate the filename for the cropped label file
                cropped_label_name = f"{os.path.splitext(image_name)[0]}_{x}_{y}.txt"
                cropped_label_path = os.path.join(labels_output_dir, cropped_label_name)
                
                # Save the cropped labels
                write_annotations(cropped_label_path, window_annotations)
                
                print(f"Saved: {cropped_image_path} with annotations {cropped_label_path}")

# User-defined parameters
window_size = 608  # Size of the sliding window (both width and height)
step_size = 480  # Step size for the sliding window

# Run the function
if __name__ == '__main__':
    root_dir = "PCB_chusaiyangliji"
    
    images_input_directories_suffix = "_Img"
    labels_input_directories_suffix = "_txt"
    
    images_output_dir = "images_output"
    labels_output_dir = "labels_output"
    
    input_directories = ["Missing_hole", 
                        "Mouse_bite", 
                        "Open_circuit", 
                        "Short", 
                        "Spur", 
                        "Spurious_copper"
                        ]
    
    # input_directories = [
    #                     "Mouse_bite", 
    #                     ]
    
    for input_directory in input_directories:
        images_input_dir = root_dir + '/'+ input_directory + images_input_directories_suffix
        labels_input_dir = root_dir + '/'+ input_directory + labels_input_directories_suffix
        images_ext = ".bmp"
        slide_and_save_images_and_labels(images_input_dir, images_output_dir, labels_input_dir, labels_output_dir, window_size, step_size, images_ext)
