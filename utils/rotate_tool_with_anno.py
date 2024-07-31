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

def rotate_image_and_annotations(image, annotations, angle, img_width, img_height):
    """
    Rotate the image and its annotations by a given angle.

    Args:
        image (ndarray): The input image.
        annotations (list): List of annotations to be rotated.
        angle (int): The angle to rotate the image and annotations.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        rotated_image (ndarray): The rotated image.
        rotated_annotations (list): The rotated annotations.
    """
    if angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_annotations = [(cls, 1 - y, x, h, w) for cls, x, y, w, h in annotations]
    elif angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        rotated_annotations = [(cls, 1 - x, 1 - y, w, h) for cls, x, y, w, h in annotations]
    elif angle == 270:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_annotations = [(cls, y, 1 - x, h, w) for cls, x, y, w, h in annotations]
    else:
        raise ValueError("Angle must be 90, 180, or 270 degrees")

    return rotated_image, rotated_annotations

def process_images_and_labels(images_input_dir, images_output_dir, labels_input_dir, labels_output_dir, image_ext=".jpg"):
    """
    Process each image and its annotations by rotating them 90, 180, and 270 degrees.

    Args:
        images_input_dir (str): Path to the directory containing the input images.
        images_output_dir (str): Path to the directory where the processed images will be saved.
        labels_input_dir (str): Path to the directory containing the input labels.
        labels_output_dir (str): Path to the directory where the processed labels will be saved.

    Returns:
        None
    """

    # Ensure output directory exists
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    
    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)

    # Define rotation angles
    angles = [90, 180, 270]

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

        # Save the original image and annotations
        original_image_path = os.path.join(images_output_dir, image_name)
        cv2.imwrite(original_image_path, image)
        original_label_path = os.path.join(labels_output_dir, os.path.splitext(image_name)[0] + ".txt")
        write_annotations(original_label_path, annotations)
        
        # Rotate and save the images and annotations
        for angle in angles:
            rotated_image, rotated_annotations = rotate_image_and_annotations(image, annotations, angle, width, height)
            
            rotated_image_name = f"{os.path.splitext(image_name)[0]}_rotate_{angle}{image_ext}"
            rotated_image_path = os.path.join(images_output_dir, rotated_image_name)
            cv2.imwrite(rotated_image_path, rotated_image)
            
            rotated_label_name = f"{os.path.splitext(image_name)[0]}_rotate_{angle}.txt"
            rotated_label_path = os.path.join(labels_output_dir, rotated_label_name)
            write_annotations(rotated_label_path, rotated_annotations)
            
            print(f"Saved: {rotated_image_path} with annotations {rotated_label_path}")

# User-defined parameters
images_input_dir = "images_output"
images_output_dir = "images_rotate_output"
labels_input_dir = "labels_output"
labels_output_dir = "labels_rotate_output"

# Run the function
if __name__ == '__main__':
    image_ext = ".bmp"
    process_images_and_labels(images_input_dir, images_output_dir, labels_input_dir, labels_output_dir, image_ext=image_ext)
