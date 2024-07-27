import os
import cv2
import xml.etree.ElementTree as ET

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((obj_name, (xmin, ymin, xmax, ymax)))
        
    return objects

def draw_bounding_boxes(image, objects):
    for obj in objects:
        name, (xmin, ymin, xmax, ymax) = obj
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def annotate_image(image_name, annotations_folder, images_folder, output_folder):
    # Construct paths
    xml_file = os.path.join(annotations_folder, f"{os.path.splitext(image_name)[0]}.xml")
    image_file = os.path.join(images_folder, image_name)
    
    # Check if files exist
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"Annotation file not found: {xml_file}")
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file not found: {image_file}")
    
    # Parse annotations
    objects = parse_annotation(xml_file)
    
    # Read image
    image = cv2.imread(image_file)
    if image is None:
        raise ValueError(f"Failed to read image: {image_file}")
    
    # Draw bounding boxes
    annotated_image = draw_bounding_boxes(image, objects)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the annotated image
    output_file = os.path.join(output_folder, image_name)
    cv2.imwrite(output_file, annotated_image)
    print(f"Annotated image saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    image_name = "01_spur_11.jpg"  # Replace with the user-provided image name
    annotations_folder = "train"  # Replace with the path to your annotations folder
    images_folder = "train"  # Replace with the path to your images folder
    output_folder = "anno_output"  # Replace with the path to your output folder
    
    annotate_image(image_name, annotations_folder, images_folder, output_folder)
