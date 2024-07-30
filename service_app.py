import gradio as gr
import os
import json
import cv2
import numpy as np
from customize_service import yolov10_detection
from skimage.filters import threshold_sauvola

# 初始化目标检测服务
service = yolov10_detection(model_name="yolov10", model_path="weights/best-3.pt")

# 用于缓存预测和标注结果的字典
cache = {}

# 预处理函数，调用自定义服务的预处理
def preprocess_image(image_path):
    post_data = {"input_txt": {os.path.basename(image_path): open(image_path, "rb")}}
    service._preprocess(post_data)
    return post_data

# 推理函数，调用自定义服务的推理
def inference(post_data):
    data = service._inference('ok')
    return data

# 后处理函数，调用自定义服务的后处理
def postprocess(data):
    result = service._postprocess(data)
    return result

# 读取标注文件
def read_annotations(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # 在此处添加数据集目录，会自动递归式查找标注文件
    search_dirs = [r"C:\Users\10051\Desktop\yolov10-test\yolov10\datasets\PCB_瑕疵初赛样例集", "datasets"]
    
    for search_dir in search_dirs:
        # 遍历目录及其所有子目录
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file == f"{base_name}.txt":
                    annotation_file = os.path.join(root, file)
                    with open(annotation_file, 'r') as f:
                        annotations = [line.strip().split() for line in f.readlines()]
                        print("found annotation file:", annotation_file)
                        return annotations
    
    return []

# 绘制检测结果函数，包括目标框和标注框
def draw_boxes(image_path, result):
    image = cv2.imread(image_path)

    # 定义一些绘图参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)
    line_type = 2

    # 绘制检测结果
    for i in range(len(result['detection_boxes'])):
        box = result['detection_boxes'][i]
        label = result['detection_classes'][i]
        score = result['detection_scores'][i]

        ymin, xmin, ymax, xmax = box
        # print(f"print a bbox at {xmin}, {ymin}, {xmax}, {ymax}")
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        
        label_text = f"{label}: {score:.2f}"
        cv2.putText(image, label_text, (int(xmin), int(ymin) - 10), font, font_scale, font_color, line_type)

    output_path = f"tmp_output/pred_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, image)
    return output_path

# 绘制标注框函数
from config import *
def draw_labels(image_path, annotations):
    image = cv2.imread(image_path)

    # 定义一些绘图参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 0, 0)
    line_type = 2

    # 绘制标注框
    for annotation in annotations:
        code, cx, cy, w, h = map(float, annotation)
        xmin = int((cx - w / 2) * image.shape[1])
        ymin = int((cy - h / 2) * image.shape[0])
        xmax = int((cx + w / 2) * image.shape[1])
        ymax = int((cy + h / 2) * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), font_color, 2)
        # label_text = f"Annotation: {int(code)}"
        label_text = f""
        cv2.putText(image, label_text, (xmin, ymin - 10), font, font_scale, font_color, line_type)

    output_path = f"tmp_output/label_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, image)
    return output_path

# 目标检测函数，综合以上各个步骤
def detect_objects(image_path):
    if image_path in cache:
        return cache[image_path]

    # 调用服务处理图像
    post_data = preprocess_image(image_path)
    data = inference(post_data)
    result = postprocess(data)
    

    
    # 绘制检测结果和标注框
    predicted_image_path = draw_boxes(image_path, result)
    

    # 将结果存入缓存
    cache[image_path] = predicted_image_path
    
    return predicted_image_path

# 定义Gradio界面
def display_predicted_image(image_path):
    predicted_image_path = detect_objects(image_path)
    return predicted_image_path

def display_labeled_image(image_path):
    print("starting display_labeled_image")
    # 读取标注文件
    annotations = read_annotations(image_path)
    labeled_image_path = draw_labels(image_path, annotations)
    
    return labeled_image_path

def update_image(choice, image_path):
    if choice == "Predicted Image":
        return display_predicted_image(image_path)
    elif choice == "Labeled Image":
        return display_labeled_image(image_path)
    elif choice == "Grayscale Image":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        output_path = f"tmp_output/gray_{os.path.basename(image_path)}"
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        cv2.imwrite(output_path, image)
        return output_path
    elif choice == "Edge Image":
        # 读取图像
        kernel_size = 3
        low_threshold = 50
        high_threshold = 150
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯滤波去噪
        gray_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)
        
        # 形态学处理（可选）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    elif choice == "Binary Image":
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        def gamma_correction(img, gamma=0.3):
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img, table)
        
        gamma_corrected_image = gamma_correction(image)

        def sauvola_threshold(img, window_size=25, k=0.3):
            thresh_sauvola = threshold_sauvola(img, window_size=window_size, k=k)
            binary_sauvola = img > thresh_sauvola
            return (binary_sauvola * 255).astype(np.uint8)
        
        binary_image = sauvola_threshold(gamma_corrected_image)
        
        # 形态学处理（可选）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        return binary_image



        
                    

def run_batch_val():
    pass

image_input = gr.Image(type="filepath", label="Upload Image")
image_output = gr.Image(type="filepath", label="Output Image")
clear_cache = gr.Button("Clear Cache(delete all cached results)")
run_batch_val_btn = gr.Button("Run Batch Validation")
# 创建一个组件让用户选择validation set的位置
val_dir_input = gr.File(file_count="directory", label="Validation Set Directory", height="2vh")

with gr.Blocks() as demo:
    with gr.Row():
        image_input.render()
    with gr.Row():
        btn_pred = gr.Button("Show Predicted Image")
        btn_label = gr.Button("Show Labeled Image")
        btn_grayscale = gr.Button("Show Grayscale Image")
        btn_binary = gr.Button("Show Binary Image")
        # 边缘检测按钮
        btn_edge = gr.Button("Show Edge Image")
    clear_cache.render()
    with gr.Row():
        image_output.render()
    run_batch_val_btn.render()
    val_dir_input.render()

    btn_pred.click(update_image, inputs=[gr.State("Predicted Image"), image_input], outputs=image_output)
    btn_label.click(update_image, inputs=[gr.State("Labeled Image"), image_input], outputs=image_output)
    btn_grayscale.click(update_image, inputs=[gr.State("Grayscale Image"), image_input], outputs=image_output)
    btn_binary.click(update_image, inputs=[gr.State("Binary Image"), image_input], outputs=image_output)
    btn_edge.click(update_image, inputs=[gr.State("Edge Image"), image_input], outputs=image_output)
    
    clear_cache.click(lambda: cache.clear())
    run_batch_val_btn.click(run_batch_val)
# 启动Gradio应用
if __name__ == "__main__":
    # 创建临时输出目录
    if not os.path.exists("tmp_output"):
        os.makedirs("tmp_output")
    demo.launch()
