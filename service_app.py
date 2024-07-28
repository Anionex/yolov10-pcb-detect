import gradio as gr
import os
import json
import cv2
import numpy as np
from customize_service import yolov10_detection

# 初始化目标检测服务
service = yolov10_detection(model_name="yolov10", model_path="weights/best-A800-mix.pt")

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
    search_dirs = [r"C:\Users\10051\Desktop\yolov10-test\yolov10\datasets\PCB_瑕疵初赛样例集", "datasets/pcb-defect-dataset"]
    
    for search_dir in search_dirs:
        # 遍历目录及其所有子目录
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file == f"{base_name}.txt":
                    annotation_file = os.path.join(root, file)
                    with open(annotation_file, 'r') as f:
                        annotations = [line.strip().split() for line in f.readlines()]
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
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
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
    
    # 读取标注文件
    annotations = read_annotations(image_path)
    
    # 绘制检测结果和标注框
    predicted_image_path = draw_boxes(image_path, result)
    labeled_image_path = draw_labels(image_path, annotations)

    # 将结果存入缓存
    cache[image_path] = (predicted_image_path, labeled_image_path)
    
    return predicted_image_path, labeled_image_path

# 定义Gradio界面
def display_predicted_image(image_path):
    predicted_image_path, _ = detect_objects(image_path)
    return predicted_image_path

def display_labeled_image(image_path):
    _, labeled_image_path = detect_objects(image_path)
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
        # with open("tmp_output/gray.txt", "w") as f:
        #     # 转换为普通数组
        #     image2 = image.tolist()
        #     f.write(str(image2))
        
        cv2.imwrite(output_path, image)
        return output_path
    elif choice == "Binary Image":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 高斯滤波，用于平滑图片，去除噪声
        image = cv2.GaussianBlur(image,(5,5),0)
        _, binary_image = cv2.threshold(image, 75, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        output_path = f"tmp_output/binary_{os.path.basename(image_path)}"
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(output_path, binary_image)
        return output_path
    elif choice == "Edge Image":
        def example_method2(image_path):
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # # 预处理（去噪和模糊处理）
            blurred = cv2.medianBlur(image, 5)  # 中值滤波去噪
            blurred = cv2.GaussianBlur(blurred, (5, 5), 0)  # 高斯模糊
            # 自适应阈值二值化
            binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
            
            # binary_image = cv2.medianBlur(binary_image, 3)  # 中值滤波去噪
            # binary_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
            return binary_image 
            # 形态学处理（闭运算和膨胀腐蚀）
            kernel = np.ones((3, 3), np.uint8)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算
            binary_image = cv2.dilate(binary_image, kernel, iterations=2)  # 膨胀
            binary_image = cv2.erode(binary_image, kernel, iterations=2)  # 腐蚀
            
            # 连通组件分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

            # 后处理（根据面积过滤小的连通组件）
            min_area = 500  # 根据实际情况调整
            filtered_binary_image = np.zeros(binary_image.shape, dtype=np.uint8)
            for i in range(1, num_labels):  # 跳过背景
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered_binary_image[labels == i] = 255

            # 轮廓检测和填充
            contours, _ = cv2.findContours(filtered_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filtered_binary_image, contours, -1, (255), thickness=cv2.FILLED)
            return filtered_binary_image
                    
        return example_method2(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edge_image = cv2.Canny(image, 100, 200)
        output_path = f"tmp_output/edge_{os.path.basename(image_path)}"
        edge_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(output_path, edge_image)
        return output_path

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
    if not os.path.exists("tmp_output"):
        os.makedirs("tmp_output")
    demo.launch()
