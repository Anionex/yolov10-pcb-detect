import gradio as gr
import os
import json
import cv2
import numpy as np
from customize_service import yolov10_detection
from skimage.filters import threshold_sauvola

# 初始化目标检测服务
service = yolov10_detection(model_name="yolov10", model_path="weights/best-use-origin-and-grey-and-add-pcb-use-update-hyper-param-2.pt")
# service = yolov10_detection(model_name="yolov10", model_path="weights/best-use-origin-and-grey-and-add-pcb-use-update-hyper-param-2.pt")
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
    search_dirs = [r"C:\Users\10051\Desktop\yolov10-test\yolov10\utils\labels_output", r"C:\Users\10051\Desktop\yolov10-test\yolov10\datasets\PCB_瑕疵初赛样例集", "datasets"]
    
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
        return cache[image_path]["predicted_image_path"]

    # 调用服务处理图像
    post_data = preprocess_image(image_path)
    data = inference(post_data)
    result = postprocess(data)
    

    
    # 绘制检测结果和标注框
    predicted_image_path = draw_boxes(image_path, result)
    

    # 将结果存入缓存
    cache[image_path] = {
        "detection_boxes": result['detection_boxes'],
        "detection_classes": result['detection_classes'],
        "detection_scores": result['detection_scores'],
        "predicted_image_path": predicted_image_path
    }
    
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



        
                    

import glob
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score

import numpy as np
from sklearn.metrics import precision_recall_curve
import os
import json

def compute_iou(box1, box2):
    """
    计算两个边界框的 IOU
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def compute_ap(recall, precision):
    """
    计算平均精度（AP）
    """
    if len(recall) == 0 or len(precision) == 0:
        return 0

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return ap

def run_batch_test(test_dir, output_dir="tmp_output/test"):
    """
    本方法用于批量测试选定目录下的所有图像，输出检测后的图像，并且通过和标注文件自动比较，
    计算mAP50，最终输出到一个json文件中
    """
    cache.clear()  # 防止缓存影响测试结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_true_boxes = []
    all_pred_boxes = []
    all_true_labels = []
    all_pred_labels = []
    all_scores = []

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(root, file)

                # 获取标注结果
                annotations = read_annotations(image_path)
                if len(annotations) == 0:  # 如果没有标注文件，跳过
                    print(f"未找到标注文件: {image_path}")
                    continue

                # 获取检测结果
                predicted_image_path = display_predicted_image(image_path)

                # 读取检测结果框和标签
                pred_data = cache[image_path]
                pred_boxes = pred_data['detection_boxes']
                pred_labels = pred_data['detection_classes']
                scores = pred_data['detection_scores']

                # 读取标注框和标签
                true_boxes = []
                true_labels = []
                for annotation in annotations:
                    code, cx, cy, w, h = map(float, annotation)
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2
                    true_boxes.append([xmin, ymin, xmax, ymax])
                    true_labels.append(int(code))

                # 将结果加入总列表
                all_true_boxes.extend(true_boxes)
                all_pred_boxes.extend(pred_boxes)
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)
                all_scores.extend(scores)

    # 计算每个类别的平均精度（AP）和mAP
    unique_labels = list(set(all_true_labels))
    aps = []
    for label in unique_labels:
        true_label_boxes = [box for i, box in enumerate(all_true_boxes) if all_true_labels[i] == label]
        pred_label_boxes = [box for i, box in enumerate(all_pred_boxes) if all_pred_labels[i] == label]
        pred_label_scores = [score for i, score in enumerate(all_scores) if all_pred_labels[i] == label]

        if not true_label_boxes or not pred_label_boxes:
            continue

        # 计算 IOU
        true_positives = []
        scores = []
        for true_box in true_label_boxes:
            best_iou = 0
            best_pred_idx = -1
            for i, pred_box in enumerate(pred_label_boxes):
                iou = compute_iou(true_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            if best_iou > 0.5:
                true_positives.append(1)
                scores.append(pred_label_scores[best_pred_idx])
                pred_label_boxes.pop(best_pred_idx)
                pred_label_scores.pop(best_pred_idx)
            else:
                true_positives.append(0)
                scores.append(0)

        false_positives = [1 - tp for tp in true_positives]

        # 保证有正样本和负样本
        if len(true_positives) == 0 or len(scores) == 0:
            continue

        labels = true_positives + [0] * len(pred_label_boxes)
        scores = scores + pred_label_scores

        precision, recall, _ = precision_recall_curve(labels, scores)
        if len(precision) > 0 and len(recall) > 0:
            ap = compute_ap(recall, precision)
            aps.append(ap)

    mAP50 = np.mean(aps) if aps else 0

    result_summary = {
        "mAP50": mAP50
    }

    # 将结果写入JSON文件
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result_summary, f, indent=4)
    
    print(f"Batch test results saved to {os.path.join(output_dir, 'results.json')}")
    return result_summary

image_input = gr.Image(type="filepath", label="Upload Image")
image_output = gr.Image(type="filepath", label="Output Image")
clear_cache = gr.Button("Clear Cache(delete all cached results)")
run_batch_test_btn = gr.Button("Run Batch Test")
# 创建一个组件让用户选择validation set的位置
test_dir_input = gr.Textbox(placeholder="input test dir", label="Test Set Directory")

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
    run_batch_test_btn.render()
    test_dir_input.render()
    test_result=gr.Textbox(placeholder="waiting", label="Test Result")

    btn_pred.click(update_image, inputs=[gr.State("Predicted Image"), image_input], outputs=image_output)
    btn_label.click(update_image, inputs=[gr.State("Labeled Image"), image_input], outputs=image_output)
    btn_grayscale.click(update_image, inputs=[gr.State("Grayscale Image"), image_input], outputs=image_output)
    btn_binary.click(update_image, inputs=[gr.State("Binary Image"), image_input], outputs=image_output)
    btn_edge.click(update_image, inputs=[gr.State("Edge Image"), image_input], outputs=image_output)
    
    clear_cache.click(lambda: cache.clear())
    run_batch_test_btn.click(run_batch_test, inputs=[test_dir_input], outputs=[test_result])
# 启动Gradio应用
if __name__ == "__main__":
    # 创建临时输出目录
    if not os.path.exists("tmp_output"):
        os.makedirs("tmp_output")
    demo.launch()
