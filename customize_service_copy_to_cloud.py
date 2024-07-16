import os
import glob
import shutil
import numpy as np
from PIL import Image
from ultralytics import YOLOv10
import torch
from model_service.pytorch_model_service import PTServingBaseService

# class yolov10_detection():
class yolov10_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('model_name:', model_name)
        print('model_path:', model_path)
                
        self.model = YOLOv10(model_path)
        self.capture = "test.png"
        self.window_size = 640  # 滑动窗口的大小
        self.step_size = 320  # 滑动窗口的步长
        self.nms_threshold = 0.5  # NMS 阈值

    def _preprocess(self, data):
        for _, v in data.items():
            for _, file_content in v.items():
                with open(self.capture, 'wb') as f:
                    file_content_bytes = file_content.read()
                    f.write(file_content_bytes)
        return "ok"
    
    def _slide_window(self, image, window_size, step_size):
        width, height = image.size
        
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                # 确保窗口在图像边缘处适当裁剪
                crop_x = min(x, width - window_size)
                crop_y = min(y, height - window_size)
                yield (crop_x, crop_y, image.crop((crop_x, crop_y, crop_x + window_size, crop_y + window_size)))
                
                


    def _inference(self, data):
        image = Image.open(self.capture)
        pred_results = []

        for (x, y, window) in self._slide_window(image, self.window_size, self.step_size):
            window_image = window.convert('RGB')
            pred_result = self.model(source=window_image, conf=0.25)
            for result in pred_result:
                # 将检测到的结果位置映射回原图
                result_cpu = result.cpu()  # 转换为 CPU 张量
                result_clone = result_cpu.boxes.xyxy.clone()  # 克隆 boxes 的张量
                result_clone[:, [0, 2]] += x
                result_clone[:, [1, 3]] += y
                
                # 直接更新 result_cpu.boxes.xyxy 的值
                result_cpu.boxes._xyxy = result_clone
                pred_results.append(result_cpu)
    
        return pred_results

    def _non_max_suppression(self, boxes, scores, threshold):
        keep = []
        idxs = scores.argsort()[::-1]

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            ious = self._iou(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < threshold]
        
        return keep

    def _iou(self, box1, boxes):
        # 计算 box1 和 boxes 之间的交并比 (IoU)
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - inter_area
        return inter_area / union_area

    def _postprocess(self, data):
        result = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []
        
        # 硬编码的类别名称列表
        class_names = [
            "Mouse_bite",
            "Spur",
            "Missing_hole",
            "Short",
            "Open_circuit",
            "Spurious_copper"
        ]

        all_boxes = []
        all_scores = []
        all_classes = []

        for res in data:
            boxes = res.boxes._xyxy.cpu().numpy()  # 获取 bounding boxes 并转换为 numpy 数组
            scores = res.boxes.conf.cpu().numpy()  # 获取置信度分数并转换为 numpy 数组
            classes = res.boxes.cls.cpu().numpy()  # 获取类别索引并转换为 numpy 数组

            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_classes.extend(classes)

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_classes = np.array(all_classes)

        keep = self._non_max_suppression(all_boxes, all_scores, self.nms_threshold)

        for i in keep:
            box = all_boxes[i]
            score = float(all_scores[i])
            cls = int(all_classes[i])

            xmin, ymin, xmax, ymax = map(float, box)
            detection_boxes.append([ymin, xmin, ymax, xmax])
            detection_scores.append(score)
            detection_classes.append(class_names[cls])
                
        result['detection_classes'] = detection_classes
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores
        
        print('result:', result)   
        
        return result
