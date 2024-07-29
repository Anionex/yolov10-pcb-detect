import os
import glob
import shutil
import numpy as np
from PIL import Image
from ultralytics import YOLOv10
import torch
import torchvision.ops as ops
from model_service.pytorch_model_service import PTServingBaseService

# class yolov10_detection():
class yolov10_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('model_name:', model_name)
        print('model_path:', model_path)
                
        self.model = YOLOv10(model_path)
        self.capture = "test.png"
        self.window_size = 640  # 滑动窗口的大小
        self.step_size = 480  # 滑动窗口的步长
        self.predict_conf = 0.6 # 预测准确阈值
        self.nms_threshold = 0.1  # NMS 阈值

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
        # .convert('L')
        for (x, y, window) in self._slide_window(image, self.window_size, self.step_size):
            window_image = window
            pred_result = self.model(source=window_image, conf=self.predict_conf)
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


    def _postprocess(self, data):
        result = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []

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
            boxes = res.boxes._xyxy.cpu()  # 获取 bounding boxes 并转换为 CPU 张量
            scores = res.boxes.conf.cpu()  # 获取置信度分数并转换为 CPU 张量
            classes = res.boxes.cls.cpu()  # 获取类别索引并转换为 CPU 张量
            print("clses:", classes)
            #如果不是missing_hole

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        all_boxes = torch.cat(all_boxes)
        all_scores = torch.cat(all_scores)
        all_classes = torch.cat(all_classes)

        keep = ops.nms(all_boxes, all_scores, self.nms_threshold)
        keep = [i for i in keep if all_classes[i] != 2]
        for i in keep:
            box = all_boxes[i].numpy()
            score = float(all_scores[i].numpy())
            cls = int(all_classes[i].numpy())

            xmin, ymin, xmax, ymax = map(float, box)
            detection_boxes.append([ymin, xmin, ymax, xmax])
            detection_scores.append(score)
            detection_classes.append(class_names[cls])

        result['detection_classes'] = detection_classes
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores

        print('result:', result)

        return result
