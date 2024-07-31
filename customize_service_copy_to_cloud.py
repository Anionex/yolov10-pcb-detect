import os
import glob
import shutil
import numpy as np
from ultralytics import YOLOv10
import torch
import cv2
import torchvision.ops as ops
from model_service.pytorch_model_service import PTServingBaseService

# class yolov10_detection():
class yolov10_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('model_name:', model_name)
        print('model_path:', model_path)
                
        self.model = YOLOv10(model_path)
        self.capture = "test.png"
        # 此处跳到608，以适应两个数据集的图片大小
        # 训练时，也应该调到608(32倍数)
        self.window_size = 608 # 滑动窗口的大小
        self.step_size = 320   # 滑动窗口的步长
        self.predict_conf = 0.4 # 预测准确阈值
        self.nms_threshold = 0.1  # NMS 阈值

    def _preprocess(self, data):
        for _, v in data.items():
            for _, file_content in v.items():
                with open(self.capture, 'wb') as f:
                    file_content_bytes = file_content.read()
                    f.write(file_content_bytes)
        return "ok"
    

    def _slide_window(self, image, window_size, step_size):
        
        height, width = image.shape[:2]  # For grayscale, use image.shape
        
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                # print(f"detect area left top: ({x}, {y})")
                # Ensure the window is properly cropped at the image edges
                crop_x = min(x, width - window_size)
                crop_y = min(y, height - window_size)
                cropped_image = image[crop_y:crop_y + window_size, crop_x:crop_x + window_size]
                
                # 判断本窗口图像有没有包含1184，1023
                # if crop_x <= 1184 and crop_x + window_size >= 1184 and crop_y <= 1023 and crop_y + window_size >= 1023:
                #     print(f"window ({crop_x}, {crop_y}) contains (1184, 1023)")
                #     # 窗口左上角打上标记，就是这个图片
                #     # cv2.circle(cropped_image, (1184 - crop_x, 1023 - crop_y), 10, (0, 255, 0), 2)
                    
                
                # 保存窗口图片到tmp_output/
                # cv2.imwrite(f"tmp_output/windows/{crop_x}_{crop_y}.png", cropped_image)
                
                yield (crop_x, crop_y, cropped_image)

                


    def _inference(self, data):
        image = cv2.imread(self.capture) # imread后的通道为BGR
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转换为RGB
        # image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # 转换为RGB
        pred_results = []
        for (x, y, window) in self._slide_window(image, self.window_size, self.step_size):
            
            window_image = window
            
            pred_result = self.model(window_image, 
                                     conf=self.predict_conf, 
                                    #  augment=True # 感觉对pcb检测没什么用，有时候有副作用，很少有正向作用
                                     )
            for result in pred_result:
                
                # 将检测到的结果位置映射回原图
                result_cpu = result.cpu()  # 转换为 CPU 张量
                result_clone = result_cpu.boxes.xyxy.clone()  # 克隆 boxes 的张量
                result_clone[:, [0, 2]] += x
                result_clone[:, [1, 3]] += y
                print(f"result_clone: {result_clone}")
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
            # print("clses:", classes)
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

        # print('result:', result)

        return result
