import os
from customize_service import yolov10_detection
import json, numpy as np
import cv2

image_name = "\
07_Spurious_copper_Img.bmp\
"
test_path = f"data/{image_name}"

anno_txt_path = f"anno/{image_name}.txt"
#获取yolo格式标注文件txt的内容，存在result中.格式为cls_id, x, y, w, h

#cls map 用来映射正确关系
# officail:
# 0 Mouse_bite 
# 1 Open_circuit 
# 2 Short 
# 3 Spur 
# 4 Spurious_copper 
cls_name = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]



result = []
with open(anno_txt_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        cls_id, x, y, w, h = line.strip().split(" ")
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        result['detection_boxes'].append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        result['detection_classes'].append(cls_name[int(cls_id)])


    
# 生成一张图片，将检测结果画在图片上。包括矩形框和类别名称
# Load the image
image = cv2.imread(test_path)

# 定义一些绘图参数
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 255, 0)
line_type = 2

# 遍历检测结果并绘制矩形框和类别名称
for i in range(len(result['detection_boxes'])):
    box = result['detection_boxes'][i]
    label = result['detection_classes'][i]
    
    ymin, xmin, ymax, xmax = box
    # 画矩形框
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
    
    # 添加标签和置信度
    label_text = f"{label}"
    cv2.putText(image, label_text, (int(xmin), int(ymin) - 10), font, font_scale, font_color, line_type)

# 保存生成的图片
output_path = f"anno_output/{image_name}"
cv2.imwrite(output_path, image)
print(f"Result image saved to {output_path}.")




