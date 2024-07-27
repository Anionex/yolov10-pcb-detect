import os
import json, numpy as np
import cv2
def anno_single(input_dir = "train", image_base_path = "01_spur_11", output_dir="anno_output"): # 无后缀
    # 更改当前工作目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_path = f"{input_dir}/{image_base_path}.jpg"
    anno_txt_path = f"{input_dir}/{image_base_path}.txt"
    #获取yolo格式标注文件txt的内容，存在result中.格式为cls_id, x, y, w, h

    #cls map 用来映射正确关系
    # officail:
    # 0 Mouse_bite 
    # 1 Open_circuit 
    # 2 Short 
    # 3 Spur 
    # 4 Spurious_copper 
    cls_name = ["Mouse_bite", "Spur", "Short", "Open_circuit", "Spurious_copper"]



    result = {
        'detection_boxes': [],
        'detection_classes': []
    }
    with open(anno_txt_path, "rt") as f:
        lines = f.readlines()
        for line in lines:
            cls_id, x, y, w, h = line.strip().split(" ")
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            result['detection_boxes'].append([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0])
            result['detection_classes'].append(cls_name[int(cls_id)])


        
    # 生成一张图片，将检测结果画在图片上。包括矩形框和类别名称
    # Load the image
    image = cv2.imread(image_path)
    print(f"Image path: {image_path}")
    # 定义一些绘图参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)
    line_type = 2

    # 获取原始图片的宽高
    IMAGE_W = image.shape[1]
    IMAGE_H = image.shape[0]

    output_path = f"{output_dir}/{image_base_path}-orgin.jpg"
    cv2.imwrite(output_path, image)

    # 遍历检测结果并绘制矩形框和类别名称
    for i in range(len(result['detection_boxes'])):
        box = result['detection_boxes'][i]
        label = result['detection_classes'][i]
        
        xmin, ymin, xmax, ymax = box
        ymin *= IMAGE_H
        xmin *= IMAGE_W
        ymax *= IMAGE_H
        xmax *= IMAGE_W
        # 画矩形框
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
        
        # 添加标签和置信度
        label_text = f"{label}"
        cv2.putText(image, label_text, (int(xmin), int(ymin) - 10), font, font_scale, font_color, line_type)

    # 保存生成的图片
    output_path = f"{output_dir}/{image_base_path}.jpg"
    cv2.imwrite(output_path, image)
    print(f"Result image saved to {output_path}.")


anno_single()


