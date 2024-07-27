import os
from customize_service import yolov10_detection
import json, numpy as np
import cv2

image_name = "\
02_Short_Img.bmp\
"
test_path = f"data/{image_name}"
service = yolov10_detection(model_name="yolov10", model_path="weights/best-A800.pt")

post_data = {"input_txt": {os.path.basename(test_path): open(test_path, "rb")}}

service._preprocess(post_data)
data = service._inference('ok')
result = service._postprocess(data)

with open("result.json", "w") as f:
    # 打印格式化之后的json
    
    # 调用 json.dumps 时提供一个 default 函数，该函数能够将不可序列化的对象转换为可序列化的类型。例如，可以将 float32 实例转换为标准的 Python float，这样就可以被 JSON 序列化了。
    
    json.dump(result, f, indent=4)
    

    print("result.json saved.")
    
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
    score = result['detection_scores'][i]
    
    ymin, xmin, ymax, xmax = box
    # 画矩形框
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
    
    # 添加标签和置信度
    label_text = f"{label}: {score:.2f}"
    cv2.putText(image, label_text, (int(xmin), int(ymin) - 10), font, font_scale, font_color, line_type)

# 保存生成的图片
output_path = f"tmp_output/{image_name}"
cv2.imwrite(output_path, image)
print(f"Result image saved to {output_path}.")

#打开图片
current_dir = os.path.dirname(os.path.abspath(__file__))
os.startfile(os.path.join(current_dir, output_path))



