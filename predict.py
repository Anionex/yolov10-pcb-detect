import cv2
import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10("C:/Users/10051/Desktop/yolov10-test/yolov10/runs/detect/best_.pt")
image = cv2.imread(f'C:/Users/10051/Desktop/ydw/项目小组/26赛道华为智检/data/PCB_瑕疵初赛样例集/Open_circuit_Img/06_Open_circuit.bmp')
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)