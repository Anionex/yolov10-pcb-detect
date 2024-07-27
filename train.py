# import torch
# import torch_npu
# import torchvision
# import torchvision_npu
# from torch_npu.contrib import transfer_to_npu
from ultralytics import YOLOv10

# Load a model
model = YOLOv10("weights/yolov10s.pt")  # load a pretrained model (recommended for training)

# since we have early stop in the training, we can train the model for a large number of epochs
results = model.train(data="pcb-defect-dataset/data.yaml", epochs=10000, imgsz=640, workers=32, device=0, plots=True, batch=64)


