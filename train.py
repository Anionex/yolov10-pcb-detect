from ultralytics import YOLOv10

# Load a model
model = YOLOv10("weights/yolov10s.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="pcb-defect-dataset/data.yaml", epochs=100, imgsz=640, workers=32, device=0, plots=True, batch=240)

print(result)
