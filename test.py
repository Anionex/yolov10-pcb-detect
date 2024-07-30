from ultralytics import YOLOv10

# Load a model
model = YOLOv10("weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(save=True, task="test")  # no arguments needed, dataset and settings remembered
