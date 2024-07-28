import comet_ml
import torch
# import torch_npu
# import torchvision
# import torchvision_npu
# from torch_npu.contrib import transfer_to_npu
from ultralytics import YOLOv10

import os
os.environ["COMET_API_KEY"] = "Imas5SnynejXZLDsdttseSZhr"
if __name__ == '__main__':
    

    comet_ml.init(project_name="yolov10-pcb-defect-detection")
    
    torch.cuda.empty_cache()
    # Load a model
    model = YOLOv10("weights/yolov10s.pt")  # load a pretrained model (recommended for training)

    # since we have early stop in the training, we can train the model for a large number of epochs
    results = model.train(project="yolov10-pcb-defect-detection", 
                          data="datasets/data.yaml", 
                          epochs=10000, 
                          imgsz=640, 
                          device=0, 
                          plots=True, 
                          batch=16,
                          degrees=180,
                          rect=True,
                          )


