# import comet_ml
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from ultralytics import YOLOv10
import os

os.environ["COMET_API_KEY"] = "Imas5SnynejXZLDsdttseSZhr"
if __name__ == '__main__':
    

    # comet_ml.init(project_name="yolov10-pcb-defect-detection")
    # Load a model
    torch.cuda.empty_cache()
    model = YOLOv10("weights/yolov10s.pt")  # load a pretrained model (recommended for training)
    
    # 用于resume
#     model = YOLOv10("yolov10-pcb-defect-detection/train19/weights/last.pt")
    # Train the model with 2 GPUs
    results = model.train(
#         resume=True,
        data="datasets/data.yaml", 
        epochs=200, # 300个epoch是一个合理的初始值，但由于我的数据集
        imgsz=608, # imgsz应该尽量贴近训练集图片的大小，但是要是32的倍数
        plots=True, batch=96,
        # close_mosaic=500,
        project="yolov10-pcb-defect-detection",
        # degrees=180,
        auto_augment="autoaugment",
        cache=True, # 缓存数据集，加快训练速度
        hsv_h=0.02, # 调整图像的色调，对提高模型对不同颜色的物体的识别能力有帮助
        translate=0.2, # 平移图像，帮助模型识别边角的物体
        flipud=0.5, # 以指定概率上下翻转图像
        # bgr=0.5, # 以指定概率随机改变图像的颜色通道顺序，提高模型对不同颜色的物体的识别能力
        close_mosaic=20, # 最后稳定训练
        scale=0.3,
        # device=[0, 1],
#         device=[0,1,2,3],
        device=0,
    )

