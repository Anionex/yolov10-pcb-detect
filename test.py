from yolov10_sliding_window import YOLOv10SlidingWindow

# 加载模型（自动使用滑动窗口验证器）
model = YOLOv10SlidingWindow("weights/best.pt")
metrics = model.val(save=True, task="test")
