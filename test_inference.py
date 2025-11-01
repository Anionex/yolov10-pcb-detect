import cv2
import supervision as sv
from ultralytics import YOLO
import os

def test_yolov10_inference():
    """测试YOLOv10推理功能"""

    # 检查CUDA是否可用
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")

    # 使用项目中的模型文件路径
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        # 尝试下载YOLOv10n模型
        print("正在下载YOLOv10n模型...")
        model = YOLO('yolov10n.pt')
    else:
        print(f"使用现有模型: {model_path}")
        model = YOLO(model_path)

    # 创建测试图像目录（如果不存在）
    test_dir = "data"
    os.makedirs(test_dir, exist_ok=True)

    # 检查是否有测试图像
    all_files = os.listdir(test_dir)
    print(f"在{test_dir}目录中所有文件: {all_files}")

    test_images = []
    for file in all_files:
        print(f"检查文件: {file}, 扩展名: {file.lower().split('.')[-1] if '.' in file else 'no_ext'}")
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_images.append(file)

    print(f"在{test_dir}目录中找到的图像文件: {test_images}")

    if not test_images:
        print(f"在{test_dir}目录中未找到测试图像")
        print("请将测试图像放入data目录中")
        return False

    # 对每个测试图像进行推理
    for image_file in test_images:
        image_path = os.path.join(test_dir, image_file)
        print(f"\n正在处理图像: {image_path}")

        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 进行推理
            results = model(image)[0]
            detections = sv.Detections.from_ultralytics(results)

            print(f"检测到 {len(detections)} 个目标")

            # 可视化结果
            try:
                # 尝试新版本API
                bounding_box_annotator = sv.BoundingBoxAnnotator()
                label_annotator = sv.LabelAnnotator()

                annotated_image = bounding_box_annotator.annotate(
                    scene=image, detections=detections)
                annotated_image = label_annotator.annotate(
                    scene=annotated_image, detections=detections)
            except AttributeError:
                # 使用旧版本API或简化版本
                print("使用简化的可视化方法")
                annotated_image = image.copy()

                # 手动绘制边界框
                for i in range(len(detections)):
                    if len(detections) > 0 and len(detections.xyxy) > i:
                        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                        confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 0.0
                        class_id = int(detections.class_id[i]) if hasattr(detections, 'class_id') else 0

                        # 绘制矩形
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 添加标签
                        label = f"Class {class_id}: {confidence:.2f}"
                        cv2.putText(annotated_image, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 保存结果
            output_dir = "tmp_output"
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"result_{image_file}")
            cv2.imwrite(output_path, annotated_image)
            print(f"结果已保存到: {output_path}")

            # 显示检测结果
            if len(detections) > 0:
                print("检测结果:")
                for i, detection in enumerate(detections):
                    class_id = int(detection[2]) if len(detection) > 2 else 0
                    confidence = float(detection[1]) if len(detection) > 1 else 0.0
                    print(f"  目标 {i+1}: 类别 {class_id}, 置信度 {confidence:.2f}")

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            continue

    print("\n推理测试完成!")
    return True

if __name__ == "__main__":
    test_yolov10_inference()