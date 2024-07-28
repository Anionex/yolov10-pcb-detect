import os
import cv2
import shutil
import numpy as np

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def adjust_contrast_and_brightness(image, contrast=1.5, brightness=50):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def backup_and_convert_images(input_dir, labels_dir):
    # 支持的图片格式
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # 获取文件列表
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    total_files = len(file_list)

    if total_files == 0:
        print("没有找到任何支持的图片格式。")
        return

    # 遍历input_dir目录中的所有图片文件
    for idx, filename in enumerate(file_list):
        img_path = os.path.join(input_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

        # 读取图像
        img = cv2.imread(img_path)


        # 调整对比度和亮度
        adjusted_img = adjust_contrast_and_brightness(img)

        # 转换为灰度图像
        gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
        gray_img_path = os.path.join(input_dir, "gray_" + filename)
        gray_label_path = os.path.join(labels_dir, "gray_" + os.path.splitext(filename)[0] + ".txt")
        cv2.imwrite(gray_img_path, gray_img)
        shutil.copyfile(label_path, gray_label_path)


        # 打印进度
        print(f"处理进度: {idx + 1}/{total_files} ({((idx + 1) / total_files) * 100:.2f}%) - 已处理文件: {filename}")

# 设置目录路径
current_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(current_directory, "images")
labels_directory = os.path.join(current_directory, "labels")

# 执行备份和转换
backup_and_convert_images(images_directory, labels_directory)
