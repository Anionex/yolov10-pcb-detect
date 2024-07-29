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

        if not os.path.exists(label_path):
            # 删除没有标注的图片
            os.remove(img_path)
            print(f"删除无标注文件: {filename}")
            continue

        # 读取图像
        img = cv2.imread(img_path)

        # 备份原始图像
        origin_img_path = os.path.join(input_dir, "origin_" + filename)
        shutil.copyfile(img_path, origin_img_path)
        #将label文件也备份,并且删除原来的那个
        origin_label_path = os.path.join(labels_dir, "origin_" + os.path.splitext(filename)[0] + ".txt")
        shutil.copyfile(label_path, origin_label_path)
        os.remove(label_path)

        # 调整对比度和亮度
        adjusted_img = adjust_contrast_and_brightness(img)

        # 转换为灰度图像
        gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
        gray_img_path = os.path.join(input_dir, "gray_" + filename)
        gray_label_path = os.path.join(labels_dir, "gray_" + os.path.splitext(filename)[0] + ".txt")
        cv2.imwrite(gray_img_path, gray_img)
        shutil.copyfile(label_path, gray_label_path)


        # 生成红色图像及标注，调整颜色接近实际的红色电路板
        red_img = img.copy()
        
        red_img[:, :, 0] = np.clip(red_img[:, :, 0] * 0.6, 0, 255)  # 减弱蓝色通道，变为棕色调
        red_img[:, :, 1] = np.clip(red_img[:, :, 1] * 0.4, 0, 255)  # 减弱绿色通道
        red_img[:, :, 2] = np.clip(red_img[:, :, 2] * 1.256, 0, 255)  # 增强红色通道，使其更接近亮橙色
        
        red_img_path = os.path.join(input_dir, "red_" + filename)
        red_label_path = os.path.join(labels_dir, "red_" + os.path.splitext(filename)[0] + ".txt")
        cv2.imwrite(red_img_path, red_img)
        shutil.copyfile(label_path, red_label_path)

        # 打印进度
        print(f"处理进度: {idx + 1}/{total_files} ({((idx + 1) / total_files) * 100:.2f}%) - 已处理文件: {filename}")

# 设置目录路径
current_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(current_directory, "images")
labels_directory = os.path.join(current_directory, "labels")

# 执行备份和转换
backup_and_convert_images(images_directory, labels_directory)
