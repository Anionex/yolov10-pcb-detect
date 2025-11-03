"""
将PCB瑕疵复赛样例集转换为YOLO格式数据集
复赛数据已经是YOLO格式，只需要整合文件结构并统一类别映射
"""

import os
import shutil
import random
from pathlib import Path

# 配置
SRC_DIR = "./data/PCB_瑕疵复赛样例集"
OUTPUT_DIR = "./data/PCB_瑕疵复赛样例集/yolo_dataset"

# 数据集分割比例（样例集仅用于测试，全部放test）
VAL_RATIO = 0
TEST_RATIO = 1.0

# 复赛类别映射（复赛ID -> PKU统一ID）
# 复赛: 0-Mouse_bite, 1-Open_circuit, 2-Short, 3-Spur, 4-Spurious_copper
# PKU: 0-mouse_bite, 1-spur, 2-missing_hole, 3-short, 4-open_circuit, 5-spurious_copper
CLASS_MAPPING = {
    0: 0,  # Mouse_bite -> mouse_bite
    1: 4,  # Open_circuit -> open_circuit
    2: 3,  # Short -> short
    3: 1,  # Spur -> spur
    4: 5,  # Spurious_copper -> spurious_copper
}

# 类别名称
CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}


def get_subset_name():
    """随机选择子数据集"""
    r = random.random()
    if r < VAL_RATIO:
        return "val"
    elif r < VAL_RATIO + TEST_RATIO:
        return "test"
    else:
        return "train"


def convert_annotation(src_file, dst_file):
    """转换标注文件，重新映射类别ID"""
    with open(src_file, 'r') as f_in:
        with open(dst_file, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) == 5:
                    old_class = int(parts[0])
                    new_class = CLASS_MAPPING.get(old_class, old_class)
                    f_out.write(f"{new_class} {' '.join(parts[1:])}\n")


def convert_dataset():
    """转换数据集"""
    print("="*60)
    print("转换PCB瑕疵复赛样例集到YOLO格式")
    print("="*60)
    
    # 源目录
    src_images = os.path.join(SRC_DIR, "images")
    src_labels = os.path.join(SRC_DIR, "Annotations")
    
    if not os.path.exists(src_images):
        print(f"❌ 错误: 找不到图像目录 {src_images}")
        return
    
    if not os.path.exists(src_labels):
        print(f"❌ 错误: 找不到标注目录 {src_labels}")
        return
    
    # 创建输出目录
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, subset, "labels"), exist_ok=True)
    
    total_count = 0
    subset_counts = {"train": 0, "val": 0, "test": 0}
    
    print("\n开始转换...")
    
    # 遍历所有图像
    for img_file in os.listdir(src_images):
        if not img_file.endswith('.bmp'):
            continue
        
        # 对应的标注文件
        base_name = os.path.splitext(img_file)[0]
        txt_file = base_name + ".txt"
        
        src_img = os.path.join(src_images, img_file)
        src_txt = os.path.join(src_labels, txt_file)
        
        if not os.path.exists(src_txt):
            print(f"⚠️  {img_file} 没有对应的标注文件，跳过")
            continue
        
        # 随机选择子集
        subset = get_subset_name()
        
        # 目标路径
        dst_img = os.path.join(OUTPUT_DIR, subset, "images", img_file)
        dst_txt = os.path.join(OUTPUT_DIR, subset, "labels", txt_file)
        
        # 复制图像
        shutil.copy(src_img, dst_img)
        
        # 转换并复制标注
        convert_annotation(src_txt, dst_txt)
        
        total_count += 1
        subset_counts[subset] += 1
        
        if total_count % 5 == 0:
            print(f"  已处理 {total_count} 张图片...")
    
    # 生成data.yaml
    generate_data_yaml()
    
    print("\n" + "="*60)
    print("✅ 转换完成！")
    print("="*60)
    print(f"总计: {total_count} 张图片")
    print(f"训练集: {subset_counts['train']} 张")
    print(f"验证集: {subset_counts['val']} 张")
    print(f"测试集: {subset_counts['test']} 张")
    print(f"\n数据集路径: {os.path.abspath(OUTPUT_DIR)}")


def generate_data_yaml():
    """生成YOLO配置文件"""
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("train: train\n")
        f.write("val: val\n")
        f.write("test: test\n")
        f.write("\n")
        f.write("names:\n")
        for idx, name in CLASS_NAMES.items():
            f.write(f"  {idx}: {name}\n")
    
    print(f"\n✅ 已生成配置文件: {yaml_path}")


if __name__ == "__main__":
    convert_dataset()

