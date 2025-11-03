"""
合并三个PCB数据集的测试集为一个统一的测试数据集
包含：PKU数据集、初赛样例集、复赛样例集
"""

import os
import shutil
from pathlib import Path

# 源数据集路径（test目录）
SOURCE_DATASETS = [
    "./data/PCB_DATASET-PKU/yolo_dataset/test",
    "./data/PCB_瑕疵初赛样例集/yolo_dataset/test",
    "./data/PCB_瑕疵复赛样例集/yolo_dataset/test",
]

# 输出目录
OUTPUT_DIR = "./data/mix_pcb_test_dataset"
OUTPUT_TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# 统一的类别映射
CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}


def merge_datasets():
    """合并所有测试数据集"""
    print("="*60)
    print("合并PCB测试数据集")
    print("="*60)
    
    # 创建输出目录（test/images 和 test/labels）
    output_images = os.path.join(OUTPUT_TEST_DIR, "images")
    output_labels = os.path.join(OUTPUT_TEST_DIR, "labels")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    total_count = 0
    dataset_counts = {}
    
    # 遍历每个源数据集
    for dataset_path in SOURCE_DATASETS:
        if not os.path.exists(dataset_path):
            print(f"⚠️  跳过 {dataset_path} (目录不存在)")
            continue
        
        dataset_name = Path(dataset_path).parent.parent.name
        print(f"\n处理 {dataset_name}...")
        
        src_images = os.path.join(dataset_path, "images")
        src_labels = os.path.join(dataset_path, "labels")
        
        if not os.path.exists(src_images) or not os.path.exists(src_labels):
            print(f"  ⚠️  缺少images或labels目录，跳过")
            continue
        
        count = 0
        
        # 复制图像文件
        for img_file in os.listdir(src_images):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 检查是否有对应的标注文件
                base_name = os.path.splitext(img_file)[0]
                label_file = base_name + ".txt"
                
                src_img = os.path.join(src_images, img_file)
                src_label = os.path.join(src_labels, label_file)
                
                if not os.path.exists(src_label):
                    print(f"  ⚠️  {img_file} 没有对应的标注文件，跳过")
                    continue
                
                # 处理文件名冲突（不同数据集可能有相同文件名）
                # 添加数据集前缀
                prefix = get_dataset_prefix(dataset_name)
                new_img_name = f"{prefix}_{img_file}"
                new_label_name = f"{prefix}_{label_file}"
                
                dst_img = os.path.join(output_images, new_img_name)
                dst_label = os.path.join(output_labels, new_label_name)
                
                # 如果目标文件已存在，添加序号
                if os.path.exists(dst_img):
                    idx = 1
                    base = os.path.splitext(img_file)[0]
                    ext = os.path.splitext(img_file)[1]
                    while os.path.exists(os.path.join(output_images, f"{prefix}_{base}_{idx}{ext}")):
                        idx += 1
                    new_img_name = f"{prefix}_{base}_{idx}{ext}"
                    new_label_name = f"{prefix}_{base}_{idx}.txt"
                    dst_img = os.path.join(output_images, new_img_name)
                    dst_label = os.path.join(output_labels, new_label_name)
                
                # 复制文件
                shutil.copy(src_img, dst_img)
                shutil.copy(src_label, dst_label)
                
                count += 1
                total_count += 1
        
        dataset_counts[dataset_name] = count
        print(f"  ✓ 复制了 {count} 张图片")
    
    # 生成data.yaml
    generate_data_yaml()
    
    print("\n" + "="*60)
    print("✅ 合并完成！")
    print("="*60)
    print(f"总计: {total_count} 张图片")
    for dataset_name, count in dataset_counts.items():
        print(f"  - {dataset_name}: {count} 张")
    print(f"\n合并数据集路径: {os.path.abspath(OUTPUT_DIR)}")


def get_dataset_prefix(dataset_name):
    """根据数据集名称获取文件前缀"""
    if "PKU" in dataset_name:
        return "pku"
    elif "初赛" in dataset_name:
        return "preliminary"
    elif "复赛" in dataset_name:
        return "final"
    else:
        return "unknown"


def generate_data_yaml():
    """生成YOLO配置文件"""
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"# 合并的PCB瑕疵测试数据集\n")
        f.write(f"# 包含: PKU数据集 + 初赛样例集 + 复赛样例集\n\n")
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("test: test\n")
        f.write("\n")
        f.write("names:\n")
        for idx, name in CLASS_NAMES.items():
            f.write(f"  {idx}: {name}\n")
    
    print(f"\n✅ 已生成配置文件: {yaml_path}")


if __name__ == "__main__":
    merge_datasets()

