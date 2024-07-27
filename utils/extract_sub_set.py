import os
import random
import shutil

def select_samples(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, num_samples):
    # 确保目标目录存在
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    
    # 获取所有图片文件名
    img_files = os.listdir(src_img_dir)
    # 过滤出图像文件（例如，只选取 .jpg 文件）
    img_files = [f for f in img_files if f.endswith('.jpg')]
    # 随机选择 num_samples 个文件
    selected_files = random.sample(img_files, num_samples)
    
    for file_name in selected_files:
        src_img_path = os.path.join(src_img_dir, file_name)
        dst_img_path = os.path.join(dst_img_dir, file_name)
        shutil.copy(src_img_path, dst_img_path)
        
        # 构造对应的标签文件名
        label_name = file_name.replace('.jpg', '.txt')
        src_label_path = os.path.join(src_label_dir, label_name)
        dst_label_path = os.path.join(dst_label_dir, label_name)
        
        if os.path.exists(src_label_path):
            # 复制标签文件
            shutil.copy(src_label_path, dst_label_path)
        else:
            # 尝试从增强后的文件中复制标签文件
            enhanced_label_name = "l_" + label_name
            enhanced_label_path = os.path.join(src_label_dir, enhanced_label_name)
            if os.path.exists(enhanced_label_path):
                shutil.copy(enhanced_label_path, dst_label_path)
                print(f"Copied label from {enhanced_label_path} to {dst_label_path}")
            else:
                print(f"Warning: Label file for {file_name} not found and no enhanced label available. Deleting image.")
                os.remove(dst_img_path)

if __name__ == "__main__":
    src_img_dir = 'train/images'
    src_label_dir = 'train/labels'
    dst_img_dir = 'train_subset/images'
    dst_label_dir = 'train_subset/labels'
    num_samples = 256
    
    select_samples(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, num_samples)
    print(f"Selected {num_samples} samples and copied to {dst_img_dir} and {dst_label_dir}")
