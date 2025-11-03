import os
from customize_service import yolov10_detection
import json
from collections import Counter
import cv2

# 配置参数
image_dir = r"D:\ydw\项目小组\26赛道华为智检\data\PCB_瑕疵复赛样例集\images"
model_name = "best.pt"
output_dir = "batch_detection_results"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs("tmp_output/windows", exist_ok=True)

# 初始化检测服务
print("正在加载模型...")
service = yolov10_detection(model_name="yolov10", model_path=f"weights/{model_name}")
print("模型加载完成！")

# 统计数据
all_defects = []  # 存储所有缺陷
image_results = {}  # 存储每张图片的检测结果

# 批量处理图片
for i in range(1, 21):  # 001到020
    image_name = f"{i:03d}.bmp"
    image_path = os.path.join(image_dir, image_name)
    
    if not os.path.exists(image_path):
        print(f"警告：图片 {image_path} 不存在，跳过...")
        continue
    
    print(f"\n{'='*60}")
    print(f"正在检测第 {i}/20 张图片: {image_name}")
    print(f"{'='*60}")
    
    try:
        # 预处理
        post_data = {"input_txt": {image_name: open(image_path, "rb")}}
        service._preprocess(post_data)
        
        # 推理
        data = service._inference('ok')
        
        # 后处理
        result = service._postprocess(data)
        
        # 统计本张图片的缺陷
        defect_count = len(result['detection_classes'])
        defect_types = Counter(result['detection_classes'])
        
        # 保存本张图片的结果
        image_results[image_name] = {
            "defect_count": defect_count,
            "defect_types": dict(defect_types),
            "details": result
        }
        
        # 添加到总统计
        all_defects.extend(result['detection_classes'])
        
        # 打印本张图片的统计
        print(f"\n图片 {image_name} 检测结果:")
        print(f"  - 缺陷总数: {defect_count}")
        if defect_count > 0:
            print(f"  - 缺陷类型分布:")
            for defect_type, count in defect_types.items():
                print(f"    * {defect_type}: {count} 个")
        else:
            print(f"  - 未检测到缺陷")
        
        # 保存详细结果到JSON
        result_json_path = os.path.join(output_dir, f"{image_name.replace('.bmp', '')}_result.json")
        with open(result_json_path, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        # 生成标注图片
        image = cv2.imread(image_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)
        line_type = 2
        
        # 绘制检测框
        for j in range(len(result['detection_boxes'])):
            box = result['detection_boxes'][j]
            label = result['detection_classes'][j]
            score = result['detection_scores'][j]
            
            ymin, xmin, ymax, xmax = box
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(image, label_text, (int(xmin), int(ymin) - 10), font, font_scale, font_color, line_type)
        
        # 保存标注图片
        output_image_path = os.path.join(output_dir, f"{image_name.replace('.bmp', '')}_detected.jpg")
        cv2.imwrite(output_image_path, image)
        
    except Exception as e:
        print(f"错误：处理图片 {image_name} 时出错: {str(e)}")
        continue

# 生成总体统计报告
print(f"\n{'='*60}")
print("总体统计报告")
print(f"{'='*60}")

total_defects = len(all_defects)
defect_counter = Counter(all_defects)

print(f"\n共检测了 20 张图片")
print(f"缺陷总数: {total_defects} 个")
print(f"\n各类缺陷统计:")

# 定义缺陷类别（确保顺序一致）
all_defect_types = [
    "Mouse_bite",
    "Spur",
    "Missing_hole",
    "Short",
    "Open_circuit",
    "Spurious_copper"
]

for defect_type in all_defect_types:
    count = defect_counter.get(defect_type, 0)
    percentage = (count / total_defects * 100) if total_defects > 0 else 0
    print(f"  - {defect_type}: {count} 个 ({percentage:.2f}%)")

# 保存总体统计报告
summary_report = {
    "total_images": 20,
    "total_defects": total_defects,
    "defect_distribution": dict(defect_counter),
    "image_results": image_results
}

summary_path = os.path.join(output_dir, "summary_report.json")
with open(summary_path, "w", encoding='utf-8') as f:
    json.dump(summary_report, f, indent=4, ensure_ascii=False)

print(f"\n所有结果已保存到: {output_dir}")
print(f"总体报告保存到: {summary_path}")

# 生成可读的文本报告
text_report_path = os.path.join(output_dir, "summary_report.txt")
with open(text_report_path, "w", encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("PCB缺陷检测统计报告\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"检测图片数量: 20 张\n")
    f.write(f"缺陷总数: {total_defects} 个\n\n")
    
    f.write("各类缺陷统计:\n")
    f.write("-"*60 + "\n")
    for defect_type in all_defect_types:
        count = defect_counter.get(defect_type, 0)
        percentage = (count / total_defects * 100) if total_defects > 0 else 0
        f.write(f"{defect_type:20s}: {count:4d} 个 ({percentage:6.2f}%)\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("各图片检测详情:\n")
    f.write("="*60 + "\n\n")
    
    for img_name in sorted(image_results.keys()):
        result_data = image_results[img_name]
        f.write(f"{img_name}:\n")
        f.write(f"  缺陷总数: {result_data['defect_count']} 个\n")
        if result_data['defect_count'] > 0:
            f.write(f"  缺陷类型:\n")
            for defect_type, count in result_data['defect_types'].items():
                f.write(f"    - {defect_type}: {count} 个\n")
        else:
            f.write(f"  未检测到缺陷\n")
        f.write("\n")

print(f"文本报告保存到: {text_report_path}")
print("\n检测完成！")

