# YOLOv10 PCB瑕疵检测系统

## 简介
第十九届"挑战杯"揭榜挂帅·华为赛道（高校赛道）复赛Rank27/320，决赛国家一等奖方案（算法部分）。使用在COCO数据集上经过预训练的YOLOv10模型进行迁移学习，专门用于PCB板瑕疵检测。

## 分类
- 深度学习/计算机视觉
- 目标检测/瑕疵检测

## 检测类别
支持以下PCB瑕疵类型的检测：
- Mouse_bite（鼠咬）
- Spur（毛刺）
- Missing_hole（缺孔）
- Short（短路）
- Open_circuit（开路）
- Spurious_copper（多余铜）

## 环境要求
- Python 3.9
- CUDA 12.x（推荐）
- PyTorch 2.0+

## 快速启动

### 1. 创建Conda环境
```bash
conda create -n yolov10 python=3.9 -y
conda activate yolov10
```

### 2. 安装PyTorch（CUDA版本）
```bash
# 根据您的CUDA版本选择合适的命令（以CUDA 12.1为例）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. 安装项目依赖
```bash
# 安装所有必需依赖（包含huggingface-hub、opencv-python、gradio、supervision、pandas等）
pip install -r requirements.txt
```

### 4. 运行推理测试
```bash
# 使用项目中的YOLOv10实现进行测试
python test_service.py

# 或者使用基础预测脚本
python predict.py
```

### 5. 启动Web界面（可选）
```bash
# 启动Gradio可视化界面
python service_app.py

# 或使用官方界面
python app-origin.py
```

## 项目结构
```
├── data/                  # 临时测试数据存放文件夹
├── tmp_output/           # 推理结果输出目录
├── utils/                # 数据处理工具
├── weights/              # 模型权重文件
├── ultralytics/          # 项目内置YOLOv10实现
├── service_app.py        # Gradio应用（推荐，功能丰富）
├── app-origin.py         # 官方Gradio应用
├── predict.py            # 基础预测脚本
├── test_service.py       # 推理测试脚本
├── customize_service.py  # 自定义服务接口
├── config.py             # 配置文件（类别定义）
├── train.py              # 训练脚本
├── val.py                # 验证脚本
└── requirements.txt      # 项目依赖列表
```

## 使用说明

### 基础推理
1. 将待检测的PCB图片放入 `data/` 目录
2. 运行 `python test_service.py` 使用项目中的YOLOv10实现进行测试
3. 检测结果将保存在 `tmp_output/` 目录和 `result.json` 文件中

### Web界面使用
1. 运行 `python service_app.py` 启动服务
2. 在浏览器中打开显示的URL（通常是 http://localhost:7860）
3. 上传PCB图片进行实时检测和可视化

### 自定义模型
- 替换 `yolov8n.pt` 为您训练的PCB瑕疵检测模型
- 在 `config.py` 中修改类别定义
- 调整推理参数以适应您的应用场景

## 性能基准
- **推理速度**: ~8-24ms/区域（608x608，RTX 4060，使用项目内置YOLOv10）
- **支持格式**: JPG, JPEG, PNG, BMP
- **推荐分辨率**: 608x608（项目默认设置）
- **检测机制**: 滑动窗口大图检测

## 数据集处理方法

### PKU数据集转换
将PKU数据集（VOC格式）转换为YOLO格式：
```bash
cd data/PCB_DATASET-PKU
python ../../utils/pku_dataset_convert_to_yolo.py
```
数据均放在test集中，共693张。

### 初赛样例集转换
将初赛样例集整合为标准YOLO格式（含类别ID统一映射）：
```bash
python utils/preliminary_dataset_convert_to_yolo.py
```
转换后数据集位置：`data/PCB_瑕疵初赛样例集/yolo_dataset/`
数据均放在test集中，共48张。

### 复赛样例集转换
将复赛样例集整合为标准YOLO格式（含类别ID统一映射）：
```bash
python utils/final_dataset_convert_to_yolo.py
```
转换后数据集位置：`data/PCB_瑕疵复赛样例集/yolo_dataset/`
数据均放在test集中，共20张。

### 合并所有测试数据集
将三个数据集的测试集合并为一个统一的测试数据集：
```bash
python utils/merge_test_datasets.py
```
合并后数据集位置：`data/mix_pcb_test_dataset/`
- 包含761张测试图片（PKU: 693张, 初赛: 48张, 复赛: 20张）
- 统一的images和labels目录结构
- 自动处理文件名冲突（添加数据集前缀）

### 类别映射说明
所有转换脚本统一使用以下类别映射：
- 0: mouse_bite（鼠咬）
- 1: spur（毛刺）
- 2: missing_hole（缺孔）
- 3: short（短路）
- 4: open_circuit（开路）
- 5: spurious_copper（多余铜）

## Benchmark评估

项目提供完整的benchmark评估工具，用于系统化测试模型性能：

### 快速开始
```bash
# 安装额外依赖
pip install ensemble-boxes tabulate

# 运行benchmark（交互式）
python run_benchmark.py

# 或直接运行
python benchmark.py --dataset_path datasets/test --model_path weights/best.pt
```

### 主要功能
- ✅ **标准评估指标**: mAP, Precision, Recall, F1-Score
- ✅ **速度分析**: FPS, 推理时间统计
- ✅ **类别级别评估**: 每个瑕疵类型的详细指标
- ✅ **批量实验**: 自动测试多种配置组合
- ✅ **结果对比**: 可视化对比不同配置
- ✅ **导出功能**: JSON和CSV格式结果

### 相关文件
- `benchmark.py` - 核心评估模块
- `run_benchmark.py` - 便捷运行脚本（交互式）
- `benchmark_config.yaml` - 配置文件（定义实验）
- `example_benchmark.py` - 使用示例
- `BENCHMARK_GUIDE.md` - 详细使用指南

详细使用方法请查看 [BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md)

## Test Time优化技巧

除了滑动窗口，还支持以下优化技术：

1. **Test Time Augmentation (TTA)** - 多角度推理融合
2. **Weighted Boxes Fusion (WBF)** - 替代NMS的框融合
3. **Multi-Scale Testing** - 多尺度检测
4. **Adaptive Sliding Window** - 自适应窗口策略
5. **Confidence Calibration** - 置信度校准
6. **Model Ensemble** - 多模型集成

详见benchmark配置文件中的实验设置。

## 开发状态
- [✔] 修复NMS和红色PCB瑕疵检测
- [✔] 更新置信度和滑动方法
- [✔] Conda环境配置和依赖管理
- [✔] 推理功能测试和验证
- [✔] 跨平台兼容性支持
- [✔] Benchmark评估系统
- [✔] Test Time优化方案

