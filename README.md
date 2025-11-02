# YOLOv10 PCB瑕疵检测系统

## 简介
第十九届"挑战杯"揭榜挂帅·华为赛道（高校赛道）复赛Rank35/320，决赛国家一等奖方案（算法部分）。使用在COCO数据集上经过预训练的YOLOv10模型进行迁移学习，专门用于PCB板瑕疵检测。

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

## 开发状态
- [✔] 修复NMS和红色PCB瑕疵检测
- [✔] 更新置信度和滑动方法
- [✔] Conda环境配置和依赖管理
- [✔] 推理功能测试和验证
- [✔] 跨平台兼容性支持

