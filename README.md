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
- Python 3.9+
- CUDA 12.x（推荐）
- PyTorch 2.0+
- Windows 11 / Linux

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
# 安装基础依赖
pip install pandas huggingface_hub

# 安装计算机视觉库
pip install ultralytics supervision opencv-python

# 安装其他依赖
pip install -r requirements.txt
```

### 4. 运行推理测试
```bash
# 测试基础推理功能
python test_inference.py

# 或者使用现有的预测脚本
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
├── ultralytics/          # YOLOv10源码
├── service_app.py        # Gradio应用（推荐，功能丰富）
├── app-origin.py         # 官方Gradio应用
├── predict.py            # 基础预测脚本
├── test_inference.py     # 推理功能测试脚本
├── customize_service.py  # 自定义服务接口
├── config.py             # 配置文件（类别定义）
├── train.py              # 训练脚本
├── val.py                # 验证脚本
└── requirements.txt      # 项目依赖列表
```

## 使用说明

### 基础推理
1. 将待检测的PCB图片放入 `data/` 目录
2. 运行 `python test_inference.py` 进行测试
3. 检测结果将保存在 `tmp_output/` 目录

### Web界面使用
1. 运行 `python service_app.py` 启动服务
2. 在浏览器中打开显示的URL（通常是 http://localhost:7860）
3. 上传PCB图片进行实时检测和可视化

### 自定义模型
- 替换 `yolov8n.pt` 为您训练的PCB瑕疵检测模型
- 在 `config.py` 中修改类别定义
- 调整推理参数以适应您的应用场景

## 性能基准
- **推理速度**: ~7-9ms/图片（640x640，RTX 4060）
- **支持格式**: JPG, JPEG, PNG, BMP
- **推荐分辨率**: 640x640

## 开发状态
- [✔] 修复NMS和红色PCB瑕疵检测
- [✔] 更新置信度和滑动方法
- [✔] Conda环境配置和依赖管理
- [✔] 推理功能测试和验证
- [✔] 跨平台兼容性支持

## 故障排除

### 常见问题
1. **CUDA不可用**: 检查CUDA版本与PyTorch版本匹配
2. **依赖冲突**: 使用独立的conda环境
3. **模型加载失败**: 检查模型文件路径和格式
4. **推理结果为空**: 调整置信度阈值或检查图片质量

### 环境验证
运行以下命令验证环境配置：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 技术支持
如遇到问题，请检查：
1. Python版本兼容性
2. CUDA和PyTorch版本匹配
3. 依赖包版本冲突
4. 模型文件完整性
