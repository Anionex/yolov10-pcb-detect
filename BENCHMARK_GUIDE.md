# PCB检测Benchmark使用指南

## 📋 目录
- [简介](#简介)
- [快速开始](#快速开始)
- [数据集准备](#数据集准备)
- [运行Benchmark](#运行benchmark)
- [配置优化实验](#配置优化实验)
- [结果分析](#结果分析)
- [FAQ](#faq)

---

## 简介

这个Benchmark模块用于系统化评估PCB瑕疵检测模型的性能，支持：

✅ **标准评估指标**: mAP, Precision, Recall, F1
✅ **速度分析**: FPS, 推理时间统计
✅ **类别级别评估**: 每个瑕疵类型的详细指标
✅ **多配置对比**: 批量测试不同参数组合
✅ **结果可视化**: JSON和CSV格式输出

---

## 快速开始

### 1. 安装依赖

```bash
# 如果还没安装，添加以下依赖
pip install pyyaml pandas tqdm
```

### 2. 准备测试数据

将测试集组织为YOLO格式：

```
datasets/
└── test/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── labels/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

或者直接将图片和标注放在同一目录：

```
datasets/test/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

### 3. 快速测试

```bash
# 方式1: 直接运行（会显示交互菜单）
python run_benchmark.py

# 方式2: 快速测试单张图片
python run_benchmark.py --mode quick --image data/test_pcb.jpg

# 方式3: 运行完整benchmark
python benchmark.py --dataset_path datasets/test --model_path weights/best.pt
```

---

## 数据集准备

### YOLO标注格式

每个`.txt`文件对应一张图片，格式为：

```
<class_id> <center_x> <center_y> <width> <height>
```

其中坐标都是归一化的(0-1)。

**例子**:
```
0 0.5 0.5 0.2 0.3
3 0.7 0.3 0.15 0.25
```

### 类别映射

在`config.py`中定义：

```python
id2cls_name_custom = [
    "Mouse_bite",      # 0
    "Spur",            # 1
    "Missing_hole",    # 2
    "Short",           # 3
    "Open_circuit",    # 4
    "Spurious_copper"  # 5
]
```

### 已有数据集转换

项目中提供了转换工具：

```bash
# PKU数据集转换
python utils/pku_dataset_convert_to_yolo.py

# DeepPCB数据集转换
python utils/deep_pcb_dataset_convert_to_yolo.py
```

---

## 运行Benchmark

### 方法1: 使用命令行

```bash
# 基础用法
python benchmark.py \
    --dataset_path datasets/test \
    --model_path weights/best.pt \
    --conf_threshold 0.4 \
    --nms_threshold 0.1

# 自定义保存目录
python benchmark.py \
    --dataset_path datasets/test \
    --model_path weights/best.pt \
    --save_dir my_benchmark_results
```

### 方法2: 使用配置文件

1. **编辑配置** (`benchmark_config.yaml`):

```yaml
base:
  model_path: "weights/best.pt"
  dataset_path: "datasets/test"  # 修改为你的路径
  conf_threshold: 0.4
  nms_threshold: 0.1
```

2. **运行**:

```bash
python run_benchmark.py --mode single
```

### 方法3: 交互式菜单

```bash
python run_benchmark.py

# 然后根据提示选择：
# 1. 快速测试
# 2. 单次Benchmark
# 3. 批量实验
# 4. 对比已有结果
```

---

## 配置优化实验

### 批量测试不同配置

`benchmark_config.yaml`中预定义了多种实验配置：

```yaml
experiments:
  baseline:
    name: "Baseline (current)"
    conf_threshold: 0.4
    nms_threshold: 0.1
    
  high_confidence:
    name: "High Confidence"
    conf_threshold: 0.5
    
  more_overlap:
    name: "More Overlap"
    step_size: 160  # 更多窗口重叠
```

**运行批量实验**:

```bash
python run_benchmark.py --mode batch
```

这会依次运行所有配置，并自动生成对比报告。

### 自定义实验

在`benchmark_config.yaml`中添加：

```yaml
experiments:
  my_experiment:
    name: "My Custom Config"
    conf_threshold: 0.45
    nms_threshold: 0.15
    window_size: 672
    step_size: 336
```

---

## 结果分析

### 输出文件

Benchmark运行后会生成：

```
benchmark_results/
├── benchmark_20241103_143025.json  # 详细结果
├── benchmark_20241103_143025.csv   # 表格格式
└── comparison_latest.json          # 对比结果（如果运行了多个）
```

### JSON结果结构

```json
{
  "timestamp": "2024-11-03 14:30:25",
  "model_path": "weights/best.pt",
  "config": {
    "conf_threshold": 0.4,
    "nms_threshold": 0.1,
    "window_size": 608,
    "step_size": 320
  },
  "metrics": {
    "overall": {
      "precision": 0.8523,
      "recall": 0.7891,
      "f1": 0.8193,
      "mAP": 0.8012
    },
    "per_class": {
      "Short": {
        "precision": 0.9012,
        "recall": 0.8523,
        "f1": 0.8760,
        "ap": 0.8654
      },
      ...
    }
  },
  "speed": {
    "mean_time": 2.345,
    "fps": 0.426,
    "total_images": 50
  }
}
```

### 关键指标解读

| 指标 | 含义 | 建议 |
|------|------|------|
| **mAP@0.5** | 平均精度 (IoU≥0.5) | 工业应用最重要指标，建议≥0.8 |
| **Precision** | 查准率 | 减少误报，建议≥0.85 |
| **Recall** | 召回率 | 减少漏检，建议≥0.80 |
| **F1 Score** | 综合指标 | 平衡精确和召回 |
| **FPS** | 每秒帧数 | 实时性要求 |

### 对比多个配置

```bash
# 手动对比
python benchmark.py --compare \
    benchmark_results/benchmark_20241103_120000.json \
    benchmark_results/benchmark_20241103_130000.json \
    benchmark_results/benchmark_20241103_140000.json
```

输出示例：

```
Model                          mAP        Precision    Recall       F1         FPS
--------------------------------------------------------------------------------
baseline                       0.8012     0.8523       0.7891       0.8193     0.43
high_confidence                0.7856     0.8945       0.7234       0.7998     0.43
more_overlap                   0.8234     0.8612       0.8123       0.8360     0.21
```

---

## 优化指南

### 根据结果调优

#### 情况1: Precision高，Recall低

**问题**: 漏检太多
**解决**:
- 降低`conf_threshold` (0.4 → 0.3)
- 增加窗口重叠 (`step_size` 320 → 160)
- 尝试多尺度测试

#### 情况2: Recall高，Precision低

**问题**: 误检太多
**解决**:
- 提高`conf_threshold` (0.4 → 0.5)
- 降低`nms_threshold` (0.1 → 0.05)
- 检查训练数据质量

#### 情况3: 速度太慢

**优化**:
- 增大`step_size` (320 → 480)
- 减小`window_size` (608 → 544)
- 考虑模型压缩/量化

#### 情况4: 某些类别AP很低

**分析**:
- 查看per_class结果
- 可能需要针对该类别收集更多训练数据
- 考虑类别平衡策略

---

## 实验建议

### 推荐的实验流程

1. **建立基线** (Baseline)
   ```bash
   python benchmark.py --dataset_path datasets/test
   ```

2. **调整置信度阈值** (0.3, 0.4, 0.5, 0.6)
   ```bash
   # 批量测试
   python run_benchmark.py --mode batch
   ```

3. **优化滑动窗口** (step_size: 160, 240, 320, 480)
   - 观察mAP vs Speed权衡

4. **测试TTA增强** (如果实现)
   - 评估精度提升是否值得时间开销

5. **尝试WBF替代NMS**
   - 修改`customize_service.py`实现

6. **Multi-scale测试**
   - 不同window_size组合

### 记录实验

建议使用表格记录：

| 实验ID | 配置 | mAP | Precision | Recall | FPS | 备注 |
|--------|------|-----|-----------|--------|-----|------|
| exp001 | baseline | 0.801 | 0.852 | 0.789 | 0.43 | 当前配置 |
| exp002 | conf=0.5 | 0.785 | 0.894 | 0.723 | 0.43 | 误检少但漏检多 |
| exp003 | step=160 | 0.823 | 0.861 | 0.812 | 0.21 | 最佳精度，慢2倍 |

---

## FAQ

### Q1: 运行报错 "No valid test data found"

**A**: 检查：
- `dataset_path`是否正确
- 图片和标注文件是否存在
- 文件扩展名是否支持 (.jpg, .png, .bmp)

### Q2: mAP计算结果为0

**A**: 可能原因：
- IoU阈值太高，没有匹配的检测框
- 类别ID不匹配（检查`config.py`）
- 模型完全没检测到目标

### Q3: 如何只测试特定类别？

**A**: 修改`benchmark.py`中的`calculate_metrics`方法，过滤特定类别。

### Q4: 速度测试不准确

**A**: 
- 第一次推理会慢（模型加载、CUDA初始化）
- 运行多次取平均
- 考虑warm-up run

### Q5: 如何评估模型泛化能力？

**A**: 准备多个测试集：
- 同分布测试集（与训练集相同来源）
- 跨域测试集（不同PCB颜色、不同拍摄条件）

### Q6: 结果CSV如何用于可视化？

**A**: 使用pandas/matplotlib：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark_results/benchmark_20241103.csv')

# 绘制各类别AP
df.groupby('class')['ap'].mean().plot(kind='bar')
plt.title('Per-Class Average Precision')
plt.ylabel('AP')
plt.show()
```

---

## 进阶使用

### 集成到CI/CD

```yaml
# .github/workflows/benchmark.yml
name: Weekly Benchmark
on:
  schedule:
    - cron: '0 0 * * 0'  # 每周日运行

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Benchmark
        run: |
          python benchmark.py --dataset_path test_data/
          python scripts/upload_results.py
```

### 自动报告生成

```python
# 生成Markdown报告
from benchmark import PCBBenchmark

benchmark = PCBBenchmark(...)
results = benchmark.run_benchmark()

with open('RESULTS.md', 'w') as f:
    f.write(f"# Benchmark Results\n\n")
    f.write(f"**Date**: {results['timestamp']}\n\n")
    f.write(f"**mAP**: {results['metrics']['overall']['mAP']:.4f}\n\n")
    # ... 更多内容
```

---

## 贡献与反馈

如果你有改进建议或发现bug，欢迎提Issue或PR！

**常见改进方向**:
- [ ] 支持COCO格式标注
- [ ] 增加混淆矩阵可视化
- [ ] PR曲线绘制
- [ ] 支持分布式评估（多GPU）
- [ ] Web界面展示结果

---

## 参考资料

- [YOLO系列论文](https://github.com/ultralytics/ultralytics)
- [COCO评估指标](https://cocodataset.org/#detection-eval)
- [目标检测评估最佳实践](https://github.com/rafaelpadilla/Object-Detection-Metrics)

