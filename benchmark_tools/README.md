# 📊 PCB检测 Benchmark 工具集

这个文件夹包含了完整的PCB瑕疵检测模型性能评估工具。

## 📁 文件结构

```
benchmark_tools/
├── benchmark.py              # 核心评估引擎
├── run_benchmark.py          # 命令行运行脚本
├── benchmark_config.yaml     # 配置文件
├── BENCHMARK_GUIDE.md        # 详细使用指南
├── benchmark_results/        # 评估结果输出目录
│   └── benchmark_*.json      # 评估结果
└── README.md                 # 本文件
```

## 🚀 快速开始

### 1. 快速验证（10张图片，~30秒）
```bash
cd benchmark_tools
python run_benchmark.py --mode single --limit 10
```

### 2. 完整评估（全部数据集）
```bash
cd benchmark_tools
python run_benchmark.py --mode single
```

### 3. 批量实验对比
```bash
cd benchmark_tools
python run_benchmark.py --mode batch --limit 100
```

## 📖 详细文档

请查看 [BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md) 获取完整使用指南。

## ⚙️ 主要功能

- ✅ **完整评估指标**: mAP, Precision, Recall, F1
- ✅ **速度分析**: FPS, 推理时间统计
- ✅ **类别级别评估**: 每个瑕疵类型的详细指标
- ✅ **多配置对比**: 批量测试不同参数组合
- ✅ **样本数量控制**: `--limit` 参数快速验证
- ✅ **结果导出**: JSON和CSV格式

## 🎯 三种运行模式

### Mode 1: Quick - 快速测试单张图片
```bash
python run_benchmark.py --mode quick --image ../data/test_pcb.jpg
```

### Mode 2: Single - 单次完整评估
```bash
python run_benchmark.py --mode single --config benchmark_config.yaml
```

### Mode 3: Batch - 批量实验对比
```bash
python run_benchmark.py --mode batch --config benchmark_config.yaml
```

## 📊 配置文件

编辑 `benchmark_config.yaml` 来自定义：
- 数据集路径
- 模型路径
- 置信度阈值
- NMS阈值
- 滑动窗口参数
- 实验配置

## 💡 重要参数

### --limit 参数
控制测试样本数量，快速验证配置：

```bash
# 快速验证（10张）
python run_benchmark.py --limit 10

# 中等测试（100张）
python run_benchmark.py --limit 100

# 完整评估（全部）
python run_benchmark.py --limit -1
```

### 时间估算

| Limit | 图片数 | 预计时间 | 适用场景 |
|-------|--------|---------|---------|
| 10 | 10 | ~30秒 | ⚡ 配置验证 |
| 100 | 100 | ~5分钟 | 📈 参数调优 |
| -1 | 全部 | ~25分钟 | ✅ 最终报告 |

## 🔧 命令行参数

```bash
python run_benchmark.py [选项]

选项:
  --mode {single,batch,quick}  运行模式
  --config PATH                配置文件路径
  --dataset PATH               数据集路径（覆盖配置）
  --model PATH                 模型路径（覆盖配置）
  --limit N                    测试图片数量上限（-1=全部）
  --image PATH                 快速测试的图片路径
```

## 📈 输出结果

运行后会在 `benchmark_results/` 生成：
- `benchmark_YYYYMMDD_HHMMSS.json` - 详细结果
- `benchmark_YYYYMMDD_HHMMSS.csv` - 表格数据
- `comparison_*.json` - 对比报告（batch模式）

## 🐍 Python API 使用

```python
import sys
sys.path.append('..')  # 添加父目录到路径

from benchmark_tools import PCBBenchmark

# 创建benchmark实例
benchmark = PCBBenchmark(
    model_path="../weights/best.pt",
    dataset_path="../data/mix_pcb_test_dataset/test",
    conf_threshold=0.4,
    nms_threshold=0.1,
    limit=100  # 只测试前100张
)

# 运行评估
results = benchmark.run_benchmark()

# 查看结果
print(f"mAP: {results['metrics']['overall']['mAP']:.4f}")
print(f"Precision: {results['metrics']['overall']['precision']:.4f}")
print(f"Recall: {results['metrics']['overall']['recall']:.4f}")
```

## ⚠️ 注意事项

1. **路径问题**: 从 `benchmark_tools/` 目录运行时，相对路径需要加 `../`
2. **数据集格式**: 必须是YOLO格式（images/ 和 labels/）
3. **性能优化**: 确保已关闭调试输出（customize_service.py中的print和cv2.imwrite）

## 🆘 常见问题

**Q: 运行报错找不到模块？**  
A: 确保在 `benchmark_tools/` 目录下运行，或者添加父目录到Python路径

**Q: 为什么这么慢？**  
A: 使用 `--limit 10` 快速测试，完整评估需要时间

**Q: 如何对比不同配置？**  
A: 使用 `--mode batch` 批量运行多个实验

## 📚 相关文件

- [../README.md](../README.md) - 项目主README
- [../customize_service.py](../customize_service.py) - 推理服务
- [../config.py](../config.py) - 全局配置

---

**版本**: 1.0  
**最后更新**: 2024-11-04

