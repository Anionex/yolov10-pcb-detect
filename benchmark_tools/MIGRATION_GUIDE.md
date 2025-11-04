# 📦 Benchmark工具迁移指南

## 概述

所有benchmark相关文件已从项目根目录迁移到 `benchmark_tools/` 目录，使项目结构更加清晰。

---

## 🔄 文件变更对照表

| 旧位置 | 新位置 | 说明 |
|--------|--------|------|
| `benchmark.py` | `benchmark_tools/benchmark.py` | 核心评估引擎 |
| `run_benchmark.py` | `benchmark_tools/run_benchmark.py` | 运行脚本 |
| `benchmark_config.yaml` | `benchmark_tools/benchmark_config.yaml` | 配置文件 |
| `BENCHMARK_GUIDE.md` | `benchmark_tools/BENCHMARK_GUIDE.md` | 使用指南 |
| `benchmark_results/` | `benchmark_tools/benchmark_results/` | 结果目录 |
| *(新增)* | `run_benchmark_tool.py` | 根目录便捷启动脚本 |

---

## 🚀 使用方式更新

### 命令行使用

#### ✅ 推荐方式1: 使用便捷脚本（从项目根目录）
```bash
# 快速验证（10张图片）
python run_benchmark_tool.py --mode single --limit 10

# 完整评估
python run_benchmark_tool.py --mode single

# 批量实验
python run_benchmark_tool.py --mode batch --limit 50
```

#### ✅ 推荐方式2: 进入目录运行
```bash
cd benchmark_tools

# 快速验证
python run_benchmark.py --mode single --limit 10

# 完整评估
python run_benchmark.py --mode single

# 批量实验
python run_benchmark.py --mode batch
```

#### ❌ 旧方式（不再支持）
```bash
# 以下命令现在会报错
python run_benchmark.py --mode single
python benchmark.py --dataset_path data/test
```

---

## 🐍 Python API 使用

### 新的导入方式

```python
# 方式1: 使用完整包路径
from benchmark_tools import PCBBenchmark, compare_benchmarks

benchmark = PCBBenchmark(
    model_path="weights/best.pt",
    dataset_path="data/mix_pcb_test_dataset/test",
    limit=100
)
results = benchmark.run_benchmark()
```

```python
# 方式2: 直接导入模块
import sys
sys.path.insert(0, 'benchmark_tools')

from benchmark import PCBBenchmark

benchmark = PCBBenchmark(
    model_path="../weights/best.pt",  # 注意路径
    dataset_path="../data/mix_pcb_test_dataset/test",
    limit=100
)
results = benchmark.run_benchmark()
```

### 旧的导入方式（不再支持）

```python
# ❌ 这将不再工作
from benchmark import PCBBenchmark
```

---

## 📝 配置文件路径更新

### benchmark_config.yaml 变更

```yaml
# ✅ 新路径（相对于 benchmark_tools/）
base:
  model_path: "../weights/best.pt"
  dataset_path: "../data/mix_pcb_test_dataset/test"
  save_dir: "benchmark_results"

# ❌ 旧路径（已不适用）
base:
  model_path: "weights/best.pt"
  dataset_path: "data/mix_pcb_test_dataset/test"
```

如果你从项目根目录使用 `run_benchmark_tool.py`，路径会自动处理。

---

## 🔧 自动化脚本更新

如果你有使用benchmark的自动化脚本，需要更新：

### 旧脚本
```bash
#!/bin/bash
cd /path/to/project
python run_benchmark.py --mode single
```

### 新脚本（选项1：使用便捷脚本）
```bash
#!/bin/bash
cd /path/to/project
python run_benchmark_tool.py --mode single --limit 100
```

### 新脚本（选项2：进入目录）
```bash
#!/bin/bash
cd /path/to/project/benchmark_tools
python run_benchmark.py --mode single --limit 100
```

---

## 📊 CI/CD 配置更新

如果在CI/CD中使用benchmark：

### GitHub Actions 示例

```yaml
# ✅ 新配置
- name: Run Benchmark
  run: |
    python run_benchmark_tool.py --mode single --limit 50
    
# 或者
- name: Run Benchmark
  run: |
    cd benchmark_tools
    python run_benchmark.py --mode single --limit 50
```

---

## 🐛 常见问题

### Q1: 运行报错 "No module named 'benchmark'"
**原因**: 导入路径错误  
**解决**: 
```python
# 使用新的导入方式
from benchmark_tools import PCBBenchmark
# 或添加路径
sys.path.insert(0, 'benchmark_tools')
```

### Q2: 运行报错 "FileNotFoundError: weights/best.pt"
**原因**: 相对路径问题  
**解决**: 使用 `run_benchmark_tool.py` 或确保配置文件使用正确路径

### Q3: 如何快速测试是否迁移成功？
```bash
# 从项目根目录运行
python run_benchmark_tool.py --help

# 应该看到帮助信息，没有报错
```

### Q4: 旧的benchmark结果文件在哪里？
已移动到 `benchmark_tools/benchmark_results/`

---

## ✨ 新增功能

这次迁移同时带来了新功能：

### 1. --limit 参数
```bash
# 快速验证（仅10张图片，~30秒）
python run_benchmark_tool.py --limit 10

# 中等测试（100张，~5分钟）
python run_benchmark_tool.py --limit 100

# 完整评估（全部）
python run_benchmark_tool.py --limit -1
```

### 2. 优化的输出
- 移除了不必要的调试信息
- 更清晰的进度显示
- 更快的运行速度

### 3. 完善的文档
- `benchmark_tools/README.md` - 快速入门
- `benchmark_tools/BENCHMARK_GUIDE.md` - 详细指南
- `benchmark_tools/CHANGELOG.md` - 更新日志

---

## 📚 推荐迁移步骤

### 步骤1: 熟悉新结构
```bash
cd benchmark_tools
ls -la
# 查看所有文件
```

### 步骤2: 测试基本功能
```bash
cd ..  # 回到项目根目录
python run_benchmark_tool.py --mode single --limit 10
```

### 步骤3: 更新你的脚本/代码
根据上面的示例更新导入和路径

### 步骤4: 删除旧的引用
检查并删除对旧路径的任何引用

---

## 📞 需要帮助？

- 查看 [README.md](./README.md) 获取快速入门
- 查看 [BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md) 获取详细文档
- 查看 [CHANGELOG.md](./CHANGELOG.md) 了解最新变更

---

**迁移日期**: 2024-11-04  
**影响范围**: 所有使用benchmark功能的代码和脚本  
**兼容性**: 不向后兼容（需要更新代码）

