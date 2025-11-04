# Benchmark Tools - 更新日志

## 2024-11-04

### 🎉 重大重构
将所有benchmark相关文件整理到 `benchmark_tools/` 目录，提高项目组织性。

### 📦 文件移动
- ✅ `benchmark.py` → `benchmark_tools/benchmark.py`
- ✅ `run_benchmark.py` → `benchmark_tools/run_benchmark.py`
- ✅ `benchmark_config.yaml` → `benchmark_tools/benchmark_config.yaml`
- ✅ `BENCHMARK_GUIDE.md` → `benchmark_tools/BENCHMARK_GUIDE.md`
- ✅ `benchmark_results/` → `benchmark_tools/benchmark_results/`

### ✨ 新增功能
1. **`--limit` 参数** - 控制测试样本数量
   ```bash
   python run_benchmark.py --limit 10    # 只测试10张
   python run_benchmark.py --limit 100   # 测试100张
   python run_benchmark.py --limit -1    # 全部测试
   ```

2. **便捷启动脚本** - 在项目根目录
   ```bash
   python run_benchmark_tool.py --mode single --limit 10
   ```

3. **完整文档** 
   - `benchmark_tools/README.md` - 工具说明
   - `benchmark_tools/BENCHMARK_GUIDE.md` - 详细指南

### 🔧 技术改进
- 自动添加父目录到Python路径，无需手动配置
- 更新配置文件中的相对路径
- 优化import语句

### 📝 使用方式变更

#### 之前:
```bash
python run_benchmark.py --mode single
```

#### 现在:
```bash
# 方式1: 根目录运行（推荐）
python run_benchmark_tool.py --mode single --limit 10

# 方式2: 进入目录运行
cd benchmark_tools
python run_benchmark.py --mode single --limit 10
```

### ⚠️ 注意事项
- 配置文件路径已更新为相对路径（`../weights/best.pt`）
- 从 `benchmark_tools/` 目录运行时会自动处理路径
- Python API 导入方式：`from benchmark_tools import PCBBenchmark`

### 🐛 Bug 修复
- 修复了长时间运行导致的性能问题（移除不必要的磁盘IO）
- 移除了大量调试输出（verbose=False）

---

## 历史版本

### 初始版本 (2024-11-03)
- 创建基础benchmark框架
- 支持mAP、Precision、Recall等指标计算
- 批量实验对比功能

