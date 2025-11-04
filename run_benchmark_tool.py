#!/usr/bin/env python
"""
便捷启动脚本 - 从项目根目录启动benchmark工具

使用方法:
    python run_benchmark_tool.py --mode single --limit 10
    
更多选项请查看 benchmark_tools/README.md
"""

import sys
import os
from pathlib import Path

# 添加benchmark_tools目录到路径
benchmark_tools_dir = Path(__file__).parent / "benchmark_tools"
sys.path.insert(0, str(benchmark_tools_dir))

# 切换到benchmark_tools目录
os.chdir(str(benchmark_tools_dir))

# 导入并运行
from run_benchmark import main

if __name__ == '__main__':
    main()

