"""
快速运行benchmark的脚本
支持批量测试多种配置
"""

import os
import sys
from pathlib import Path
import yaml

# 添加父目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_tools.benchmark import PCBBenchmark, compare_benchmarks
from customize_service import yolov10_detection


def run_single_benchmark(config: dict, experiment_name: str, limit: int = -1):
    """运行单个benchmark测试"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # 更新服务配置
    benchmark = PCBBenchmark(
        model_path=config['model_path'],
        dataset_path=config['dataset_path'],
        conf_threshold=config.get('conf_threshold', 0.4),
        nms_threshold=config.get('nms_threshold', 0.1),
        save_dir=config['save_dir'],
        limit=limit
    )
    
    # 如果配置中有window_size和step_size，更新服务
    if 'window_size' in config:
        benchmark.service.window_size = config['window_size']
    if 'step_size' in config:
        benchmark.service.step_size = config['step_size']
    
    # 运行benchmark
    result = benchmark.run_benchmark(save_predictions=True)
    
    return result


def run_all_experiments(config_file: str = "benchmark_config.yaml", limit: int = -1):
    """运行配置文件中的所有实验"""
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_config = config['base']
    experiments = config.get('experiments', {})
    
    if not experiments:
        print("No experiments defined in config file")
        return
    
    print(f"\n发现 {len(experiments)} 个实验配置")
    print("实验列表:")
    for i, (exp_name, exp_config) in enumerate(experiments.items(), 1):
        print(f"  {i}. {exp_name}: {exp_config.get('name', exp_name)}")
    
    # 询问用户
    choice = input("\n运行所有实验? (y/n, 或输入实验编号): ").strip().lower()
    
    result_files = []
    
    if choice == 'y':
        # 运行所有实验
        for exp_name, exp_config in experiments.items():
            # 合并base配置和实验配置
            full_config = {**base_config, **exp_config}
            
            try:
                result = run_single_benchmark(full_config, exp_config.get('name', exp_name), limit=limit)
                # 假设结果文件名包含在返回中，或者从save_dir获取最新文件
            except Exception as e:
                print(f"Error in experiment {exp_name}: {e}")
                continue
    
    elif choice.isdigit():
        # 运行特定实验
        idx = int(choice) - 1
        exp_items = list(experiments.items())
        
        if 0 <= idx < len(exp_items):
            exp_name, exp_config = exp_items[idx]
            full_config = {**base_config, **exp_config}
            run_single_benchmark(full_config, exp_config.get('name', exp_name), limit=limit)
        else:
            print("Invalid experiment number")
    
    else:
        print("取消运行")
        return
    
    # 对比结果
    result_dir = Path(base_config['save_dir'])
    if result_dir.exists():
        result_files = sorted(result_dir.glob("benchmark_*.json"))
        
        if len(result_files) > 1:
            print(f"\n发现 {len(result_files)} 个结果文件")
            do_compare = input("是否生成对比报告? (y/n): ").strip().lower()
            
            if do_compare == 'y':
                compare_benchmarks(
                    [str(f) for f in result_files[-5:]],  # 对比最近5个
                    output_path=str(result_dir / "comparison_latest.json")
                )


def quick_test(image_path: str = "data/test_pcb.jpg"):
    """快速测试单张图片"""
    print(f"Quick test on: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    from customize_service import yolov10_detection
    import time
    
    service = yolov10_detection(
        model_name="yolov10",
        model_path="weights/best.pt"
    )
    
    post_data = {"input_txt": {os.path.basename(image_path): open(image_path, "rb")}}
    
    service._preprocess(post_data)
    
    start = time.time()
    data = service._inference('ok')
    result = service._postprocess(data)
    elapsed = time.time() - start
    
    print(f"\n推理时间: {elapsed:.3f}s")
    print(f"检测到 {len(result['detection_boxes'])} 个目标:")
    
    for i, (cls, score) in enumerate(zip(result['detection_classes'], result['detection_scores'])):
        print(f"  {i+1}. {cls}: {score:.3f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run PCB Detection Benchmarks')
    parser.add_argument('--mode', choices=['single', 'batch', 'quick'], default='single',
                       help='Benchmark mode')
    parser.add_argument('--config', type=str, default='benchmark_config.yaml',
                       help='Config file path')
    parser.add_argument('--dataset', type=str, 
                       help='Dataset path (override config)')
    parser.add_argument('--model', type=str,
                       help='Model path (override config)')
    parser.add_argument('--image', type=str, default='data/test_pcb.jpg',
                       help='Image path for quick test')
    parser.add_argument('--limit', type=int, default=-1,
                       help='Limit number of test images (-1 for all, e.g., 10 for quick test)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_test(args.image)
        
    elif args.mode == 'batch':
        run_all_experiments(args.config, limit=args.limit)
        
    else:  # single
        # 加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        base_config = config['base']
        
        # 覆盖配置
        if args.dataset:
            base_config['dataset_path'] = args.dataset
        if args.model:
            base_config['model_path'] = args.model
        
        run_single_benchmark(base_config, "Single Run", limit=args.limit)


if __name__ == '__main__':
    # 如果没有命令行参数，显示菜单
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("PCB Detection Benchmark Tool")
        print("="*60)
        print("\n请选择运行模式:")
        print("1. 快速测试 (单张图片)")
        print("2. 单次Benchmark (使用配置文件)")
        print("3. 批量实验 (运行多个配置)")
        print("4. 对比已有结果")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == '1':
            img_path = input("图片路径 (默认: data/test_pcb.jpg): ").strip()
            quick_test(img_path if img_path else "data/test_pcb.jpg")
            
        elif choice == '2':
            config_file = input("配置文件 (默认: benchmark_config.yaml): ").strip()
            if not config_file:
                config_file = "benchmark_config.yaml"
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            run_single_benchmark(config['base'], "Single Run")
            
        elif choice == '3':
            config_file = input("配置文件 (默认: benchmark_config.yaml): ").strip()
            run_all_experiments(config_file if config_file else "benchmark_config.yaml")
            
        elif choice == '4':
            result_dir = Path("benchmark_results")
            result_files = sorted(result_dir.glob("benchmark_*.json"))
            
            if len(result_files) < 2:
                print("需要至少2个结果文件才能对比")
            else:
                print(f"\n找到 {len(result_files)} 个结果文件:")
                for i, f in enumerate(result_files, 1):
                    print(f"  {i}. {f.name}")
                
                indices = input("\n输入要对比的文件编号 (用逗号分隔，如: 1,2,3): ").strip()
                selected = [result_files[int(i)-1] for i in indices.split(',') if i.strip().isdigit()]
                
                if selected:
                    compare_benchmarks(
                        [str(f) for f in selected],
                        output_path="benchmark_results/comparison_manual.json"
                    )
        else:
            print("无效选项")
    else:
        main()

