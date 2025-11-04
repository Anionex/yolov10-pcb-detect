"""
PCB瑕疵检测Benchmark模块

用于评估不同配置和优化策略的性能
支持多种评估指标和详细的性能分析

使用方法:
    python benchmark.py --dataset_path ./datasets/test --model_path weights/best.pt
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 添加父目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from customize_service import yolov10_detection
from config import id2cls_name_custom


class PCBBenchmark:
    """PCB检测性能评估类"""
    
    def __init__(self, 
                 model_path: str,
                 dataset_path: str = None,
                 conf_threshold: float = 0.4,
                 nms_threshold: float = 0.1,
                 save_dir: str = "benchmark_results",
                 limit: int = -1):
        """
        初始化Benchmark
        
        Args:
            model_path: 模型权重路径
            dataset_path: 测试数据集路径 (YOLO格式: images/ 和 labels/)
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            save_dir: 结果保存目录
            limit: 测试图片数量上限，-1表示测试全部
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.limit = limit
        
        # 初始化服务
        print(f"Loading model from {model_path}...")
        self.service = yolov10_detection(
            model_name="yolov10", 
            model_path=model_path
        )
        self.service.predict_conf = conf_threshold
        self.service.nms_threshold = nms_threshold
        
        # 类别名称
        self.class_names = id2cls_name_custom
        self.num_classes = len(self.class_names)
        
        # 结果存储
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'inference_times': [],
            'image_paths': []
        }
        
    def load_dataset(self) -> List[Tuple[str, str]]:
        """
        加载测试数据集
        
        Returns:
            [(image_path, label_path), ...] 列表
        """
        if self.dataset_path is None:
            raise ValueError("Dataset path not specified!")
            
        dataset_path = Path(self.dataset_path)
        
        # 支持两种目录结构
        # 1. dataset_path/images/ 和 dataset_path/labels/
        # 2. dataset_path 直接包含图片和txt
        
        images_dir = dataset_path / "images" if (dataset_path / "images").exists() else dataset_path
        labels_dir = dataset_path / "labels" if (dataset_path / "labels").exists() else dataset_path
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        dataset = []
        for img_path in sorted(image_files):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dataset.append((str(img_path), str(label_path)))
            else:
                print(f"Warning: No label found for {img_path.name}")
        
        # 应用数量限制
        if self.limit > 0 and len(dataset) > self.limit:
            print(f"Found {len(dataset)} images with labels")
            print(f"⚠️  Limiting to first {self.limit} images for quick testing")
            dataset = dataset[:self.limit]
        else:
            print(f"Found {len(dataset)} images with labels")
        
        return dataset
    
    def parse_yolo_label(self, label_path: str, img_width: int, img_height: int) -> List[Dict]:
        """
        解析YOLO格式的标注文件
        
        Args:
            label_path: 标注文件路径
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            [{'class': cls_id, 'bbox': [xmin, ymin, xmax, ymax]}, ...]
        """
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                # 转换为绝对坐标
                xmin = (cx - w / 2) * img_width
                ymin = (cy - h / 2) * img_height
                xmax = (cx + w / 2) * img_width
                ymax = (cy + h / 2) * img_height
                
                boxes.append({
                    'class': cls_id,
                    'bbox': [xmin, ymin, xmax, ymax]
                })
                
        return boxes
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 交集
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # 并集
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def run_inference_single(self, image_path: str) -> Tuple[Dict, float]:
        """
        对单张图片进行推理
        
        Returns:
            (result_dict, inference_time)
        """
        # 预处理
        post_data = {"input_txt": {os.path.basename(image_path): open(image_path, "rb")}}
        self.service._preprocess(post_data)
        
        # 推理 (计时)
        start_time = time.time()
        data = self.service._inference('ok')
        result = self.service._postprocess(data)
        inference_time = time.time() - start_time
        
        return result, inference_time
    
    def calculate_metrics(self, iou_threshold: float = 0.5) -> Dict:
        """
        计算评估指标
        
        Args:
            iou_threshold: IoU阈值，默认0.5
            
        Returns:
            metrics字典
        """
        # 初始化统计量
        stats = {
            'tp': np.zeros(self.num_classes),  # True Positives
            'fp': np.zeros(self.num_classes),  # False Positives
            'fn': np.zeros(self.num_classes),  # False Negatives
            'scores': [[] for _ in range(self.num_classes)],  # 置信度分数
        }
        
        # 遍历每张图片的结果
        for pred, gt in zip(self.results['predictions'], self.results['ground_truths']):
            pred_boxes = pred['detection_boxes']
            pred_classes = pred['detection_classes']
            pred_scores = pred['detection_scores']
            
            gt_boxes = [g['bbox'] for g in gt]
            gt_classes = [g['class'] for g in gt]
            
            # 转换预测类别名称为ID
            pred_class_ids = [self.class_names.index(cls) for cls in pred_classes]
            
            # 标记已匹配的GT
            gt_matched = [False] * len(gt_boxes)
            
            # 按置信度降序处理预测框
            sorted_indices = np.argsort(pred_scores)[::-1]
            
            for idx in sorted_indices:
                pred_bbox = pred_boxes[idx]
                pred_cls = pred_class_ids[idx]
                pred_score = pred_scores[idx]
                
                # 转换bbox格式 [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax]
                pred_bbox_converted = [pred_bbox[1], pred_bbox[0], pred_bbox[3], pred_bbox[2]]
                
                # 查找最佳匹配的GT
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_bbox, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if gt_matched[gt_idx]:
                        continue
                    if gt_cls != pred_cls:
                        continue
                        
                    iou = self.calculate_iou(pred_bbox_converted, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # 判断TP或FP
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    stats['tp'][pred_cls] += 1
                    gt_matched[best_gt_idx] = True
                    stats['scores'][pred_cls].append(pred_score)
                else:
                    stats['fp'][pred_cls] += 1
            
            # 统计FN (未匹配的GT)
            for gt_idx, (gt_cls, matched) in enumerate(zip(gt_classes, gt_matched)):
                if not matched:
                    stats['fn'][gt_cls] += 1
        
        # 计算各类别指标
        metrics = {
            'per_class': {},
            'overall': {}
        }
        
        all_tp = 0
        all_fp = 0
        all_fn = 0
        aps = []
        
        for cls_id, cls_name in enumerate(self.class_names):
            tp = stats['tp'][cls_id]
            fp = stats['fp'][cls_id]
            fn = stats['fn'][cls_id]
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 简化的AP计算 (实际应该用PR曲线)
            ap = precision * recall if recall > 0 else 0
            
            metrics['per_class'][cls_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'ap': float(ap),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }
            
            all_tp += tp
            all_fp += fp
            all_fn += fn
            if (tp + fn) > 0:  # 只统计有GT的类别
                aps.append(ap)
        
        # 总体指标
        overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0
        
        metrics['overall'] = {
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1),
            'mAP': float(np.mean(aps)) if len(aps) > 0 else 0,
            'total_tp': int(all_tp),
            'total_fp': int(all_fp),
            'total_fn': int(all_fn)
        }
        
        return metrics
    
    def run_benchmark(self, save_predictions: bool = True) -> Dict:
        """
        运行完整的benchmark评估
        
        Args:
            save_predictions: 是否保存预测结果
            
        Returns:
            完整的评估结果
        """
        print("\n" + "="*60)
        print("Starting PCB Detection Benchmark")
        print("="*60)
        
        # 加载数据集
        dataset = self.load_dataset()
        
        if len(dataset) == 0:
            raise ValueError("No valid test data found!")
        
        print(f"\nProcessing {len(dataset)} images...")
        
        # 推理
        for img_path, label_path in tqdm(dataset, desc="Inference"):
            # 读取图片获取尺寸
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            
            # 推理
            result, inf_time = self.run_inference_single(img_path)
            
            # 加载GT
            gt = self.parse_yolo_label(label_path, img_w, img_h)
            
            # 存储结果
            self.results['predictions'].append(result)
            self.results['ground_truths'].append(gt)
            self.results['inference_times'].append(inf_time)
            self.results['image_paths'].append(img_path)
        
        # 计算指标
        print("\nCalculating metrics...")
        metrics = self.calculate_metrics(iou_threshold=0.5)
        
        # 速度统计
        inference_times = np.array(self.results['inference_times'])
        speed_stats = {
            'mean_time': float(np.mean(inference_times)),
            'std_time': float(np.std(inference_times)),
            'min_time': float(np.min(inference_times)),
            'max_time': float(np.max(inference_times)),
            'fps': float(1.0 / np.mean(inference_times)),
            'total_images': len(dataset)
        }
        
        # 组装完整结果
        benchmark_result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': self.model_path,
            'dataset_path': self.dataset_path,
            'config': {
                'conf_threshold': self.conf_threshold,
                'nms_threshold': self.nms_threshold,
                'window_size': self.service.window_size,
                'step_size': self.service.step_size
            },
            'metrics': metrics,
            'speed': speed_stats
        }
        
        # 打印结果
        self.print_results(benchmark_result)
        
        # 保存结果
        if save_predictions:
            self.save_results(benchmark_result)
        
        return benchmark_result
    
    def print_results(self, results: Dict):
        """打印benchmark结果"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        # 总体指标
        overall = results['metrics']['overall']
        print(f"\n【总体性能】")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1 Score:  {overall['f1']:.4f}")
        print(f"  mAP@0.5:   {overall['mAP']:.4f}")
        print(f"  TP/FP/FN:  {overall['total_tp']}/{overall['total_fp']}/{overall['total_fn']}")
        
        # 各类别指标
        print(f"\n【各类别性能】")
        per_class = results['metrics']['per_class']
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AP':<12}")
        print("-" * 68)
        for cls_name, metrics in per_class.items():
            print(f"{cls_name:<20} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} "
                  f"{metrics['ap']:<12.4f}")
        
        # 速度指标
        speed = results['speed']
        print(f"\n【速度性能】")
        print(f"  平均推理时间: {speed['mean_time']:.3f}s ± {speed['std_time']:.3f}s")
        print(f"  FPS:          {speed['fps']:.2f}")
        print(f"  最快/最慢:     {speed['min_time']:.3f}s / {speed['max_time']:.3f}s")
        
        print("\n" + "="*60)
    
    def save_results(self, results: Dict):
        """保存benchmark结果"""
        # 保存JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.save_dir / f"benchmark_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {json_path}")
        
        # 保存CSV (更方便对比)
        csv_path = self.save_dir / f"benchmark_{timestamp}.csv"
        
        # 准备CSV数据
        csv_data = []
        for cls_name, metrics in results['metrics']['per_class'].items():
            row = {
                'timestamp': results['timestamp'],
                'model': Path(self.model_path).name,
                'class': cls_name,
                **metrics,
                'mean_time': results['speed']['mean_time'],
                'fps': results['speed']['fps']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        print(f"✓ CSV saved to: {csv_path}")
        
        return json_path, csv_path


def compare_benchmarks(result_files: List[str], output_path: str = None):
    """
    对比多个benchmark结果
    
    Args:
        result_files: benchmark结果JSON文件列表
        output_path: 对比结果保存路径
    """
    results = []
    for file in result_files:
        with open(file, 'r', encoding='utf-8') as f:
            results.append(json.load(f))
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    # 对比总体指标
    print(f"\n{'Model':<30} {'mAP':<10} {'Precision':<12} {'Recall':<12} {'F1':<10} {'FPS':<8}")
    print("-" * 82)
    
    for result in results:
        model_name = Path(result['model_path']).name
        overall = result['metrics']['overall']
        fps = result['speed']['fps']
        
        print(f"{model_name:<30} {overall['mAP']:<10.4f} "
              f"{overall['precision']:<12.4f} {overall['recall']:<12.4f} "
              f"{overall['f1']:<10.4f} {fps:<8.2f}")
    
    print("="*80)
    
    # 保存对比结果
    if output_path:
        comparison = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models': [Path(r['model_path']).name for r in results],
            'comparison': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PCB Detection Benchmark')
    parser.add_argument('--model_path', type=str, default='weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to test dataset (YOLO format)')
    parser.add_argument('--conf_threshold', type=float, default=0.4,
                       help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.1,
                       help='NMS threshold')
    parser.add_argument('--save_dir', type=str, default='benchmark_results',
                       help='Directory to save results')
    parser.add_argument('--limit', type=int, default=-1,
                       help='Limit number of test images (-1 for all)')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare multiple benchmark results')
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比模式
        compare_benchmarks(args.compare, 
                          output_path='benchmark_results/comparison.json')
    else:
        # 单次评估模式
        benchmark = PCBBenchmark(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            save_dir=args.save_dir,
            limit=args.limit
        )
        
        results = benchmark.run_benchmark()


if __name__ == '__main__':
    main()

