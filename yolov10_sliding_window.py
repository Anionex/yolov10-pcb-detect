"""
自定义YOLOv10类，使 model.val() 自动使用滑动窗口推理
与 customize_service.py 的推理方式完全一致

使用方法：
    from yolov10_sliding_window import YOLOv10SlidingWindow
    
    model = YOLOv10SlidingWindow("weights/best.pt")
    metrics = model.val(save=True)  # 自动使用滑动窗口推理
"""

import cv2
import torch
import numpy as np
import torchvision.ops as ops
from ultralytics import YOLOv10
from ultralytics.models.yolov10.val import YOLOv10DetectionValidator
from ultralytics.utils import LOGGER
from pathlib import Path


class SlidingWindowValidator(YOLOv10DetectionValidator):
    """
    重写YOLOv10的验证器，使其在推理时使用滑动窗口
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 滑动窗口参数（与customize_service.py完全一致）
        self.window_size = 608
        self.step_size = 320
        self.nms_threshold = 0.1
        self.predict_conf = 0.4
        
        LOGGER.info(f"✓ 使用滑动窗口验证器（与部署推理一致）")
        LOGGER.info(f"  窗口大小: {self.window_size}x{self.window_size}")
        LOGGER.info(f"  步长: {self.step_size}")
        LOGGER.info(f"  NMS阈值: {self.nms_threshold}")
    
    def _slide_window(self, image):
        """
        滑动窗口生成器（与customize_service.py完全一致）
        
        Args:
            image: numpy数组 (H, W, C)
        
        Yields:
            (crop_x, crop_y, cropped_image)
        """
        height, width = image.shape[:2]
        
        for y in range(0, height, self.step_size):
            for x in range(0, width, self.step_size):
                crop_x = min(x, width - self.window_size)
                crop_y = min(y, height - self.window_size)
                cropped_image = image[crop_y:crop_y + self.window_size, 
                                     crop_x:crop_x + self.window_size]
                yield (crop_x, crop_y, cropped_image)
    
    def preprocess(self, batch):
        """
        重写预处理方法
        保存原始图像路径用于滑动窗口推理
        """
        # 保存原始图像路径和形状
        self.current_im_files = batch.get('im_file', [])
        self.current_ori_shapes = batch.get('ori_shape', [])
        
        # 调用父类预处理（但我们不会使用它预处理的图像）
        return super().preprocess(batch)
    
    def __call__(self, trainer=None, model=None):
        """
        重写验证的主循环
        拦截推理过程，使用滑动窗口替代标准推理
        """
        # 设置标志
        self._use_sliding_window = True
        
        # 保存模型引用
        self._model = model
        
        # 调用父类的验证流程
        return super().__call__(trainer=trainer, model=model)
    
    def update_metrics(self, preds, batch):
        """
        重写update_metrics，在这里拦截推理过程
        使用滑动窗口推理替换原始预测
        """
        # 如果启用了滑动窗口，替换预测结果
        if hasattr(self, '_use_sliding_window') and self._use_sliding_window:
            # 对batch中的每张图像进行滑动窗口推理
            new_preds = []
            
            for idx, im_file in enumerate(self.current_im_files):
                # 使用滑动窗口推理
                pred = self._sliding_window_inference(im_file, batch, idx)
                new_preds.append(pred)
            
            # 将列表转换为适当的格式
            if new_preds:
                # 过滤掉None
                new_preds = [p for p in new_preds if p is not None]
                if new_preds:
                    preds = new_preds
        
        # 调用父类的update_metrics
        super().update_metrics(preds, batch)
    
    def _sliding_window_inference(self, im_file, batch, idx):
        """
        对单张图像进行滑动窗口推理
        
        Args:
            im_file: 图像文件路径
            batch: 当前batch数据
            idx: 图像在batch中的索引
        
        Returns:
            合并后的预测结果 (N, 6) [x1, y1, x2, y2, conf, class]
        """
        # 读取原始大图
        image = cv2.imread(str(im_file))
        if image is None:
            LOGGER.warning(f"无法读取图像: {im_file}")
            return torch.zeros((0, 6), device=batch['img'].device)
        
        original_h, original_w = image.shape[:2]
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # 对每个窗口进行推理
        for (x, y, window) in self._slide_window(image):
            # 转换为模型输入格式
            # window是BGR格式，需要转RGB并归一化
            window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            window_tensor = torch.from_numpy(window_rgb).to(batch['img'].device)
            window_tensor = window_tensor.permute(2, 0, 1).float() / 255.0
            window_tensor = window_tensor.unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                preds = self.model(window_tensor, augment=False)
            
            # 后处理
            processed_preds = self.postprocess(preds)
            
            if len(processed_preds) > 0 and processed_preds[0] is not None:
                pred = processed_preds[0]
                
                if len(pred) > 0:
                    # pred格式: [x1, y1, x2, y2, conf, class]
                    boxes = pred[:, :4].clone()
                    scores = pred[:, 4]
                    classes = pred[:, 5]
                    
                    # 映射回原图坐标
                    boxes[:, [0, 2]] += x
                    boxes[:, [1, 3]] += y
                    
                    # 限制在图像范围内
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_h)
                    
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_classes.append(classes)
        
        # 如果没有检测到任何目标
        if len(all_boxes) == 0:
            return torch.zeros((0, 6), device=batch['img'].device)
        
        # 合并所有窗口的检测结果
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_classes = torch.cat(all_classes, dim=0)
        
        # NMS去重（与customize_service.py一致）
        keep = ops.nms(all_boxes, all_scores, self.nms_threshold)
        
        # 过滤Missing_hole类别（类别索引2）
        keep = [i for i in keep if all_classes[i] != 2]
        
        if len(keep) == 0:
            return torch.zeros((0, 6), device=batch['img'].device)
        
        # 组合最终结果
        final_boxes = all_boxes[keep]
        final_scores = all_scores[keep].unsqueeze(1)
        final_classes = all_classes[keep].unsqueeze(1)
        
        # 返回格式: [x1, y1, x2, y2, conf, class]
        result = torch.cat([final_boxes, final_scores, final_classes], dim=1)
        
        return result


class YOLOv10SlidingWindow(YOLOv10):
    """
    自定义YOLOv10类，使 val() 方法自动使用滑动窗口推理
    
    使用方法：
        model = YOLOv10SlidingWindow("weights/best.pt")
        metrics = model.val(save=True)  # 自动使用滑动窗口
    """
    
    @property
    def task_map(self):
        """
        重写task_map，使用自定义的滑动窗口验证器
        """
        return {
            "detect": {
                "predictor": self._get_default_task_map()["detect"]["predictor"],
                "trainer": self._get_default_task_map()["detect"]["trainer"],
                "validator": SlidingWindowValidator,  # 使用自定义验证器
                "model": self._get_default_task_map()["detect"]["model"],
            }
        }
    
    def _get_default_task_map(self):
        """获取默认的task_map"""
        return super().task_map


# 便捷函数
def load_model_with_sliding_window(weights):
    """
    加载使用滑动窗口验证的模型
    
    Args:
        weights: 权重文件路径
    
    Returns:
        YOLOv10SlidingWindow模型实例
    
    Example:
        >>> model = load_model_with_sliding_window("weights/best.pt")
        >>> metrics = model.val(save=True)
    """
    return YOLOv10SlidingWindow(weights)

