import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from typing import List, Dict, Tuple
import pycocotools.mask as mask_util
import random

# 尝试导入skimage用于纹理特征
try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class CellDataset(Dataset):
    """活体细胞检测数据集"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, max_size: int = 1024):
        """
        Args:
            data_dir: 数据目录路径
            split: 'train', 'val', 'test'
            transform: 图像变换
            max_size: 图像最大尺寸（用于resize，避免内存溢出）
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_size = max_size
        
        # 获取所有图像文件
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        all_files.sort()
        
        # 划分数据集 (70% train, 15% val, 15% test)
        n_total = len(all_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        if split == 'train':
            self.files = all_files[:n_train]
        elif split == 'val':
            self.files = all_files[n_train:n_train+n_val]
        else:  # test
            self.files = all_files[n_train+n_val:]
        
        print(f"{split} dataset: {len(self.files)} images")
    
    def __len__(self):
        return len(self.files)
    
    def _apply_cell_specific_preprocessing(self, image: np.ndarray, instance_masks: List[np.ndarray], 
                                           instance_labels: List[int]) -> np.ndarray:
        """针对活细胞和死细胞的高级特征工程和预处理 - 优化版本，提升速度"""
        # ========== 第一阶段：快速CLAHE增强 ==========
        # 使用单一CLAHE而不是多个，提升速度
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 使用单一CLAHE，参数平衡
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        image_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # ========== 第二阶段：简化边缘增强 ==========
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 只使用主要的Sobel边缘检测（x和y方向）
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edges_normalized = np.clip(edges_magnitude / (edges_magnitude.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        
        # 简化纹理特征（跳过LBP，使用简单的拉普拉斯）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_normalized = np.clip(np.abs(laplacian) / (np.abs(laplacian).max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        
        # 合并边缘特征（简化）
        edges_combined = (edges_normalized.astype(np.float32) * 0.7 + 
                         laplacian_normalized.astype(np.float32) * 0.3).astype(np.uint8)
        edges_rgb = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2RGB)
        
        # ========== 第三阶段：简化区域特定增强 ==========
        # 创建活细胞和死细胞的mask
        live_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        dead_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for mask, label in zip(instance_masks, instance_labels):
            if label == 0:  # live
                live_mask = np.maximum(live_mask, mask)
            else:  # dead
                dead_mask = np.maximum(dead_mask, mask)
        
        # 3.1 对活细胞区域：轻微增强亮度
        if live_mask.sum() > 0:
            live_mask_3d = np.stack([live_mask, live_mask, live_mask], axis=2)
            live_enhanced = np.clip(image_clahe.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
            image_clahe = np.where(live_mask_3d > 0, live_enhanced, image_clahe)
        
        # 3.2 对死细胞区域：增强对比度
        if dead_mask.sum() > 0:
            dead_mask_3d = np.stack([dead_mask, dead_mask, dead_mask], axis=2)
            dead_gray = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2GRAY)
            dead_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(dead_gray)
            dead_clahe_rgb = cv2.cvtColor(dead_clahe, cv2.COLOR_GRAY2RGB)
            image_clahe = np.where(dead_mask_3d > 0, dead_clahe_rgb, image_clahe)
        
        # ========== 第四阶段：简化特征融合 ==========
        # 融合CLAHE增强图像和边缘特征
        image_with_edges = np.clip(image_clahe.astype(np.float32) * 0.9 + 
                                  edges_rgb.astype(np.float32) * 0.1, 0, 255).astype(np.uint8)
        
        # 最终混合：保留原始信息
        image_final = (image_with_edges.astype(np.float32) * 0.85 + 
                      image.astype(np.float32) * 0.15).astype(np.uint8)
        
        # ========== 第五阶段：最终优化（简化） ==========
        # 轻微锐化
        gaussian = cv2.GaussianBlur(image_final, (3, 3), 1.0)
        unsharp = cv2.addWeighted(image_final, 1.3, gaussian, -0.3, 0)
        image_final = np.clip(unsharp, 0, 255).astype(np.uint8)
        
        return image_final
    
    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        json_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.json'))
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        original_size = image.shape[:2]
        
        # Resize图像以避免内存溢出和尺寸问题
        h, w = original_size
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # 确保尺寸能被32整除（某些模型要求）
            new_h = (new_h // 32) * 32
            new_w = (new_w // 32) * 32
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        else:
            # 即使不resize，也确保尺寸能被32整除
            h = (h // 32) * 32
            w = (w // 32) * 32
            if h != original_size[0] or w != original_size[1]:
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 加载标注
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 创建实例分割mask和类别标签
        # 计算resize的scale
        scale_h = h / original_size[0]
        scale_w = w / original_size[1]
        
        instance_masks = []
        instance_labels = []  # 0: live, 1: dead
        bboxes = []
        
        for shape in annotations.get('shapes', []):
            label = shape['label'].lower()
            if label not in ['live', 'dead']:
                continue
            
            points = np.array(shape['points'], dtype=np.float32)
            # 缩放点坐标
            points[:, 0] *= scale_w
            points[:, 1] *= scale_h
            points = points.astype(np.int32)
            
            # 创建mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            
            # 计算bbox
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            bbox = [x_min, y_min, x_max, y_max]
            
            instance_masks.append(mask)
            instance_labels.append(0 if label == 'live' else 1)
            bboxes.append(bbox)
        
        # 创建语义分割mask (用于某些模型)
        semantic_mask = np.zeros((h, w), dtype=np.int64)
        for i, (mask, label) in enumerate(zip(instance_masks, instance_labels)):
            semantic_mask[mask > 0] = label + 1  # 0: background, 1: live, 2: dead
        
        # ========== 特征工程和数据预处理 ==========
        # 针对活细胞和死细胞的特征工程
        image = self._apply_cell_specific_preprocessing(image, instance_masks, instance_labels)
        
        # 数据增强（仅在训练时）
        if self.split == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                # 翻转mask
                for mask in instance_masks:
                    mask[:] = cv2.flip(mask, 1)
                semantic_mask = cv2.flip(semantic_mask, 1)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                # 翻转mask
                for mask in instance_masks:
                    mask[:] = cv2.flip(mask, 0)
                semantic_mask = cv2.flip(semantic_mask, 0)
            
            # 注意：不进行旋转增强，因为会导致batch中图像尺寸不一致
            
            # ========== 针对活细胞和死细胞的针对性数据增强 ==========
            
            # 分析当前图像中活细胞和死细胞的比例
            live_pixels = (semantic_mask == 1).sum()
            dead_pixels = (semantic_mask == 2).sum()
            total_cell_pixels = live_pixels + dead_pixels
            
            if total_cell_pixels > 0:
                live_ratio = live_pixels / total_cell_pixels
            else:
                live_ratio = 0.5
            
            # 根据活细胞比例调整增强策略
            # 如果活细胞多，使用更温和的增强；如果死细胞多，使用更强的对比度增强
            
            # 1. 随机亮度调整（针对活细胞：更亮；死细胞：更暗）
            if random.random() > 0.3:
                if live_ratio > 0.6:  # 活细胞多
                    alpha = random.uniform(0.8, 1.3)  # 偏向增亮
                elif live_ratio < 0.4:  # 死细胞多
                    alpha = random.uniform(0.6, 1.1)  # 偏向变暗以增强对比
                else:
                    alpha = random.uniform(0.7, 1.3)
                image = np.clip(image * alpha, 0, 255).astype(np.uint8)
            
            # 2. 随机对比度调整（死细胞需要更强的对比度）
            if random.random() > 0.3:
                if live_ratio < 0.4:  # 死细胞多，增强对比度
                    beta = random.uniform(-20, 40)  # 更大的对比度调整范围
                else:
                    beta = random.uniform(-30, 30)
                image = np.clip(image + beta, 0, 255).astype(np.uint8)
            
            # 3. 随机饱和度调整（增强颜色特征）
            if random.random() > 0.5:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                saturation_factor = random.uniform(0.8, 1.3)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # 4. 随机CLAHE增强（自适应对比度）
            if random.random() > 0.4:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.0), tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                image = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
            
            # 5. 随机高斯噪声（模拟真实噪声）
            if random.random() > 0.5:
                noise = np.random.normal(0, random.uniform(3, 10), image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # 6. 随机Gamma校正
            if random.random() > 0.5:
                gamma = random.uniform(0.7, 1.3)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                image = cv2.LUT(image, table)
            
            # 7. 随机锐化（突出细胞边界）
            if random.random() > 0.6:
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]]) * random.uniform(0.1, 0.3)
                image = cv2.filter2D(image, -1, kernel)
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # 8. 随机颜色抖动（模拟不同光照条件）
            if random.random() > 0.6:
                # 在HSV空间进行颜色抖动
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180  # 色调
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.9, 1.1), 0, 255)  # 明度
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # 确保数组是连续的，避免负stride问题
        if not semantic_mask.flags['C_CONTIGUOUS']:
            semantic_mask = np.ascontiguousarray(semantic_mask)
        semantic_mask = torch.from_numpy(semantic_mask).long()
        
        return {
            'image': image,
            'instance_masks': instance_masks,
            'instance_labels': instance_labels,
            'bboxes': bboxes,
            'semantic_mask': semantic_mask,
            'image_id': img_name,
            'original_size': original_size
        }
    
    def get_coco_format(self, idx):
        """获取COCO格式的标注（用于Mask2Former）"""
        item = self.__getitem__(idx)
        
        # 转换为COCO格式
        coco_annotations = []
        for i, (mask, label, bbox) in enumerate(zip(
            item['instance_masks'], 
            item['instance_labels'], 
            item['bboxes']
        )):
            # 编码mask为RLE
            rle = mask_util.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            coco_annotations.append({
                'id': i,
                'category_id': label,
                'bbox': bbox,
                'segmentation': rle,
                'area': int(mask.sum()),
                'iscrowd': 0
            })
        
        return {
            'image': item['image'],
            'annotations': coco_annotations,
            'image_id': item['image_id'],
            'original_size': item['original_size']
        }


def collate_fn(batch):
    """自定义collate函数"""
    images = torch.stack([item['image'] for item in batch])
    return {
        'images': images,
        'batch_items': batch
    }

