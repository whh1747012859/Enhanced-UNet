"""
Detectron2数据集适配器
将现有的细胞检测数据集转换为Detectron2格式
"""
import os
import json
import numpy as np
from PIL import Image
import cv2
import torch
from detectron2.data import DatasetMapper, MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Instances
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import pycocotools.mask as mask_util
from typing import List, Dict
import random


def register_cell_dataset(data_dir: str, max_size: int = 640):
    """注册细胞检测数据集到Detectron2"""
    
    def get_cell_dicts(split: str):
        """获取指定split的数据字典列表"""
        # 获取所有图像文件
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        all_files.sort()
        
        # 划分数据集 (70% train, 15% val, 15% test)
        n_total = len(all_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        if split == 'train':
            files = all_files[:n_train]
        elif split == 'val':
            files = all_files[n_train:n_train+n_val]
        else:  # test
            files = all_files[n_train+n_val:]
        
        dataset_dicts = []
        print(f"正在处理 {split} 数据集，共 {len(files)} 张图像...")
        for idx, img_name in enumerate(files):
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(files)} 张图像...")
            
            img_path = os.path.join(data_dir, img_name)
            json_path = os.path.join(data_dir, img_name.replace('.jpg', '.json'))
            
            if not os.path.exists(json_path):
                continue
            
            # 加载图像获取原始尺寸（不进行resize，让mapper处理）
            # 只读取图像尺寸，不加载完整图像以节省内存
            try:
                with Image.open(img_path) as image:
                    original_h, original_w = image.size[1], image.size[0]  # PIL返回(width, height)，需要转换
            except Exception as e:
                print(f"Warning: Failed to open image {img_name}: {e}")
                continue
            
            # 记录原始尺寸，resize在mapper中进行
            h, w = original_h, original_w
            
            # 加载标注
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 创建Detectron2格式的标注
            objs = []
            for shape in annotations.get('shapes', []):
                label = shape['label'].lower()
                if label not in ['live', 'dead']:
                    continue
                
                points = np.array(shape['points'], dtype=np.float32)
                # 不进行缩放，保持原始坐标（mapper会处理resize和坐标变换）
                points = points.astype(np.int32)
                
                # 确保点在图像范围内
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                
                # 计算bbox (XYWH格式)
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                # 确保bbox有效
                if x_max <= x_min or y_max <= y_min:
                    continue
                bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                
                # 对于大图像，使用多边形格式而不是RLE（更快，节省内存）
                # 如果图像面积超过200万像素，使用多边形；否则使用RLE
                if h * w > 2000000:  # 大图像使用多边形格式
                    # 使用多边形格式（Detectron2支持）
                    segmentation = [points.flatten().tolist()]
                    # 估算面积（多边形面积）
                    area = float((x_max - x_min) * (y_max - y_min) * 0.8)  # 粗略估算
                else:
                    # 创建mask并编码为RLE（小图像）
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 1)
                    try:
                        rle = mask_util.encode(np.asfortranarray(mask))
                        if isinstance(rle['counts'], bytes):
                            rle['counts'] = rle['counts'].decode('utf-8')
                        segmentation = rle
                        area = float(mask.sum())
                    except Exception as e:
                        print(f"Warning: Failed to encode mask for {img_name}: {e}")
                        # 如果RLE失败，使用多边形
                        segmentation = [points.flatten().tolist()]
                        area = float((x_max - x_min) * (y_max - y_min) * 0.8)
                
                # category_id: 0=live, 1=dead (Detectron2不包括背景)
                category_id = 0 if label == 'live' else 1
                
                objs.append({
                    'bbox': bbox,
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': category_id,
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0
                })
            
            if len(objs) == 0:
                continue  # 跳过没有标注的图像
            
            record = {
                'file_name': img_path,
                'image_id': idx,
                'height': h,
                'width': w,
                'annotations': objs
            }
            dataset_dicts.append(record)
        
        print(f"注册 {split} 数据集: {len(dataset_dicts)} 张图像")
        return dataset_dicts
    
    # 注册数据集
    for split in ['train', 'val', 'test']:
        DatasetCatalog.register(f'cell_{split}', lambda s=split: get_cell_dicts(s))
        MetadataCatalog.get(f'cell_{split}').set(
            thing_classes=['live', 'dead'],
            evaluator_type='coco'
        )
    
    print("已注册Detectron2数据集: cell_train, cell_val, cell_test")


class CellDatasetMapper(DatasetMapper):
    """细胞检测数据集Mapper，适配Detectron2"""
    
    def __init__(self, cfg, is_train: bool = True):
        """
        Args:
            cfg: Detectron2配置
            is_train: 是否为训练模式
        """
        # 获取图像尺寸
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
        
        # 构建数据增强pipeline
        augmentations = []
        if is_train:
            # 训练时的数据增强
            # ResizeShortestEdge可以接受单个值或范围
            # 如果min_size是元组/列表且长度为2，表示范围采样
            if isinstance(min_size, (list, tuple)) and len(min_size) == 2:
                # 范围采样：从[min_size[0], min_size[1]]中随机选择
                augmentations.append(T.ResizeShortestEdge(min_size, max_size))
            elif isinstance(min_size, (list, tuple)) and len(min_size) == 1:
                # 单元素元组，转换为整数
                augmentations.append(T.ResizeShortestEdge(min_size[0], max_size))
            else:
                # 单个整数，固定尺寸
                augmentations.append(T.ResizeShortestEdge(min_size, max_size))
            
            # Detectron2的RandomFlip不支持同时使用水平和垂直翻转
            # 只使用水平翻转（对细胞检测任务足够，且更常见）
            augmentations.append(T.RandomFlip(horizontal=True, prob=0.5))
            # 注意：不添加旋转，因为会导致尺寸不一致
            
            # 使用Detectron2内置的颜色增强方法
            # RandomBrightness: 亮度调整 (0.9-1.1 表示90%-110%的亮度)
            augmentations.append(
                T.RandomApply(
                    T.RandomBrightness(intensity_min=0.8, intensity_max=1.2),
                    prob=0.5
                )
            )
            # RandomContrast: 对比度调整 (0.9-1.1 表示90%-110%的对比度)
            augmentations.append(
                T.RandomApply(
                    T.RandomContrast(intensity_min=0.8, intensity_max=1.2),
                    prob=0.5
                )
            )
        else:
            # 测试时只resize
            augmentations.append(T.ResizeShortestEdge(min_size, max_size))
        
        super().__init__(
            cfg,
            is_train=is_train,
            augmentations=augmentations,
            image_format=cfg.INPUT.FORMAT,
            use_instance_mask=cfg.MODEL.MASK_ON,
            instance_mask_format=cfg.INPUT.MASK_FORMAT,
            use_keypoint=False,
            recompute_boxes=cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads"
        )
    
    def __call__(self, dataset_dict):
        """
        将数据集字典转换为模型输入格式
        """
        dataset_dict = dataset_dict.copy()
        
        # 加载图像
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # 应用数据增强（resize等）
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        # 更新图像尺寸（在resize后）
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]
        
        # 将图像转换为tensor格式（CHW格式，float32）
        # 这是Detectron2模型要求的格式
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        dataset_dict["image"] = image_tensor
        
        # 处理标注
        if not self.is_train:
            # 测试时不需要标注
            dataset_dict.pop("annotations", None)
            return dataset_dict
        
        # 应用变换到标注
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image.shape[:2]
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        
        # 创建Instances对象
        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.instance_mask_format
        )
        
        # 应用额外的变换（如果有）
        instances = utils.filter_empty_instances(instances)
        dataset_dict["instances"] = instances
        
        return dataset_dict

