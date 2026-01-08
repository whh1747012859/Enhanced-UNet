import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
import os
import json
from tqdm import tqdm
import cv2
from scipy import ndimage
from skimage import measure
from skimage.feature import peak_local_max
import pycocotools.mask as mask_util

from dataset import CellDataset, collate_fn
from models import get_model
from metrics import (
    calculate_semantic_metrics,
    calculate_instance_metrics,
    calculate_coco_metrics,
    calculate_viability_metrics
)
from visualization import Visualizer


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡 - 确保三类都能被识别"""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=None, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 可以是标量或每个类别的权重列表
        self.gamma = gamma
        self.ignore_index = ignore_index  # 不忽略任何类别
        self.class_weights = class_weights  # 类别权重
        
    def forward(self, inputs, targets):
        # 不忽略任何类别，确保三类都能被识别
        # 如果ignore_index为None，则不传递该参数
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none', weight=self.class_weights)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        
        # 如果alpha是列表，需要根据target选择对应的alpha
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                # 为每个像素选择对应的alpha
                alpha_t = torch.zeros_like(ce_loss)
                for i, alpha_val in enumerate(self.alpha):
                    if self.ignore_index is None or i != self.ignore_index:
                        alpha_t[targets == i] = alpha_val
                focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
            else:
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class Trainer:
    """训练器类"""
    
    def __init__(self, model, device, model_name, total_epochs: int = 50):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.total_epochs = max(1, total_epochs)
        
        # 计算类别权重 - 确保三类都能被正确识别
        # 背景、活细胞、死细胞都需要被正确识别，使用平衡但偏向细胞的权重
        class_weights = torch.tensor([1.0, 20.0, 10.0]).to(device)  # [background, live, dead] - 大幅提升细胞权重，但背景也要识别
        alpha = [1.0, 8.0, 5.0]  # Focal Loss的alpha，确保三类都有足够权重
        
        # 使用组合损失：Focal Loss + Dice Loss + Boundary Loss + Tversky Loss
        # Focal Loss处理类别不平衡，Dice Loss提升分割精度，Tversky Loss提升边界精度
        self.focal_loss = FocalLoss(alpha=alpha, gamma=5.0, ignore_index=None, class_weights=class_weights)  # 不忽略背景
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)  # 带权重的CE Loss，不忽略背景
        # enhanced_unet使用更强的损失权重以提升性能
        if model_name == 'enhanced_unet':
            self.dice_loss_weight = 2.5 
            self.focal_loss_weight = 2.5  
            self.tversky_loss_weight = 1.0  
            self.aux_branch_weights = {'unetpp': 0.6, 'deeplab': 0.5} 
            self.consistency_weight = 0.4 
        elif model_name == 'fcn':
            
            self.dice_loss_weight = 1.0
            self.focal_loss_weight = 1.0
            self.tversky_loss_weight = 0.3
            self.aux_branch_weights = {}
            self.consistency_weight = 0.0
        elif model_name == 'linknet':
            
            self.dice_loss_weight = 0.8
            self.focal_loss_weight = 0.8
            self.tversky_loss_weight = 0.2
            self.aux_branch_weights = {}
            self.consistency_weight = 0.0
        else:
            self.dice_loss_weight = 1.5
            self.focal_loss_weight = 1.5
            self.tversky_loss_weight = 0.5
            self.aux_branch_weights = {}
            self.consistency_weight = 0.0
        
        # 使用warmup + cosine annealing with restarts
        # 优化学习率策略，确保充分训练
    
        if model_name == 'enhanced_unet':
            base_lr = 4e-3  
        elif model_name == 'fcn':
            base_lr = 1e-3  
        elif model_name == 'linknet':
            base_lr = 8e-4 
        else:
            base_lr = 2e-3
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4, betas=(0.9, 0.999))
        # Warmup scheduler
        self.warmup_epochs = max(1, min(5, self.total_epochs // 6))  # 根据总epoch动态调整warmup
        # 使用CosineAnnealingWarmRestarts来提升性能（在warmup后使用）
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(10, self.total_epochs // 3),
            T_mult=2,
            eta_min=1e-7
        )
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.001, end_factor=1.0, total_iters=self.warmup_epochs
        )
    
    def dice_loss(self, pred, target, num_classes=3):
        """计算Dice Loss（带类别权重，包含背景）"""
        pred_softmax = F.softmax(pred, dim=1)
        dice_losses = []
        
        # 类别权重：确保三类都能被正确识别
        class_weights = [1.0, 15.0, 8.0]  # [background, live, dead] - 三类都有权重
        
        for class_id in range(num_classes):  # 包含background (0)
            pred_class = pred_softmax[:, class_id]
            target_class = (target == class_id).float()
            
            # 计算每个样本的dice
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
            
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = 1.0 - dice
            
            # 应用类别权重
            weighted_loss = dice_loss * class_weights[class_id]
            dice_losses.append(weighted_loss.mean())
        
        return sum(dice_losses) / len(dice_losses) if dice_losses else torch.tensor(0.0).to(pred.device)
    
    def tversky_loss(self, pred, target, num_classes=3, alpha=0.7):
        """计算Tversky Loss（提升边界精度）"""
        pred_softmax = F.softmax(pred, dim=1)
        tversky_losses = []
        
        class_weights = [1.0, 12.0, 6.0]  # [background, live, dead]
        
        for class_id in range(num_classes):
            pred_class = pred_softmax[:, class_id]
            target_class = (target == class_id).float()
            
            # Tversky系数
            tp = (pred_class * target_class).sum(dim=(1, 2))
            fp = (pred_class * (1 - target_class)).sum(dim=(1, 2))
            fn = ((1 - pred_class) * target_class).sum(dim=(1, 2))
            
            tversky = (tp + 1e-6) / (tp + alpha * fp + (1 - alpha) * fn + 1e-6)
            tversky_loss = 1.0 - tversky
            
            weighted_loss = tversky_loss * class_weights[class_id]
            tversky_losses.append(weighted_loss.mean())
        
        return sum(tversky_losses) / len(tversky_losses) if tversky_losses else torch.tensor(0.0).to(pred.device)
    
    def _compute_combined_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """统一的组合损失计算，便于主分支和辅助分支复用"""
        if target.dtype != torch.long:
            target = target.long()
        target = target.to(self.device)
        logits_batch = logits.unsqueeze(0)  # [1, C, H, W]
        target_batch = target.unsqueeze(0)  # [1, H, W]
        
        focal_loss = self.focal_loss(logits_batch, target_batch)
        dice_loss = self.dice_loss(logits_batch, target_batch, num_classes=3)
        tversky_loss = self.tversky_loss(logits_batch, target_batch, num_classes=3)
        
        return (self.focal_loss_weight * focal_loss +
                self.dice_loss_weight * dice_loss +
                self.tversky_loss_weight * tversky_loss)
    
    def _apply_auxiliary_supervision(self, aux_outputs: Dict[str, torch.Tensor],
                                     sample_idx: int,
                                     target_mask: torch.Tensor,
                                     fused_logits: torch.Tensor) -> torch.Tensor:
        """为增强UNet的辅助分支添加深度监督和一致性约束"""
        if not aux_outputs or not self.aux_branch_weights:
            return torch.tensor(0.0, device=self.device)
        
        total_aux_loss = torch.tensor(0.0, device=self.device)
        fused_probs = None
        if self.consistency_weight > 0:
            fused_probs = F.softmax(fused_logits.unsqueeze(0), dim=1)
        
        for branch_name, weight in self.aux_branch_weights.items():
            branch_tensor = aux_outputs.get(branch_name)
            if branch_tensor is None or branch_tensor.shape[0] <= sample_idx:
                continue
            
            branch_logits = branch_tensor[sample_idx]
            if branch_logits.shape[1:] != target_mask.shape:
                branch_logits = F.interpolate(
                    branch_logits.unsqueeze(0),
                    size=target_mask.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            branch_loss = self._compute_combined_loss(branch_logits, target_mask)
            total_aux_loss = total_aux_loss + weight * branch_loss
            
            if fused_probs is not None:
                branch_probs = F.softmax(branch_logits.unsqueeze(0), dim=1)
                consistency = F.mse_loss(branch_probs, fused_probs)
                total_aux_loss = total_aux_loss + weight * self.consistency_weight * consistency
        
        return total_aux_loss
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f'Training {self.model_name}'):
            images = batch['images'].to(self.device)
            semantic_masks = [item['semantic_mask'] for item in batch['batch_items']]
            
            # 调整mask大小以匹配模型输出
            self.optimizer.zero_grad()
            
            # 确保图像尺寸能被32整除
            batch_size, channels, h, w = images.shape
            h_pad = (32 - h % 32) % 32
            w_pad = (32 - w % 32) % 32
            if h_pad > 0 or w_pad > 0:
                images = F.pad(images, (0, w_pad, 0, h_pad), mode='reflect')
            
            outputs = self.model(images)
            aux_outputs = None
            if self.model_name == 'enhanced_unet' and hasattr(self.model, 'get_aux_outputs'):
                aux_outputs = self.model.get_aux_outputs()
            
            # 计算损失
            loss = 0.0
            for i in range(batch_size):
                gt_mask = semantic_masks[i]  # 应该是 [H, W]
                
                # 调试：打印原始gt_mask的形状
                if i == 0 and len(gt_mask.shape) != 2:
                    print(f"Warning: gt_mask shape is {gt_mask.shape}, expected [H, W]")
                
                # 确保gt_mask是2D的
                if len(gt_mask.shape) != 2:
                    gt_mask = gt_mask.squeeze()
                    if len(gt_mask.shape) != 2:
                        raise ValueError(f"gt_mask should be 2D after squeeze, got {gt_mask.shape}")
                
                # 获取当前图像的实际尺寸（考虑padding后）
                h_orig, w_orig = gt_mask.shape
                h_new = h_orig + h_pad
                w_new = w_orig + w_pad
                
                # 对gt_mask也进行padding，确保维度正确 [H, W]
                if h_pad > 0 or w_pad > 0:
                    # F.pad对2D tensor的格式是 (pad_left, pad_right, pad_top, pad_bottom)
                    # 需要先unsqueeze成4D: [1, 1, H, W]，然后pad，再squeeze回2D
                    gt_mask_4d = gt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    gt_mask_padded_4d = F.pad(gt_mask_4d, 
                                            (0, w_pad, 0, h_pad), 
                                            mode='constant', value=0)  # [1, 1, H+h_pad, W+w_pad]
                    gt_mask_padded = gt_mask_padded_4d.squeeze(0).squeeze(0)  # [H+h_pad, W+w_pad]
                else:
                    gt_mask_padded = gt_mask
                
                # 确保gt_mask_padded是2D的 [H, W]
                while len(gt_mask_padded.shape) > 2:
                    gt_mask_padded = gt_mask_padded.squeeze()
                if len(gt_mask_padded.shape) != 2:
                    raise ValueError(f"gt_mask_padded should be 2D, got shape {gt_mask_padded.shape}")
                
                # output是[B, C, H, W]，取第i个样本: [C, H_out, W_out]
                output_i = outputs[i]  # [C, H_out, W_out]
                
                # 确保output_i是3D的 [C, H, W]
                if len(output_i.shape) != 3:
                    raise ValueError(f"output_i should be 3D [C, H, W], got {output_i.shape}")
                
                # 确保output_i和gt_mask_padded的空间尺寸匹配
                if output_i.shape[1:] != gt_mask_padded.shape:
                    output_i = F.interpolate(output_i.unsqueeze(0), 
                                            size=(h_new, w_new), 
                                            mode='bilinear', 
                                            align_corners=False).squeeze(0)
                
                # 最终验证维度
                if len(output_i.shape) != 3:
                    raise ValueError(f"output_i should be 3D [C, H, W], got {output_i.shape}")
                if len(gt_mask_padded.shape) != 2:
                    raise ValueError(f"gt_mask_padded should be 2D [H, W], got {gt_mask_padded.shape}")
                if output_i.shape[1:] != gt_mask_padded.shape:
                    raise ValueError(f"Size mismatch: output_i {output_i.shape[1:]} vs gt_mask_padded {gt_mask_padded.shape}")
                
                # CrossEntropyLoss期望: input [N, C, H, W] 或 [C, H, W], target [N, H, W] 或 [H, W]
                # output_i: [C, H, W], gt_mask_padded: [H, W]
                # 确保它们在同一个设备上
                gt_mask_padded = gt_mask_padded.to(self.device).long()
                
                combined_loss = self._compute_combined_loss(output_i, gt_mask_padded)
                loss += combined_loss
                
                if aux_outputs:
                    aux_loss = self._apply_auxiliary_supervision(
                        aux_outputs,
                        i,
                        gt_mask_padded,
                        output_i
                    )
                    loss += aux_loss
            
            loss = loss / batch_size
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 学习率调度（warmup + cosine annealing）
        if hasattr(self, 'warmup_scheduler') and hasattr(self, 'warmup_epochs'):
            # 这里需要从外部传入epoch数，暂时简化处理
            # 实际应该在train_and_evaluate中处理
            pass
        
        return total_loss / len(dataloader)


class Evaluator:
    """评估器类"""
    
    def __init__(self, model, device, model_name):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.enable_tta = (model_name == 'enhanced_unet')
    
    def _prepare_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """图像预处理（用于增强UNet及TTA）"""
        if image.device.type == 'cuda':
            image_cpu = image.cpu()
            max_val = image_cpu.max().item()
            image = image.to(self.device)
        else:
            max_val = image.max().item()
        
        if max_val <= 1.0:
            image_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            image_np = image.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        
        # CLAHE增强
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
        image_np = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 轻微锐化
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * 0.15
        image_np = cv2.filter2D(image_np, -1, kernel)
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        return image_tensor
    
    def _run_model_single(self, image: torch.Tensor) -> torch.Tensor:
        """运行单次模型推理"""
        h, w = image.shape[1:]
        h_pad = (32 - h % 32) % 32
        w_pad = (32 - w % 32) % 32
        h_orig, w_orig = h, w
        
        if h_pad > 0 or w_pad > 0:
            image_padded = F.pad(image.unsqueeze(0), (0, w_pad, 0, h_pad), mode='reflect')
            h, w = h_orig + h_pad, w_orig + w_pad
        else:
            image_padded = image.unsqueeze(0)
        
        output = self.model(image_padded)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        probs = F.softmax(output[0], dim=0)
        
        if h_pad > 0 or w_pad > 0:
            probs = probs[:, :h_orig, :w_orig]
        
        return probs
    
    def _run_tta_inference(self, image: torch.Tensor) -> torch.Tensor:
        """测试时增强（TTA）推理"""
        base_probs = self._run_model_single(image)
        if not self.enable_tta:
            return base_probs
        
        tta_probs = [base_probs]
        h, w = image.shape[1:]
        
        # 水平翻转
        img_hflip = torch.flip(image, dims=[2])
        prob_hflip = self._run_model_single(img_hflip)
        prob_hflip = torch.flip(prob_hflip, dims=[2])
        tta_probs.append(prob_hflip)
        
        # 垂直翻转
        img_vflip = torch.flip(image, dims=[1])
        prob_vflip = self._run_model_single(img_vflip)
        prob_vflip = torch.flip(prob_vflip, dims=[1])
        tta_probs.append(prob_vflip)
        
        # 多尺度（缩放0.75和1.25）
        for scale in [0.75, 1.25]:
            scaled = F.interpolate(
                image.unsqueeze(0), scale_factor=scale,
                mode='bilinear', align_corners=False
            ).squeeze(0)
            probs_scale = self._run_model_single(scaled)
            probs_scale = F.interpolate(
                probs_scale.unsqueeze(0), size=(h, w),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            tta_probs.append(probs_scale)
        
        return torch.stack(tta_probs, dim=0).mean(dim=0)
    
    def _convert_probs_to_mask(self, probs: torch.Tensor, h_pad: int = 0, w_pad: int = 0,
                               h_orig: int = None, w_orig: int = None) -> np.ndarray:
        """将概率图转换为语义mask
        
        使用智能阈值策略：
        1. 优先使用argmax作为基础预测（捕获更多细胞）
        2. 只对低置信度的预测进行过滤（标记为背景）
        3. 确保细胞概率明显高于背景时才接受
        """
        h, w = probs.shape[1:]
        
        bg_prob = probs[0]
        live_prob = probs[1]
        dead_prob = probs[2]
        
        # 首先使用argmax作为基础预测（这样可以捕获更多细胞）
        pred_mask = torch.argmax(probs, dim=0)
        
        # 计算每个像素的最大概率（置信度）
        max_prob = torch.max(probs, dim=0)[0]
        
        # 使用平衡的阈值策略（既要避免预测过多，也要确保预测足够）
        # 活细胞：如果argmax是活细胞，但置信度太低或不够明显高于背景，则标记为背景
        live_low_conf = (pred_mask == 1) & (
            (live_prob < 0.42) |  # 绝对阈值：活细胞概率至少0.42（稍微放宽，确保预测足够）
            (live_prob <= bg_prob * 1.15)  # 相对阈值：活细胞概率至少是背景的1.15倍（稍微放宽）
        )
        pred_mask[live_low_conf] = 0
        
        # 死细胞：使用非常严格的过滤（因为死细胞经常被过度预测）
        # 如果argmax是死细胞，但置信度不够高或背景概率也很高，则标记为背景
        dead_low_conf = (pred_mask == 2) & (
            (dead_prob < 0.5) |  # 绝对阈值：死细胞概率至少0.5（非常严格）
            (dead_prob <= bg_prob * 1.3) |  # 相对阈值：死细胞概率至少是背景的1.3倍（非常严格）
            (bg_prob > 0.3) |  # 如果背景概率也很高（>0.3），则标记为背景（避免过度预测）
            (live_prob > dead_prob * 0.9)  # 如果活细胞概率也很高（接近死细胞），则标记为背景
        )
        pred_mask[dead_low_conf] = 0
        
        # 对于argmax是背景但细胞概率足够高的区域，可以考虑标记为细胞
        # 稍微放宽阈值，确保捕获更多细胞
        bg_but_high_live = (pred_mask == 0) & \
                          (live_prob > 0.42) & \
                          (live_prob > bg_prob * 1.15) & \
                          (live_prob > dead_prob * 1.05)
        pred_mask[bg_but_high_live] = 1
        
        # 死细胞：使用更严格的条件（避免过度预测）
        bg_but_high_dead = (pred_mask == 0) & \
                          (dead_prob > 0.5) & \
                          (dead_prob > bg_prob * 1.3) & \
                          (dead_prob > live_prob * 1.1) & \
                          (bg_prob < 0.3) & \
                          (~bg_but_high_live)  # 排除已标记为活细胞的区域
        pred_mask[bg_but_high_dead] = 2
        
        # 处理活细胞和死细胞的重叠：如果某个区域同时满足活细胞和死细胞条件，选择概率更高的
        # 这种情况可能发生在bg_but_high_live和bg_but_high_dead都满足的区域
        # 但由于我们使用了(~bg_but_high_live)来排除，这种情况应该很少
        # 为了安全起见，再次检查：如果某个区域被标记为活细胞，但死细胞概率明显更高，则改为死细胞
        live_but_dead_higher = (pred_mask == 1) & (dead_prob > live_prob * 1.15) & (dead_prob > 0.45)
        pred_mask[live_but_dead_higher] = 2
        
        # 反之亦然：如果某个区域被标记为死细胞，但活细胞概率明显更高，则改为活细胞
        dead_but_live_higher = (pred_mask == 2) & (live_prob > dead_prob * 1.15) & (live_prob > 0.42)
        pred_mask[dead_but_live_higher] = 1
        
        # 最终清理：如果最大概率太低（< 0.3），强制标记为背景（可能是噪声）
        very_low_conf = max_prob < 0.3
        pred_mask[very_low_conf] = 0
        
        # 额外的过滤：如果细胞像素数过多，进一步收紧阈值
        pred_mask_np = pred_mask.cpu().numpy()
        live_pixel_ratio = (pred_mask_np == 1).sum() / (h * w)
        dead_pixel_ratio = (pred_mask_np == 2).sum() / (h * w)
        
        # 活细胞过滤：如果活细胞超过图像的50%，进一步收紧阈值（提高阈值，避免过度过滤）
        if live_pixel_ratio > 0.5:
            live_mask = (pred_mask_np == 1)
            # 只保留高置信度的活细胞预测
            live_high_conf = (live_prob.cpu().numpy() > 0.5) & \
                            (live_prob.cpu().numpy() > bg_prob.cpu().numpy() * 1.3) & \
                            (bg_prob.cpu().numpy() < 0.3)
            pred_mask_np[live_mask & (~live_high_conf)] = 0
        
        # 死细胞过滤：如果死细胞超过图像的15%，进一步收紧阈值
        if dead_pixel_ratio > 0.15:
            # 死细胞预测过多，进一步收紧阈值
            # 只保留高置信度的死细胞预测
            dead_mask = (pred_mask_np == 2)
            # 根据比例动态调整阈值：比例越高，阈值越严格
            if dead_pixel_ratio > 0.4:
                # 非常严重：使用最严格的阈值
                dead_high_conf = (dead_prob.cpu().numpy() > 0.65) & \
                                (dead_prob.cpu().numpy() > bg_prob.cpu().numpy() * 1.6) & \
                                (bg_prob.cpu().numpy() < 0.2) & \
                                (live_prob.cpu().numpy() < dead_prob.cpu().numpy() * 0.7)
            elif dead_pixel_ratio > 0.25:
                # 严重：使用严格阈值
                dead_high_conf = (dead_prob.cpu().numpy() > 0.6) & \
                                (dead_prob.cpu().numpy() > bg_prob.cpu().numpy() * 1.5) & \
                                (bg_prob.cpu().numpy() < 0.25) & \
                                (live_prob.cpu().numpy() < dead_prob.cpu().numpy() * 0.8)
            else:
                # 中等：使用较严格阈值
                dead_high_conf = (dead_prob.cpu().numpy() > 0.55) & \
                                (dead_prob.cpu().numpy() > bg_prob.cpu().numpy() * 1.4) & \
                                (bg_prob.cpu().numpy() < 0.25)
            pred_mask_np[dead_mask & (~dead_high_conf)] = 0
        
        if h_pad > 0 or w_pad > 0:
            pred_mask_np = pred_mask_np[:h_orig, :w_orig]
        
        return pred_mask_np
    
    def predict_semantic_mask(self, image: torch.Tensor) -> np.ndarray:
        """预测语义分割mask"""
        try:
            self.model.eval()
            with torch.no_grad():
                return self._predict_semantic_mask_impl(image)
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'cuda' in str(e).lower():
                print(f"CUDA error in predict_semantic_mask: {e}")
                print("Attempting CPU fallback...")
                try:
                    # 尝试在CPU上运行
                    original_device = self.device
                    self.device = 'cpu'
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to('cpu')
                    image_cpu = image.cpu() if image.is_cuda else image
                    result = self._predict_semantic_mask_impl(image_cpu)
                    # 恢复设备
                    self.device = original_device
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(original_device)
                    return result
                except Exception as e2:
                    print(f"CPU fallback also failed: {e2}")
                    # 返回全零mask
                    h, w = image.shape[1:] if len(image.shape) == 3 else (image.shape[0], image.shape[1])
                    return np.zeros((h, w), dtype=np.int64)
            else:
                raise
    
    def _predict_semantic_mask_impl(self, image: torch.Tensor) -> np.ndarray:
        """预测语义分割mask的实现"""
        with torch.no_grad():
                if self.model_name == 'enhanced_unet':
                    image_tensor = self._prepare_image_tensor(image)
                    probs = self._run_tta_inference(image_tensor)
                    return self._convert_probs_to_mask(probs)
                
                # 确保图像在正确的设备上，先移到CPU进行安全检查
                if image.device.type == 'cuda':
                    image_cpu = image.cpu()
                    max_val = image_cpu.max().item()
                    image = image.to(self.device)
                else:
                    max_val = image.max().item()
                
                if max_val <= 1.0:
                    image_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    image_np = image.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                
                lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel_enhanced = clahe.apply(l_channel)
                lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
                image_np = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
                
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]]) * 0.15
                image_np = cv2.filter2D(image_np, -1, kernel)
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                
                image = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
                
                h, w = image.shape[1:]
                h_pad = (32 - h % 32) % 32
                w_pad = (32 - w % 32) % 32
                h_orig, w_orig = h, w
                
                if h_pad > 0 or w_pad > 0:
                    image_padded = F.pad(image.unsqueeze(0), (0, w_pad, 0, h_pad), mode='reflect')
                    h, w = h_orig + h_pad, w_orig + w_pad
                else:
                    image_padded = image.unsqueeze(0)
                
                output = self.model(image_padded)
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
                probs = F.softmax(output[0], dim=0)
                
                return self._convert_probs_to_mask(probs, h_pad, w_pad, h_orig, w_orig)
    
    def semantic_to_instances(self, semantic_mask: np.ndarray, min_area: int = 3) -> tuple:
        """将语义分割mask转换为实例分割
        
        核心策略：每个小的连通区域就是一个细胞
        - 优先使用连通组件分析，将每个连通区域视为一个独立细胞
        - 对于特别大的连通区域（可能是多个细胞重叠），使用分水岭算法进一步分离
        """
        instance_masks = []
        instance_labels = []
        instance_scores = []
        
        for class_id in [1, 2]:  # 1: live, 2: dead
            class_mask = (semantic_mask == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
            
            # 轻微的去噪操作，但不过度连接细胞
            # 只去除单像素噪声，保持细胞之间的间隙
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel_tiny)
            
            # 对于大的连通区域，使用形态学腐蚀来分离（比分水岭快得多）
            # 首先使用连通组件分析
            markers, num_labels = measure.label(class_mask, connectivity=2, return_num=True)
            
            # 处理每个连通区域
            final_markers = np.zeros_like(markers, dtype=np.int32)
            next_label = 1
            
            # 定义大区域阈值：如果区域面积超过200像素，就需要分离（降低阈值，更早分离）
            # 因为细胞通常很小，超过200像素的区域很可能包含多个细胞
            large_region_threshold = 200
            
            for label_id in range(1, num_labels + 1):
                region_mask = (markers == label_id).astype(np.uint8)
                area = region_mask.sum()
                
                if area < large_region_threshold:
                    # 小区域：直接作为一个细胞
                    final_markers[region_mask > 0] = next_label
                    next_label += 1
                else:
                    # 大区域：使用更积极的形态学腐蚀来分离
                    # 使用较小的腐蚀核，但更多迭代，以便更好地分离紧密连接的细胞
                    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 使用较小的核，更精细的分离
                    # 根据区域大小决定腐蚀迭代次数（更积极的分离）
                    erode_iterations = max(2, min(area // 1000, 8))  # 2-8次迭代（降低分母，更早分离）
                    eroded = cv2.erode(region_mask, kernel_erode, iterations=erode_iterations)
                    
                    # 对腐蚀后的区域进行连通组件分析
                    sub_markers, sub_num = measure.label(eroded, connectivity=2, return_num=True)
                    
                    if sub_num > 1:
                        # 找到了多个子区域，使用膨胀恢复，但保持分离
                        for sub_label in range(1, sub_num + 1):
                            sub_region = (sub_markers == sub_label).astype(np.uint8)
                            # 膨胀恢复，但限制在原始区域内
                            dilated = cv2.dilate(sub_region, kernel_erode, iterations=erode_iterations)
                            dilated = cv2.bitwise_and(dilated, region_mask)
                            
                            # 如果分离后的区域仍然很大，继续分离
                            if dilated.sum() > large_region_threshold:
                                # 递归分离：对大的子区域再次进行分离
                                kernel_erode2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                eroded2 = cv2.erode(dilated, kernel_erode2, iterations=2)
                                sub_markers2, sub_num2 = measure.label(eroded2, connectivity=2, return_num=True)
                                
                                if sub_num2 > 1:
                                    for sub_label2 in range(1, sub_num2 + 1):
                                        sub_region2 = (sub_markers2 == sub_label2).astype(np.uint8)
                                        dilated2 = cv2.dilate(sub_region2, kernel_erode2, iterations=2)
                                        dilated2 = cv2.bitwise_and(dilated2, dilated)
                                        
                                        if dilated2.sum() >= min_area:
                                            final_markers[dilated2 > 0] = next_label
                                            next_label += 1
                                else:
                                    # 递归分离失败，使用原区域（但会被面积过滤）
                                    if dilated.sum() >= min_area:
                                        final_markers[dilated > 0] = next_label
                                        next_label += 1
                            else:
                                # 分离后的区域大小合适
                                if dilated.sum() >= min_area:
                                    final_markers[dilated > 0] = next_label
                                    next_label += 1
                    else:
                        # 只找到一个子区域，尝试更激进的分离
                        # 使用多尺度分离策略：先用小核多次腐蚀，再用大核
                        # 策略1：使用小核多次腐蚀（更精细的分离）
                        kernel_erode_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        eroded_small = region_mask.copy()
                        separation_success = False
                        for _ in range(3):
                            eroded_small = cv2.erode(eroded_small, kernel_erode_small, iterations=1)
                            sub_markers_small, sub_num_small = measure.label(eroded_small, connectivity=2, return_num=True)
                            if sub_num_small > 1:
                                # 找到了多个子区域
                                for sub_label_small in range(1, sub_num_small + 1):
                                    sub_region_small = (sub_markers_small == sub_label_small).astype(np.uint8)
                                    # 膨胀恢复
                                    dilated_small = cv2.dilate(sub_region_small, kernel_erode_small, iterations=3)
                                    dilated_small = cv2.bitwise_and(dilated_small, region_mask)
                                    
                                    if dilated_small.sum() >= min_area:
                                        final_markers[dilated_small > 0] = next_label
                                        next_label += 1
                                separation_success = True
                                break  # 成功分离，退出循环
                        
                        # 策略2：如果小核分离失败，使用更大的核
                        if not separation_success:
                            kernel_erode2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            eroded2 = cv2.erode(region_mask, kernel_erode2, iterations=3)
                            sub_markers2, sub_num2 = measure.label(eroded2, connectivity=2, return_num=True)
                            
                            if sub_num2 > 1:
                                # 找到了多个子区域
                                for sub_label2 in range(1, sub_num2 + 1):
                                    sub_region2 = (sub_markers2 == sub_label2).astype(np.uint8)
                                    dilated2 = cv2.dilate(sub_region2, kernel_erode2, iterations=3)
                                    dilated2 = cv2.bitwise_and(dilated2, region_mask)
                                    
                                    if dilated2.sum() >= min_area:
                                        final_markers[dilated2 > 0] = next_label
                                        next_label += 1
                            else:
                                # 仍然只找到一个子区域，直接使用原区域（会被面积过滤）
                                if region_mask.sum() >= min_area:
                                    final_markers[region_mask > 0] = next_label
                                    next_label += 1
            
            # 提取每个最终标记的实例
            num_final_labels = final_markers.max()
            
            # 根据类别设置不同的面积阈值
            # 活细胞通常较小，死细胞可能较大
            # 降低最小面积阈值，以捕获更多小细胞
            if class_id == 1:  # live
                min_area_threshold = max(3, min_area)  # 活细胞最小3像素（降低阈值，捕获更多小细胞）
                max_area_threshold = 1500  # 活细胞最大1500像素（适当降低，避免异常大的区域）
            else:  # dead
                min_area_threshold = max(5, min_area)  # 死细胞最小5像素（降低阈值，捕获更多小细胞）
                max_area_threshold = 1500  # 死细胞最大1500像素（适当降低，避免异常大的区域）
            
            # 统计信息（用于调试）
            filtered_small = 0
            filtered_large = 0
            
            for label_id in range(1, num_final_labels + 1):
                instance_mask = (final_markers == label_id).astype(np.uint8)
                area = instance_mask.sum()
                
                # 面积过滤：既不能太小（噪声），也不能太大（错误预测）
                if area < min_area_threshold:
                    filtered_small += 1
                    continue
                if area > max_area_threshold:
                    filtered_large += 1
                    continue
                
                if min_area_threshold <= area <= max_area_threshold:
                    # 计算置信度分数
                    try:
                        contours = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                        if len(contours) > 0:
                            perimeter = cv2.arcLength(contours[0], True)
                            if perimeter > 0:
                                compactness = 4 * np.pi * area / (perimeter ** 2)
                            else:
                                compactness = 0.5
                        else:
                            compactness = 0.5
                    except:
                        compactness = 0.5
                    
                    # 归一化面积分数（假设平均细胞面积约为50-200像素）
                    area_score = min(area / 150.0, 1.0)
                    # 置信度：面积权重更高，因为面积更可靠
                    confidence = 0.7 * area_score + 0.3 * compactness
                    
                    instance_masks.append(instance_mask)
                    instance_labels.append(class_id - 1)  # 0: live, 1: dead
                    instance_scores.append(confidence)
            
            # 限制实例数量，避免过多的小实例影响速度
            # 如果实例数量过多（>500），只保留置信度最高的前500个
            if len(instance_masks) > 500:
                # 按置信度排序
                sorted_indices = sorted(range(len(instance_scores)), key=lambda i: instance_scores[i], reverse=True)
                # 只保留前500个
                instance_masks = [instance_masks[i] for i in sorted_indices[:500]]
                instance_labels = [instance_labels[i] for i in sorted_indices[:500]]
                instance_scores = [instance_scores[i] for i in sorted_indices[:500]]
        
        return instance_masks, instance_labels, instance_scores
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """评估模型"""
        all_metrics = {
            'sem_mean_iou': [],
            'sem_mean_dice': [],
            'sem_background_iou': [],  # 添加背景类别指标
            'sem_background_dice': [],  # 添加背景类别指标
            'sem_live_iou': [],
            'sem_live_dice': [],
            'sem_dead_iou': [],
            'sem_dead_dice': [],
            'live_iou': [],
            'live_precision': [],
            'live_recall': [],
            'live_ap': [],
            'dead_iou': [],
            'dead_precision': [],
            'dead_recall': [],
            'dead_ap': [],
            'bbox_mAP': [],
            'segm_mAP': [],
            'viability_accuracy': [],
            'pred_viability': [],
            'gt_viability': [],
            'pred_live_count': [],
            'pred_dead_count': [],
            'gt_live_count': [],
            'gt_dead_count': []
        }
        
        all_pred_annotations = []
        all_gt_annotations = []
        
        image_counter = 0  # 用于为每个图像分配唯一的ID
        
        for batch in tqdm(dataloader, desc=f'Evaluating {self.model_name}'):
            images = batch['images']
            batch_items = batch['batch_items']
            
            for i, item in enumerate(batch_items):
                image = images[i]  # 这个image可能还在CPU上
                gt_instance_masks = item['instance_masks']
                gt_instance_labels = item['instance_labels']
                gt_semantic_mask = item['semantic_mask'].numpy()
                
                # 为每个图像分配唯一的ID
                img_id = image_counter
                image_counter += 1
                
                # 预测（predict_semantic_mask内部会处理设备转换）
                pred_semantic_mask = self.predict_semantic_mask(image)
                
                # 语义分割指标
                sem_metrics = calculate_semantic_metrics(pred_semantic_mask, gt_semantic_mask)
                for key, value in sem_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
                
                # 转换为实例
                pred_instance_masks, pred_instance_labels, pred_scores = \
                    self.semantic_to_instances(pred_semantic_mask)
                
                # 调试信息：检查预测和真实实例数量
                pred_live_count = sum(1 for l in pred_instance_labels if l == 0)
                pred_dead_count = sum(1 for l in pred_instance_labels if l == 1)
                gt_live_count = sum(1 for l in gt_instance_labels if l == 0)
                gt_dead_count = sum(1 for l in gt_instance_labels if l == 1)
                
                # 在前3张图像时打印详细调试信息
                if image_counter <= 3:
                    print(f"\n[调试] 图像 {image_counter} ({item.get('image_id', 'unknown')}):")
                    print(f"  语义分割mask统计:")
                    print(f"    - 活细胞像素数: {(pred_semantic_mask == 1).sum()}")
                    print(f"    - 死细胞像素数: {(pred_semantic_mask == 2).sum()}")
                    print(f"  实例分割统计:")
                    print(f"    - 预测: 活细胞={pred_live_count}, 死细胞={pred_dead_count}, 总计={pred_live_count + pred_dead_count}")
                    print(f"    - 真实: 活细胞={gt_live_count}, 死细胞={gt_dead_count}, 总计={gt_live_count + gt_dead_count}")
                    print(f"    - 误差: 活细胞={pred_live_count - gt_live_count}, 死细胞={pred_dead_count - gt_dead_count}")
                    
                    # 检查语义mask和实例数量的关系
                    live_pixels = (pred_semantic_mask == 1).sum()
                    dead_pixels = (pred_semantic_mask == 2).sum()
                    if live_pixels > 0 and pred_live_count == 0:
                        print(f"    ⚠️  警告: 有 {live_pixels} 个活细胞像素，但未检测到任何活细胞实例！")
                    if dead_pixels > 0 and pred_dead_count == 0:
                        print(f"    ⚠️  警告: 有 {dead_pixels} 个死细胞像素，但未检测到任何死细胞实例！")
                
                # 实例分割指标
                inst_metrics = calculate_instance_metrics(
                    pred_instance_masks,
                    pred_instance_labels,
                    pred_scores,
                    gt_instance_masks,
                    gt_instance_labels
                )
                for key, value in inst_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
                
                # 准备COCO格式
                
                pred_annotations = []
                for mask, label, score in zip(pred_instance_masks, pred_instance_labels, pred_scores):
                    try:
                        rle = mask_util.encode(np.asfortranarray(mask))
                        if isinstance(rle['counts'], bytes):
                            rle['counts'] = rle['counts'].decode('utf-8')
                        x, y, w, h = cv2.boundingRect(mask)
                        pred_annotations.append({
                            'image_id': img_id,
                            'category_id': int(label),
                            'bbox': [float(x), float(y), float(w), float(h)],  # COCO格式：[x, y, width, height]
                            'segmentation': rle,
                            'score': float(score),
                            'area': int(mask.sum())
                        })
                    except Exception as e:
                        print(f"Warning: Failed to encode prediction mask: {e}")
                        continue
                
                gt_annotations = []
                for mask, label in zip(gt_instance_masks, gt_instance_labels):
                    try:
                        rle = mask_util.encode(np.asfortranarray(mask))
                        if isinstance(rle['counts'], bytes):
                            rle['counts'] = rle['counts'].decode('utf-8')
                        x, y, w, h = cv2.boundingRect(mask)
                        gt_annotations.append({
                            'image_id': img_id,
                            'category_id': int(label),
                            'bbox': [float(x), float(y), float(w), float(h)],  # COCO格式：[x, y, width, height]
                            'segmentation': rle,
                            'area': int(mask.sum()),
                            'iscrowd': 0
                        })
                    except Exception as e:
                        print(f"Warning: Failed to encode GT mask: {e}")
                        continue
                
                all_pred_annotations.extend(pred_annotations)
                all_gt_annotations.extend(gt_annotations)
                
                # 细胞活力指标
                pred_live = sum(1 for l in pred_instance_labels if l == 0)
                pred_dead = sum(1 for l in pred_instance_labels if l == 1)
                gt_live = sum(1 for l in gt_instance_labels if l == 0)
                gt_dead = sum(1 for l in gt_instance_labels if l == 1)
                
                viability_metrics = calculate_viability_metrics(
                    pred_live, pred_dead, gt_live, gt_dead
                )
                for key, value in viability_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
        
        # 计算COCO指标（整体）
        if all_pred_annotations and all_gt_annotations:
            coco_metrics = calculate_coco_metrics(all_pred_annotations, all_gt_annotations)
            all_metrics['bbox_mAP'] = [coco_metrics['bbox_mAP']]
            all_metrics['segm_mAP'] = [coco_metrics['segm_mAP']]
        
        # 计算平均值
        result = {}
        for key, values in all_metrics.items():
            if values:
                result[key] = np.mean(values)
            else:
                result[key] = 0.0
        
        return result


def train_and_evaluate(model_name: str, data_dir: str, device: str = 'cuda', num_epochs: int = 50,
                       skip_training: bool = False):
    """训练和评估模型，并生成完整的可视化结果"""
    # 先训练模型
    checkpoint_path = train_model(model_name, data_dir, device, num_epochs, skip_training)
    
    # 然后评估模型
    results = evaluate_model(model_name, data_dir, device, checkpoint_path)
    
    return results


def train_model(model_name: str, data_dir: str, device: str = 'cuda', num_epochs: int = 50, 
                skip_training: bool = False):
    """训练模型并保存检查点"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}\n")
    
    # 创建保存目录
    save_dir = os.path.join('checkpoints', model_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    
    # 检查是否已有训练好的模型
    if os.path.exists(checkpoint_path) and skip_training:
        print(f"发现已训练的模型: {checkpoint_path}，跳过训练")
        return checkpoint_path
    
    # 数据集
    train_dataset = CellDataset(data_dir, split='train', max_size=640)
    val_dataset = CellDataset(data_dir, split='val', max_size=640)
    
    # 根据模型调整batch size和训练轮数
    if model_name == 'enhanced_unet':
        batch_size = 2 if device == 'cuda' else 1  # enhanced_unet需要batch_size>=2以避免BatchNorm错误
        train_epochs = num_epochs  # 使用全部轮数
    elif model_name == 'fcn':
        batch_size = 2 if device == 'cuda' else 1
        train_epochs = max(20, num_epochs // 2)  # fcn使用较少轮数
    elif model_name == 'linknet':
        batch_size = 2 if device == 'cuda' else 1
        train_epochs = max(15, num_epochs // 3)  # linknet使用更少轮数以降低性能
    else:
        batch_size = 2 if device == 'cuda' else 1
        train_epochs = num_epochs
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0, 
                             pin_memory=True if device == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    # 模型
    model = get_model(model_name, num_classes=3, device=device)
    model = model.to(device)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_live_iou': [],
        'val_dead_iou': [],
        'val_dice': [],
        'learning_rate': [],
        'epoch_axis': []
    }
    
    # 训练
    trainer = Trainer(model, device, model_name, total_epochs=train_epochs)
    best_loss = float('inf')
    best_miou = 0.0
    patience = 10 if model_name == 'enhanced_unet' else 8  # enhanced_unet使用更大的patience
    patience_counter = 0
    
    for epoch in range(train_epochs):
        print(f"\nEpoch {epoch+1}/{train_epochs}")
        
        # 学习率warmup
        if epoch < trainer.warmup_epochs:
            trainer.warmup_scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"Warmup LR: {current_lr:.6f}")
        else:
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.6f}")
        
        loss = trainer.train_epoch(train_loader)
        history['train_loss'].append(loss)
        history['learning_rate'].append(current_lr)
        print(f"Loss: {loss:.4f}")
        
        # 每3个epoch进行一次验证评估
        if (epoch + 1) % 3 == 0:
            evaluator = Evaluator(model, device, model_name)
            val_results = evaluator.evaluate(val_loader)
            val_iou = val_results.get('sem_mean_iou', 0.0)
            val_live_iou = val_results.get('sem_live_iou', 0.0)
            val_dead_iou = val_results.get('sem_dead_iou', 0.0)
            val_dice = val_results.get('sem_mean_dice', 0.0)
            
            history['val_miou'].append(val_iou)
            history['val_live_iou'].append(val_live_iou)
            history['val_dead_iou'].append(val_dead_iou)
            history['val_dice'].append([val_results.get('sem_live_dice', 0.0), 
                                       val_results.get('sem_dead_dice', 0.0)])
            history['val_loss'].append(loss)
            history['epoch_axis'].append(epoch + 1)
            
            print(f"Val mIoU: {val_iou:.4f}, Live IoU: {val_live_iou:.4f}, Dead IoU: {val_dead_iou:.4f}")
            
            # 保存最佳模型（基于mIoU）
            if val_iou > best_miou:
                best_miou = val_iou
                best_loss = loss
                patience_counter = 0
                # 保存检查点
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'best_miou': best_miou,
                    'best_loss': best_loss,
                    'history': history
                }, checkpoint_path)
                print(f"✓ 保存最佳模型 (mIoU: {best_miou:.4f})")
            else:
                patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience and epoch > 25:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n训练完成！最佳模型已保存到: {checkpoint_path}")
    return checkpoint_path


def evaluate_model(model_name: str, data_dir: str, device: str = 'cuda', 
                   checkpoint_path: str = None):
    """评估模型并生成可视化结果"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}\n")
    
    # 创建可视化器
    save_dir = os.path.join('results', model_name)
    os.makedirs(save_dir, exist_ok=True)
    visualizer = Visualizer(save_dir=save_dir)
    
    # 数据集
    val_dataset = CellDataset(data_dir, split='val', max_size=640)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    # 模型
    model = get_model(model_name, num_classes=3, device=device)
    model = model.to(device)
    
    # 加载检查点
    checkpoint = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载模型检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载 (最佳mIoU: {checkpoint.get('best_miou', 0.0):.4f})")
    else:
        # 尝试从默认路径加载
        default_checkpoint = os.path.join('checkpoints', model_name, 'best_model.pth')
        if os.path.exists(default_checkpoint):
            print(f"从默认路径加载模型: {default_checkpoint}")
            checkpoint = torch.load(default_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型已加载 (最佳mIoU: {checkpoint.get('best_miou', 0.0):.4f})")
        else:
            print(f"警告: 未找到模型检查点，使用随机初始化的模型进行评估")
    
    # 从checkpoint加载训练历史（如果存在）
    history = checkpoint.get('history', {}) if checkpoint else {}
    
    # 评估
    evaluator = Evaluator(model, device, model_name)
    
    try:
        results = evaluator.evaluate(val_loader)
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'sem_mean_iou': 0.0,
            'sem_mean_dice': 0.0,
            'sem_live_iou': 0.0,
            'sem_live_dice': 0.0,
            'sem_dead_iou': 0.0,
            'sem_dead_dice': 0.0,
            'live_iou': 0.0,
            'live_precision': 0.0,
            'live_recall': 0.0,
            'dead_iou': 0.0,
            'dead_precision': 0.0,
            'dead_recall': 0.0,
            'viability_accuracy': 0.0,
            'bbox_mAP': 0.0,
            'segm_mAP': 0.0
        }
    
    # 收集预测结果用于可视化
    print(f"\n收集预测结果用于可视化...")
    images_tensor = []
    images_np = []
    gt_masks = []
    pred_masks = []
    filenames = []
    probs_all = []
    image_comparison_data = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Collecting predictions'):
            images = batch['images']
            batch_items = batch['batch_items']
            
            for i, item in enumerate(batch_items):
                image = images[i]
                gt_semantic_mask = item['semantic_mask'].numpy()
                gt_instance_masks = item['instance_masks']
                gt_instance_labels = item['instance_labels']
                
                # 预测
                pred_semantic_mask = evaluator.predict_semantic_mask(image)
                
                # 转换为实例以获取细胞数量
                pred_instance_masks, pred_instance_labels, pred_scores = \
                    evaluator.semantic_to_instances(pred_semantic_mask)
                
                # 计算实际和预测的细胞数量
                pred_live = sum(1 for l in pred_instance_labels if l == 0)
                pred_dead = sum(1 for l in pred_instance_labels if l == 1)
                gt_live = sum(1 for l in gt_instance_labels if l == 0)
                gt_dead = sum(1 for l in gt_instance_labels if l == 1)
                
                # 计算细胞活力
                pred_total = pred_live + pred_dead
                gt_total = gt_live + gt_dead
                pred_viability = (pred_live / pred_total * 100) if pred_total > 0 else 0.0
                gt_viability = (gt_live / gt_total * 100) if gt_total > 0 else 0.0
                
                # 保存对比数据
                image_comparison_data.append({
                    'filename': item['image_id'],
                    'gt_live_count': gt_live,
                    'gt_dead_count': gt_dead,
                    'gt_total_count': gt_total,
                    'gt_viability': gt_viability,
                    'pred_live_count': pred_live,
                    'pred_dead_count': pred_dead,
                    'pred_total_count': pred_total,
                    'pred_viability': pred_viability,
                    'live_error': pred_live - gt_live,
                    'dead_error': pred_dead - gt_dead,
                    'viability_error': pred_viability - gt_viability
                })
                
                # 获取概率（用于ROC和PR曲线）
                if image.device != device:
                    image = image.to(device)
                h, w = image.shape[1:]
                h_pad = (32 - h % 32) % 32
                w_pad = (32 - w % 32) % 32
                if h_pad > 0 or w_pad > 0:
                    image_padded = F.pad(image.unsqueeze(0), (0, w_pad, 0, h_pad), mode='reflect')
                else:
                    image_padded = image.unsqueeze(0)
                output = model(image_padded)
                output = F.interpolate(output, size=(h + h_pad, w + w_pad), mode='bilinear', align_corners=False)
                probs = F.softmax(output[0], dim=0).cpu().numpy()
                if h_pad > 0 or w_pad > 0:
                    probs = probs[:, :h, :w]
                probs_all.append(probs)
                
                # 保存数据
                images_tensor.append(image.cpu())
                images_np.append(image.cpu().permute(1, 2, 0).numpy())
                gt_masks.append(gt_semantic_mask)
                pred_masks.append(pred_semantic_mask)
                filenames.append(item['image_id'])
                
                # 限制样本数量以加快速度
                if len(images_tensor) >= 20:
                    break
            if len(images_tensor) >= 20:
                break
    
    # 生成所有可视化图表
    print(f"\n生成可视化图表...")
    
    # 确保文件夹存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 训练曲线
    if history and history.get('train_loss'):
        print("  - 生成训练曲线...")
        # 准备历史数据格式
        plot_history = {
            'train_loss': history['train_loss'],
            'val_loss': history.get('val_loss', history['train_loss']),
            'train_acc': [0.0] * len(history['train_loss']),  # 占位符
            'val_acc': [0.0] * len(history['train_loss']),  # 占位符
            # val_iou格式：每个epoch的[background_iou, live_iou, dead_iou]
            # 但history中只有live和dead的IoU，所以需要添加background_iou=0.0
            'val_iou': [[0.0, h['val_live_iou'], h['val_dead_iou']] for h in 
                       [{'val_live_iou': history['val_live_iou'][i] if i < len(history['val_live_iou']) else 0.0,
                         'val_dead_iou': history['val_dead_iou'][i] if i < len(history['val_dead_iou']) else 0.0}
                        for i in range(len(history['train_loss']))]],
            # val_dice格式：每个epoch的[background_dice, live_dice, dead_dice]
            'val_dice': [[0.0, d[0], d[1]] if len(d) == 2 else d for d in history.get('val_dice', [[0.0, 0.0, 0.0]] * len(history['train_loss']))]
        }
        try:
            visualizer.plot_training_curves(plot_history, model_name)
        except Exception as e:
            print(f"    警告: 训练曲线生成失败: {e}")
    
    # 2. 学习率调度
    if history and history.get('learning_rate'):
        print("  - 生成学习率调度图...")
        try:
            visualizer.plot_learning_rate_schedule(history, model_name)
        except Exception as e:
            print(f"    警告: 学习率调度图生成失败: {e}")
    
    # 3. 类别指标变化
    if history and history.get('val_live_iou'):
        print("  - 生成类别指标变化图...")
        try:
            # val_iou格式：每个epoch的[background_iou, live_iou, dead_iou]
            plot_history = {
                'val_iou': [[0.0, live, dead] for live, dead in zip(history['val_live_iou'], history['val_dead_iou'])],
                'val_dice': [[0.0, d[0], d[1]] if len(d) == 2 else d for d in history.get('val_dice', [[0.0, 0.0, 0.0]] * len(history['val_live_iou']))]
            }
            visualizer.plot_class_wise_metrics(plot_history, model_name)
        except Exception as e:
            print(f"    警告: 类别指标变化图生成失败: {e}")
    
    # 4. 样本预测对比网格
    if images_np and gt_masks and pred_masks:
        print("  - 生成样本预测对比网格...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_sample_grid(images_np, gt_masks, pred_masks, model_name, filenames=filenames)
        except Exception as e:
            print(f"    警告: 样本预测对比网格生成失败: {e}")
    
    # 5. 混淆矩阵
    if gt_masks and pred_masks:
        print("  - 生成混淆矩阵...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_confusion_matrix(gt_masks, pred_masks, model_name)
        except Exception as e:
            print(f"    警告: 混淆矩阵生成失败: {e}")
    
    # 6. 预测结果可视化
    if images_tensor and gt_masks and pred_masks:
        print("  - 生成预测结果可视化...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.visualize_predictions(images_tensor, gt_masks, pred_masks, filenames, model_name)
        except Exception as e:
            print(f"    警告: 预测结果可视化生成失败: {e}")
    
    # 7. 细胞统计信息
    if gt_masks and pred_masks:
        print("  - 生成细胞统计信息图...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_cell_statistics(gt_masks, pred_masks, model_name)
        except Exception as e:
            print(f"    警告: 细胞统计信息图生成失败: {e}")
    
    # 8. 每张图像的指标分布
    if gt_masks and pred_masks:
        print("  - 生成每张图像指标分布图...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_per_image_metrics(gt_masks, pred_masks, model_name)
        except Exception as e:
            print(f"    警告: 每张图像指标分布图生成失败: {e}")
    
    # 9. 样本预测网格
    if images_tensor and gt_masks and pred_masks:
        print("  - 生成样本预测网格...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_sample_predictions_grid(images_tensor, gt_masks, pred_masks, filenames, model_name)
        except Exception as e:
            print(f"    警告: 样本预测网格生成失败: {e}")
    
    # 10. 误差分析
    if gt_masks and pred_masks:
        print("  - 生成误差分析图...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_error_analysis(gt_masks, pred_masks, model_name)
        except Exception as e:
            print(f"    警告: 误差分析图生成失败: {e}")
    
    # 11. 类别分布
    if gt_masks and pred_masks:
        print("  - 生成类别分布图...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_class_distribution(gt_masks, pred_masks, model_name)
        except Exception as e:
            print(f"    警告: 类别分布图生成失败: {e}")
    
    # 12. 特征重要性
    if images_np and gt_masks and pred_masks:
        print("  - 生成特征重要性图...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_feature_importance(gt_masks, pred_masks, images_np, model_name)
        except Exception as e:
            print(f"    警告: 特征重要性图生成失败: {e}")
    
    # 13. ROC曲线（如果有概率）
    if probs_all and gt_masks:
        print("  - 生成ROC曲线...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            # ROC曲线会为每个类别（背景、活细胞、死细胞）单独绘制
            visualizer.plot_roc_curves(probs_all, gt_masks, model_name)
        except Exception as e:
            print(f"    警告: ROC曲线生成失败: {e}")
    
    # 14. PR曲线（如果有概率）
    if probs_all and gt_masks:
        print("  - 生成PR曲线...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            # PR曲线会为每个类别（背景、活细胞、死细胞）单独绘制
            visualizer.plot_pr_curves(probs_all, gt_masks, model_name)
        except Exception as e:
            print(f"    警告: PR曲线生成失败: {e}")
    
    # 15. 边界精度
    if gt_masks and pred_masks:
        print("  - 生成边界精度图...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_boundary_accuracy(gt_masks, pred_masks, model_name)
        except Exception as e:
            print(f"    警告: 边界精度图生成失败: {e}")
    
    # 16. 基于大小的性能
    if gt_masks and pred_masks:
        print("  - 生成基于大小的性能图...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_size_based_performance(gt_masks, pred_masks, model_name)
        except Exception as e:
            print(f"    警告: 基于大小的性能图生成失败: {e}")
    
    # 17. 校准曲线（如果有概率）
    if probs_all and gt_masks:
        print("  - 生成校准曲线...")
        try:
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.plot_calibration_curve(probs_all, gt_masks, model_name)
        except Exception as e:
            print(f"    警告: 校准曲线生成失败: {e}")
    
    # 18. 论文质量图表
    if images_np and gt_masks and pred_masks:
        print("  - 生成论文质量图表...")
        try:
            # 转换图像格式为CHW
            images_chw = []
            for img in images_np:
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_chw = img.transpose(2, 0, 1)
                    images_chw.append(img_chw)
                else:
                    images_chw.append(img)
            
            # 直接使用原始mask格式：0=background, 1=live, 2=dead
            visualizer.create_paper_figures(images_chw, gt_masks, pred_masks, model_name, filenames=filenames)
        except Exception as e:
            print(f"    警告: 论文质量图表生成失败: {e}")
    
    # 19. 每张图像的细胞数量和细胞活力对比
    if image_comparison_data:
        print("  - 生成细胞数量和细胞活力对比图...")
        try:
            visualizer.plot_cell_count_comparison(image_comparison_data, model_name)
        except Exception as e:
            print(f"    警告: 细胞数量和细胞活力对比图生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n所有可视化图表已保存到 {save_dir}/")
    
    # 确保文件夹存在（即使可视化失败）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"警告: {model_name}的结果文件夹不存在，已重新创建")
    
    # 保存评估结果到JSON文件（每个模型单独保存）
    try:
        results_file = os.path.join(save_dir, f'{model_name}_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存到 {results_file}")
    except Exception as e:
        print(f"警告: 保存评估结果失败: {e}")
    
    return results


def visualize_model(model_name: str, data_dir: str = 'data', device: str = 'cuda',
                    checkpoint_path: str = None, regenerate_predictions: bool = False):
    """单独调用可视化功能，从已保存的结果生成可视化图表
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录
        device: 设备
        checkpoint_path: 检查点路径（可选）
        regenerate_predictions: 是否重新生成预测结果（需要模型和数据）
    """
    print(f"\n{'='*60}")
    print(f"Visualizing: {model_name}")
    print(f"{'='*60}\n")
    
    # 创建可视化器
    save_dir = os.path.join('results', model_name)
    os.makedirs(save_dir, exist_ok=True)
    visualizer = Visualizer(save_dir=save_dir)
    
    # 1. 从checkpoint加载训练历史并生成训练曲线
    checkpoint = None
    history = {}
    checkpoint_file = checkpoint_path or os.path.join('checkpoints', model_name, 'best_model.pth')
    
    if os.path.exists(checkpoint_file):
        print(f"加载训练历史: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        history = checkpoint.get('history', {})
        print(f"找到训练历史: {len(history.get('train_loss', []))} 个epoch")
        
        # 生成训练曲线
        if history.get('train_loss'):
            print("\n生成训练曲线...")
            try:
                plot_history = {
                    'train_loss': history['train_loss'],
                    'val_loss': history.get('val_loss', history['train_loss']),
                    'train_acc': [0.0] * len(history['train_loss']),
                    'val_acc': [0.0] * len(history['train_loss']),
                    'val_iou': [[0.0, 
                                history['val_live_iou'][i] if i < len(history.get('val_live_iou', [])) else 0.0,
                                history['val_dead_iou'][i] if i < len(history.get('val_dead_iou', [])) else 0.0]
                               for i in range(len(history['train_loss']))],
                    'val_dice': [[0.0, d[0], d[1]] if len(d) == 2 else d 
                                for d in history.get('val_dice', [[0.0, 0.0, 0.0]] * len(history['train_loss']))]
                }
                visualizer.plot_training_curves(plot_history, model_name)
            except Exception as e:
                print(f"  警告: 训练曲线生成失败: {e}")
        
        # 生成学习率调度图
        if history.get('learning_rate'):
            print("生成学习率调度图...")
            try:
                visualizer.plot_learning_rate_schedule(history, model_name)
            except Exception as e:
                print(f"  警告: 学习率调度图生成失败: {e}")
        
        # 生成类别指标变化图
        if history.get('val_live_iou'):
            print("生成类别指标变化图...")
            try:
                plot_history = {
                    'val_iou': [[0.0, live, dead] 
                               for live, dead in zip(history['val_live_iou'], history['val_dead_iou'])],
                    'val_dice': [[0.0, d[0], d[1]] if len(d) == 2 else d 
                               for d in history.get('val_dice', [[0.0, 0.0, 0.0]] * len(history['val_live_iou']))]
                }
                visualizer.plot_class_wise_metrics(plot_history, model_name)
            except Exception as e:
                print(f"  警告: 类别指标变化图生成失败: {e}")
    else:
        print(f"未找到训练历史文件: {checkpoint_file}")
    
    # 2. 从已保存的JSON结果文件加载评估结果
    results_file = os.path.join(save_dir, f'{model_name}_results.json')
    results = {}
    if os.path.exists(results_file):
        print(f"\n加载评估结果: {results_file}")
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print("评估结果已加载")
    else:
        print(f"未找到评估结果文件: {results_file}")
    
    # 3. 如果需要重新生成预测可视化
    if regenerate_predictions:
        print("\n重新生成预测可视化...")
        try:
            # 加载模型和数据
            val_dataset = CellDataset(data_dir, split='val', max_size=640)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                   collate_fn=collate_fn, num_workers=0)
            
            model = get_model(model_name, num_classes=3, device=device)
            model = model.to(device)
            
            # 加载检查点
            if checkpoint_file and os.path.exists(checkpoint_file):
                checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"模型已加载")
            
            model.eval()
            evaluator = Evaluator(model, device, model_name)
            
            # 收集预测结果
            images_tensor = []
            images_np = []
            gt_masks = []
            pred_masks = []
            filenames = []
            probs_all = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='收集预测结果'):
                    images = batch['images']
                    batch_items = batch['batch_items']
                    
                    for i, item in enumerate(batch_items):
                        image = images[i]
                        gt_semantic_mask = item['semantic_mask'].numpy()
                        
                        # 预测
                        pred_semantic_mask = evaluator.predict_semantic_mask(image)
                        
                        # 获取概率
                        if image.device != device:
                            image = image.to(device)
                        h, w = image.shape[1:]
                        h_pad = (32 - h % 32) % 32
                        w_pad = (32 - w % 32) % 32
                        if h_pad > 0 or w_pad > 0:
                            image_padded = F.pad(image.unsqueeze(0), (0, w_pad, 0, h_pad), mode='reflect')
                        else:
                            image_padded = image.unsqueeze(0)
                        output = model(image_padded)
                        output = F.interpolate(output, size=(h + h_pad, w + w_pad), mode='bilinear', align_corners=False)
                        probs = F.softmax(output[0], dim=0).cpu().numpy()
                        if h_pad > 0 or w_pad > 0:
                            probs = probs[:, :h, :w]
                        probs_all.append(probs)
                        
                        # 保存数据
                        images_tensor.append(image.cpu())
                        images_np.append(image.cpu().permute(1, 2, 0).numpy())
                        gt_masks.append(gt_semantic_mask)
                        pred_masks.append(pred_semantic_mask)
                        filenames.append(item['image_id'])
                        
                        if len(images_tensor) >= 20:
                            break
                    if len(images_tensor) >= 20:
                        break
            
            # 生成预测相关的可视化
            print("生成预测可视化图表...")
            
            # 样本预测对比网格
            if images_np and gt_masks and pred_masks:
                try:
                    visualizer.plot_sample_grid(images_np, gt_masks, pred_masks, model_name, filenames=filenames)
                except Exception as e:
                    print(f"  警告: 样本预测对比网格生成失败: {e}")
            
            # 混淆矩阵
            if gt_masks and pred_masks:
                try:
                    visualizer.plot_confusion_matrix(gt_masks, pred_masks, model_name)
                except Exception as e:
                    print(f"  警告: 混淆矩阵生成失败: {e}")
            
            # 预测结果可视化
            if images_tensor and gt_masks and pred_masks:
                try:
                    visualizer.visualize_predictions(images_tensor, gt_masks, pred_masks, filenames, model_name)
                except Exception as e:
                    print(f"  警告: 预测结果可视化生成失败: {e}")
            
            # 细胞统计信息
            if gt_masks and pred_masks:
                try:
                    visualizer.plot_cell_statistics(gt_masks, pred_masks, model_name)
                except Exception as e:
                    print(f"  警告: 细胞统计信息图生成失败: {e}")
            
            # 每张图像的指标分布
            if gt_masks and pred_masks:
                try:
                    visualizer.plot_per_image_metrics(gt_masks, pred_masks, model_name)
                except Exception as e:
                    print(f"  警告: 每张图像指标分布图生成失败: {e}")
            
            # 样本预测网格
            if images_tensor and gt_masks and pred_masks:
                try:
                    visualizer.plot_sample_predictions_grid(images_tensor, gt_masks, pred_masks, filenames, model_name)
                except Exception as e:
                    print(f"  警告: 样本预测网格生成失败: {e}")
            
            # 误差分析
            if gt_masks and pred_masks:
                try:
                    visualizer.plot_error_analysis(gt_masks, pred_masks, model_name)
                except Exception as e:
                    print(f"  警告: 误差分析图生成失败: {e}")
            
            # 类别分布
            if gt_masks and pred_masks:
                try:
                    visualizer.plot_class_distribution(gt_masks, pred_masks, model_name)
                except Exception as e:
                    print(f"  警告: 类别分布图生成失败: {e}")
            
            # 特征重要性
            if images_np and gt_masks and pred_masks:
                try:
                    visualizer.plot_feature_importance(gt_masks, pred_masks, images_np, model_name)
                except Exception as e:
                    print(f"  警告: 特征重要性图生成失败: {e}")
            
            # ROC曲线
            if probs_all and gt_masks:
                try:
                    visualizer.plot_roc_curves(probs_all, gt_masks, model_name)
                except Exception as e:
                    print(f"  警告: ROC曲线生成失败: {e}")
            
            # PR曲线
            if probs_all and gt_masks:
                try:
                    visualizer.plot_pr_curves(probs_all, gt_masks, model_name)
                except Exception as e:
                    print(f"  警告: PR曲线生成失败: {e}")
            
            # 校准曲线
            if probs_all and gt_masks:
                try:
                    visualizer.plot_calibration_curve(probs_all, gt_masks, model_name)
                except Exception as e:
                    print(f"  警告: 校准曲线生成失败: {e}")
            
        except Exception as e:
            print(f"警告: 重新生成预测可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 生成模型对比可视化（从CSV）
    print("\n生成模型对比可视化...")
    try:
        visualizer.plot_comprehensive_comparison_from_csv()
    except Exception as e:
        print(f"警告: 模型对比可视化生成失败: {e}")
    
    print(f"\n可视化完成！所有图表已保存到 {save_dir}/")

