"""
Visualization System - All visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import os
from typing import List, Dict
import pandas as pd
from matplotlib import font_manager as fm


class Visualizer:
    """Visualization Tool Class"""

    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Set font for scientific figures (优先使用中文字体，避免缺失提示)
        # 完全抑制字体相关的警告（包括glyph缺失警告）
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
        warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
        warnings.filterwarnings('ignore', message='.*missing from font.*')
        
        # 设置matplotlib不显示字体警告
        import logging
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        chosen_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                chosen_font = font
                break
        if chosen_font:
            plt.rcParams['font.sans-serif'] = [chosen_font]
            plt.rcParams['font.family'] = chosen_font
        else:
            # Linux云平台通常没有中文字体，使用系统默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
            plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置matplotlib的警告过滤（在运行时也生效）
        import matplotlib
        matplotlib.rcParams['font.family'] = plt.rcParams['font.family']

        # Scientific color palette
        self.palette = ['#7F5994', '#7D70A6', '#7685AE', '#6F97B0', '#65A8B0',
                       '#61B9AB', '#6ECAA3', '#8FD892', '#B8E475', '#E8EB5E']

        # Class color mapping for segmentation masks
        # 3个类别: 背景(0), 活细胞(1), 死细胞(2)
        # 未标注区域(255)不参与评估
        self.colors = {
            0: [180, 180, 180],  # Background - Light Gray (增强可见性)
            1: [0, 255, 0],       # Live cells - Green
            2: [255, 0, 0],      # Dead cells - Red
            255: [0, 0, 0]       # Unlabeled - Black
        }

        # 3个类别：背景、活细胞、死细胞
        self.class_names = ['Background', 'Live Cells', 'Dead Cells']

        # Class colors for paper figures (RGB normalized)
        self.class_colors = [
            [0.7, 0.7, 0.7, 0.8],  # Background - Light Gray with alpha (增强可见性)
            [0, 1, 0, 0.6],        # Live cells - Green with alpha
            [1, 0, 0, 0.6]         # Dead cells - Red with alpha
        ]

        # 是否同时保存SVG格式（用于论文）
        self.save_svg = True

    def _save_figure(self, fig, filename: str, dpi: int = 300):
        """保存图表为PNG和SVG格式（高清小图）

        Args:
            fig: matplotlib figure对象
            filename: 文件名（不含扩展名）
            dpi: PNG分辨率（默认300，高清）
        """
        # 抑制保存时的字体警告
        import warnings
        import logging
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
            warnings.filterwarnings('ignore', message='.*missing from font.*')
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
            
            # 临时设置日志级别
            old_level = logging.getLogger('matplotlib.font_manager').level
            logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
            
            try:
                # 保存PNG格式（高分辨率，用于论文和展示）
                png_path = os.path.join(self.save_dir, f'{filename}.png')
                fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', 
                           edgecolor='none', pad_inches=0.1)
                print(f"PNG saved: {png_path}")

                # 保存SVG格式（矢量图，用于论文，可无损缩放）
                if self.save_svg:
                    svg_path = os.path.join(self.save_dir, f'{filename}.svg')
                    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white',
                               edgecolor='none', pad_inches=0.1)
                    print(f"SVG saved: {svg_path}")
            finally:
                # 恢复日志级别
                logging.getLogger('matplotlib.font_manager').setLevel(old_level)

    def plot_training_curves(self, history: Dict, model_name: str):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Loss curve
        if 'train_loss' in history and len(history['train_loss']) > 0:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0, 0].plot(epochs, history['train_loss'], label='Training Loss', linewidth=2.5, color=self.palette[0])
            if 'val_loss' in history and len(history['val_loss']) > 0:
                val_epochs = range(1, len(history['val_loss']) + 1)
                axes[0, 0].plot(val_epochs, history['val_loss'], label='Validation Loss', linewidth=2.5, color=self.palette[4])
        axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10, frameon=True, shadow=True)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)

        # Accuracy curve (如果有)
        if 'train_acc' in history and len(history['train_acc']) > 0:
            epochs = range(1, len(history['train_acc']) + 1)
            axes[0, 1].plot(epochs, history['train_acc'], label='Training Accuracy', linewidth=2.5, color=self.palette[1])
            if 'val_acc' in history and len(history['val_acc']) > 0:
                val_epochs = range(1, len(history['val_acc']) + 1)
                axes[0, 1].plot(val_epochs, history['val_acc'], label='Validation Accuracy', linewidth=2.5, color=self.palette[5])
        else:
            axes[0, 1].text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_title(f'{model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10, frameon=True, shadow=True)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)

        # IoU curve
        if 'val_iou' in history and len(history['val_iou']) > 0:
            val_iou = np.array(history['val_iou'])
            if len(val_iou.shape) == 2 and val_iou.shape[1] >= len(self.class_names):
                epochs = range(1, len(val_iou) + 1)
            for i, class_name in enumerate(self.class_names):
                    if i < val_iou.shape[1]:
                        axes[1, 0].plot(epochs, val_iou[:, i], label=class_name, linewidth=2.5, color=self.palette[i])
            axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('IoU', fontsize=12, fontweight='bold')
            axes[1, 0].set_title(f'{model_name} - IoU Curve', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=10, frameon=True, shadow=True)
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
            axes[1, 0].spines['top'].set_visible(False)
            axes[1, 0].spines['right'].set_visible(False)
        else:
            axes[1, 0].text(0.5, 0.5, 'IoU data not available', ha='center', va='center', transform=axes[1, 0].transAxes)

        # Dice curve
        if 'val_dice' in history and len(history['val_dice']) > 0:
            val_dice = np.array(history['val_dice'])
            if len(val_dice.shape) == 2 and val_dice.shape[1] >= len(self.class_names):
                epochs = range(1, len(val_dice) + 1)
            for i, class_name in enumerate(self.class_names):
                    if i < val_dice.shape[1]:
                        axes[1, 1].plot(epochs, val_dice[:, i], label=class_name, linewidth=2.5, color=self.palette[i])
            axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Dice Coefficient', fontsize=12, fontweight='bold')
            axes[1, 1].set_title(f'{model_name} - Dice Coefficient Curve', fontsize=14, fontweight='bold')
            axes[1, 1].legend(fontsize=10, frameon=True, shadow=True)
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
            axes[1, 1].spines['top'].set_visible(False)
            axes[1, 1].spines['right'].set_visible(False)
        else:
            axes[1, 1].text(0.5, 0.5, 'Dice data not available', ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_training_curves')
        plt.close()

    def plot_sample_grid(self, images, masks_true, masks_pred, model_name: str, filenames=None):
        """绘制样本预测对比网格（四列：预处理前的原图、预处理后的图、真实标注、预测标注）"""
        num_samples = min(8, len(images))

        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # 第1列：预处理前的原图（直接从data目录加载）
            original_img = None
            if filenames and i < len(filenames):
                # 直接从data目录加载原始图像
                try:
                    from PIL import Image
                    import os
                    # 直接从data目录获取原图
                    img_path = os.path.join('data', filenames[i])
                    if os.path.exists(img_path):
                        original_img = np.array(Image.open(img_path).convert('RGB'))
                        original_img = original_img.astype(np.float32) / 255.0
                    else:
                        print(f"警告: 未找到原图文件: {img_path}，使用反归一化近似")
                except Exception as e:
                    print(f"警告: 加载原图失败: {e}，使用反归一化近似")
                    pass  # 如果加载失败，使用反归一化近似
            
            # 如果无法加载原图，使用反归一化来近似预处理前的图像
            if original_img is None:
                img_preprocessed = images[i]
                if isinstance(img_preprocessed, np.ndarray):
                    # 如果是CHW格式，转换为HWC
                    if len(img_preprocessed.shape) == 3 and img_preprocessed.shape[0] == 3:
                        img_preprocessed = img_preprocessed.transpose(1, 2, 0)
                    
                    # 反归一化：恢复预处理前的图像
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    
                    if img_preprocessed.dtype != np.float32:
                        img_preprocessed = img_preprocessed.astype(np.float32)
                    
                    # 反归一化得到预处理前的图像（近似）
                    original_img = img_preprocessed * std + mean
                    original_img = np.clip(original_img, 0, 1)

            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f'Sample {i+1} - 预处理前的原图', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')

            # 第2列：预处理后的图（归一化后的图像）
            img_preprocessed = images[i]
            if isinstance(img_preprocessed, np.ndarray):
                # 如果是CHW格式，转换为HWC
                if len(img_preprocessed.shape) == 3 and img_preprocessed.shape[0] == 3:
                    img_preprocessed = img_preprocessed.transpose(1, 2, 0)
                
                # 反归一化以便显示
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                if img_preprocessed.dtype != np.float32:
                    img_preprocessed = img_preprocessed.astype(np.float32)
                
                img_for_display = img_preprocessed * std + mean
                img_for_display = np.clip(img_for_display, 0, 1)
            else:
                img_for_display = img_preprocessed
            
            axes[i, 1].imshow(img_for_display)
            axes[i, 1].set_title('预处理后的图', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')

            # 第3列：真实标注
            if isinstance(masks_true[i], torch.Tensor):
                mask_true_np = masks_true[i].cpu().numpy()
            else:
                mask_true_np = np.array(masks_true[i])
            mask_true_colored = self._colorize_mask(mask_true_np)
            axes[i, 2].imshow(mask_true_colored)
            axes[i, 2].set_title('真实标注', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')

            # 第4列：预测标注
            if isinstance(masks_pred[i], torch.Tensor):
                mask_pred_np = masks_pred[i].cpu().numpy()
            else:
                mask_pred_np = np.array(masks_pred[i])
            mask_pred_colored = self._colorize_mask(mask_pred_np)
            axes[i, 3].imshow(mask_pred_colored)
            axes[i, 3].set_title('预测标注', fontsize=12, fontweight='bold')
            axes[i, 3].axis('off')

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_sample_grid')
        plt.close()

    def plot_confusion_matrix(self, masks_true, masks_pred, model_name: str):
        """Plot confusion matrix
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        """
        # Flatten all masks
        y_true = np.concatenate([mask.flatten() for mask in masks_true])
        y_pred = np.concatenate([mask.flatten() for mask in masks_pred])

        # Filter out ignore index (255)
        valid_mask = (y_true != 255) & (y_pred != 255)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        # 确保类别标签在0-2范围内（背景、活细胞、死细胞）
        y_true = np.clip(y_true, 0, 2)
        y_pred = np.clip(y_pred, 0, 2)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'},
                    ax=ax1, square=True)
        ax1.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name} - Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Percentage'},
                    ax=ax2, square=True)
        ax2.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax2.set_title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_confusion_matrix')
        plt.close()

    def visualize_predictions(self, images, masks_true, masks_pred,
                            filenames, model_name: str, num_samples=8):
        """Visualize prediction results"""
        num_samples = min(num_samples, len(images))

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Original image
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Original Image\n{filenames[i]}', fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')

            # Ground truth mask
            if isinstance(masks_true[i], torch.Tensor):
                mask_true_np = masks_true[i].cpu().numpy()
            else:
                mask_true_np = masks_true[i]
            mask_true_colored = self.mask_to_color(mask_true_np)
            axes[i, 1].imshow(mask_true_colored)
            axes[i, 1].set_title('Ground Truth', fontsize=10, fontweight='bold')
            axes[i, 1].axis('off')

            # Predicted mask
            if isinstance(masks_pred[i], torch.Tensor):
                mask_pred_np = masks_pred[i].cpu().numpy()
            else:
                mask_pred_np = masks_pred[i]
            mask_pred_colored = self.mask_to_color(mask_pred_np)
            axes[i, 2].imshow(mask_pred_colored)
            axes[i, 2].set_title('Prediction', fontsize=10, fontweight='bold')
            axes[i, 2].axis('off')

            # Overlay (mask_pred_colored已经是归一化的0-1范围)
            overlay = (img * 0.6 + mask_pred_colored * 0.4)
            overlay = np.clip(overlay, 0, 1)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay', fontsize=10, fontweight='bold')
            axes[i, 3].axis('off')

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_predictions')
        plt.close()

    def mask_to_color(self, mask):
        """Convert mask to colored image
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        """
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask = mask.astype(np.int64)

        for class_id, color in self.colors.items():
                colored_mask[mask == class_id] = color

        return colored_mask.astype(np.float32) / 255.0

    def plot_cell_statistics(self, masks_true, masks_pred, model_name: str):
        """绘制细胞统计信息
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 统计每个类别的像素数量
        # 3个类别：背景(0)、活细胞(1)、死细胞(2)
        num_classes = 3
        true_counts = []
        pred_counts = []

        for mask_true, mask_pred in zip(masks_true, masks_pred):
            # 统计有效类别（0, 1, 2），忽略255
            true_count = [np.sum(mask_true == i) for i in range(num_classes)]
            pred_count = [np.sum(mask_pred == i) for i in range(num_classes)]
            true_counts.append(true_count)
            pred_counts.append(pred_count)

        true_counts = np.array(true_counts)  # shape: (num_images, 3)
        pred_counts = np.array(pred_counts)  # shape: (num_images, 3)

        # 类别分布对比
        x = np.arange(num_classes)  # [0, 1, 2]
        width = 0.35

        mean_true = true_counts.mean(axis=0)  # shape: (3,)
        mean_pred = pred_counts.mean(axis=0)  # shape: (3,)

        axes[0, 0].bar(x - width/2, mean_true, width,
                      label='Ground Truth', alpha=0.8, color=self.palette[0])
        axes[0, 0].bar(x + width/2, mean_pred, width,
                      label='Prediction', alpha=0.8, color=self.palette[4])
        axes[0, 0].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Average Pixel Count', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(['Background', 'Live Cells', 'Dead Cells'], rotation=15, ha='right')
        axes[0, 0].legend(frameon=True, shadow=True)
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)

        # 活细胞 vs 死细胞比例（不包括背景）
        live_true = true_counts[:, 1]  # 类别1是活细胞
        dead_true = true_counts[:, 2]  # 类别2是死细胞
        live_pred = pred_counts[:, 1]
        dead_pred = pred_counts[:, 2]

        ratio_true = live_true / (live_true + dead_true + 1e-6)
        ratio_pred = live_pred / (live_pred + dead_pred + 1e-6)

        axes[0, 1].scatter(ratio_true, ratio_pred, alpha=0.6, s=50, color=self.palette[2])
        axes[0, 1].plot([0, 1], [0, 1], '--', color=self.palette[8], linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('True Live Cell Ratio', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Predicted Live Cell Ratio', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Live Cell Ratio Prediction', fontsize=14, fontweight='bold')
        axes[0, 1].legend(frameon=True, shadow=True)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)

        # Cell count distribution
        axes[1, 0].hist(live_true, bins=20, alpha=0.6, label='Live Cells (GT)', color='#6ECAA3')
        axes[1, 0].hist(dead_true, bins=20, alpha=0.6, label='Dead Cells (GT)', color='#E8EB5E')
        axes[1, 0].set_xlabel('Pixel Count', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Cell Count Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend(frameon=True, shadow=True)
        axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)

        # Prediction error distribution
        error_live = np.abs(live_pred - live_true)
        error_dead = np.abs(dead_pred - dead_true)

        bp = axes[1, 1].boxplot([error_live, error_dead], labels=['Live Cells', 'Dead Cells'],
                                patch_artist=True)
        for patch, color in zip(bp['boxes'], [self.palette[6], self.palette[9]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_ylabel('Prediction Error (Pixel Count)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1, 1].spines['top'].set_visible(False)
        axes[1, 1].spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_cell_statistics')
        plt.close()

    def plot_model_comparison(self, results: Dict[str, Dict]):
        """绘制模型对比图（基础版本）"""
        models = list(results.keys())
        metrics = ['IoU', 'Dice', 'Accuracy']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, metric in enumerate(metrics):
            data = []
            for model in models:
                if metric == 'IoU':
                    # 使用CSV中的mean_iou，不重新计算
                    data.append(results[model].get('mean_iou', results[model].get('accuracy', 0)))
                elif metric == 'Dice':
                    # 使用CSV中的mean_dice，不重新计算
                    data.append(results[model].get('mean_dice', 0))
                else:
                    data.append(results[model].get('accuracy', 0))

            colors = [self.palette[i % len(self.palette)] for i in range(len(models))]
            axes[idx].bar(models, data, alpha=0.8, color=colors)
            axes[idx].set_ylabel(metric, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(np.arange(len(models)))
            axes[idx].set_xticklabels(models, rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)

            # Add value labels
            for i, v in enumerate(data):
                axes[idx].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom',
                             fontsize=9, fontweight='bold')

        plt.tight_layout()
        self._save_figure(fig, 'model_comparison')
        plt.close()
    
    def plot_comprehensive_comparison(self, results: Dict[str, Dict]):
        """绘制全面的模型对比可视化（10+张图）"""
        models = list(results.keys())
        
        # 1. 总体指标对比（Mean IoU, Mean Dice, Accuracy）
        # 使用CSV中的实际值，不重新计算
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics_data = {
            'Mean IoU': [results[m].get('mean_iou', results[m].get('accuracy', 0)) for m in models],  # 使用CSV中的语义分割 mIoU
            'Mean Dice': [results[m].get('mean_dice', 0) for m in models],  # 使用CSV中的语义分割 mDice
            'Accuracy': [results[m].get('accuracy', 0) for m in models]  # 使用CSV中的"细胞活力准确率"
        }
        
        for idx, (metric, data) in enumerate(metrics_data.items()):
            colors = [self.palette[i % len(self.palette)] for i in range(len(models))]
            bars = axes[idx].bar(models, data, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
            axes[idx].set_ylabel(metric, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(np.arange(len(models)))
            axes[idx].set_xticklabels(models, rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            axes[idx].set_ylim([0, max(data) * 1.15 if max(data) > 0 else 1])
            
            for i, (bar, v) in enumerate(zip(bars, data)):
                axes[idx].text(bar.get_x() + bar.get_width()/2., v + max(data)*0.02,
                             f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_overall_metrics')
        plt.close()
        
        # 2. 类别级别的IoU对比（3个类别：background, live, dead）
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        for class_idx, class_name in enumerate(self.class_names):
            if class_idx >= len(axes):
                break
            # iou格式可能是[0, 0]（只有live和dead）或[0, 0, 0]（包括background）
            iou_data = []
            for m in models:
                iou_list = results[m].get('iou', [0, 0, 0])
                # 如果只有2个元素，说明没有background，需要添加
                if len(iou_list) == 2:
                    if class_idx == 0:  # background
                        iou_data.append(0.0)
                    else:  # live (1) or dead (2)
                        iou_data.append(iou_list[class_idx - 1])
                else:  # 有3个元素
                    iou_data.append(iou_list[class_idx] if class_idx < len(iou_list) else 0.0)
            colors = [self.palette[i % len(self.palette)] for i in range(len(models))]
            bars = axes[class_idx].bar(models, iou_data, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
            axes[class_idx].set_ylabel('IoU Score', fontsize=12, fontweight='bold')
            axes[class_idx].set_title(f'{class_name} IoU Comparison', fontsize=14, fontweight='bold')
            axes[class_idx].set_xticks(np.arange(len(models)))
            axes[class_idx].set_xticklabels(models, rotation=45, ha='right')
            axes[class_idx].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[class_idx].spines['top'].set_visible(False)
            axes[class_idx].spines['right'].set_visible(False)
            axes[class_idx].set_ylim([0, max(iou_data) * 1.15 if max(iou_data) > 0 else 1])
            
            for i, (bar, v) in enumerate(zip(bars, iou_data)):
                axes[class_idx].text(bar.get_x() + bar.get_width()/2., v + max(iou_data)*0.02,
                                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_class_iou')
        plt.close()
        
        # 3. 类别级别的Dice对比（3个类别：background, live, dead）
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        for class_idx, class_name in enumerate(self.class_names):
            if class_idx >= len(axes):
                break
            # dice格式可能是[0, 0]（只有live和dead）或[0, 0, 0]（包括background）
            dice_data = []
            for m in models:
                dice_list = results[m].get('dice', [0, 0, 0])
                # 如果只有2个元素，说明没有background，需要添加
                if len(dice_list) == 2:
                    if class_idx == 0:  # background
                        dice_data.append(0.0)
                    else:  # live (1) or dead (2)
                        dice_data.append(dice_list[class_idx - 1])
                else:  # 有3个元素
                    dice_data.append(dice_list[class_idx] if class_idx < len(dice_list) else 0.0)
            colors = [self.palette[i % len(self.palette)] for i in range(len(models))]
            bars = axes[class_idx].bar(models, dice_data, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
            axes[class_idx].set_ylabel('Dice Score', fontsize=12, fontweight='bold')
            axes[class_idx].set_title(f'{class_name} Dice Comparison', fontsize=14, fontweight='bold')
            axes[class_idx].set_xticks(np.arange(len(models)))
            axes[class_idx].set_xticklabels(models, rotation=45, ha='right')
            axes[class_idx].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[class_idx].spines['top'].set_visible(False)
            axes[class_idx].spines['right'].set_visible(False)
            axes[class_idx].set_ylim([0, max(dice_data) * 1.15 if max(dice_data) > 0 else 1])
            
            for i, (bar, v) in enumerate(zip(bars, dice_data)):
                axes[class_idx].text(bar.get_x() + bar.get_width()/2., v + max(dice_data)*0.02,
                                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_class_dice')
        plt.close()
        
        # 4. 雷达图对比（多维度性能）
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        metrics_labels = ['Live Cells IoU', 'Dead Cells IoU', 'Live Cells Dice', 'Dead Cells Dice', 'Accuracy']
        num_metrics = len(metrics_labels)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for model in models:
            # 正确处理数据格式：[background, live, dead]
            iou_list = results[model].get('iou', [0, 0, 0])
            dice_list = results[model].get('dice', [0, 0, 0])
            # 确保至少有3个元素
            if len(iou_list) == 2:
                iou_list = [0.0] + iou_list
            if len(dice_list) == 2:
                dice_list = [0.0] + dice_list
            values = [
                iou_list[1] if len(iou_list) > 1 else 0.0,  # Live Cells IoU (索引1)
                iou_list[2] if len(iou_list) > 2 else 0.0,  # Dead Cells IoU (索引2)
                dice_list[1] if len(dice_list) > 1 else 0.0,  # Live Cells Dice (索引1)
                dice_list[2] if len(dice_list) > 2 else 0.0,  # Dead Cells Dice (索引2)
                results[model].get('accuracy', 0)  # Accuracy (细胞活力准确率)
            ]
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, label=model, markersize=8)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=10)
        ax.set_ylim([0, 1])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_radar')
        plt.close()
        
        # 5. 热力图对比（所有指标）
        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap_data = []
        row_labels = []
        for model in models:
            # 正确处理数据格式：[background, live, dead]
            iou_list = results[model].get('iou', [0, 0, 0])
            dice_list = results[model].get('dice', [0, 0, 0])
            # 确保至少有3个元素
            if len(iou_list) == 2:
                iou_list = [0.0] + iou_list
            if len(dice_list) == 2:
                dice_list = [0.0] + dice_list
            row = [
                iou_list[1] if len(iou_list) > 1 else 0.0,  # Live IoU (索引1)
                iou_list[2] if len(iou_list) > 2 else 0.0,  # Dead IoU (索引2)
                results[model].get('mean_iou', results[model].get('accuracy', 0)),  # Mean IoU (使用CSV值)
                dice_list[1] if len(dice_list) > 1 else 0.0,  # Live Dice (索引1)
                dice_list[2] if len(dice_list) > 2 else 0.0,  # Dead Dice (索引2)
                results[model].get('mean_dice', 0),  # Mean Dice (使用CSV值)
                results[model].get('accuracy', 0)  # Accuracy (细胞活力准确率)
            ]
            heatmap_data.append(row)
            row_labels.append(model)
        
        col_labels = ['Live IoU', 'Dead IoU', 'Mean IoU', 'Live Dice', 'Dead Dice', 'Mean Dice', 'Accuracy']
        heatmap_data = np.array(heatmap_data)
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
        ax.set_yticklabels(row_labels)
        ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        
        # 添加数值标注
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold', fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Score', fraction=0.046, pad=0.04)
        plt.tight_layout()
        self._save_figure(fig, 'comparison_heatmap')
        plt.close()
        
        # 6. 箱线图对比（类别指标分布）- 3个类别：background, live, dead
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        for class_idx, class_name in enumerate(self.class_names):
            if class_idx >= len(axes):
                break
            # iou和dice格式可能是[0, 0]（只有live和dead）或[0, 0, 0]（包括background）
            iou_data = []
            dice_data = []
            for m in models:
                iou_list = results[m].get('iou', [0, 0, 0])
                dice_list = results[m].get('dice', [0, 0, 0])
                # 如果只有2个元素，说明没有background，需要添加
                if len(iou_list) == 2:
                    if class_idx == 0:  # background
                        iou_data.append(0.0)
                    else:  # live (1) or dead (2)
                        iou_data.append(iou_list[class_idx - 1])
                else:  # 有3个元素
                    iou_data.append(iou_list[class_idx] if class_idx < len(iou_list) else 0.0)
                
                if len(dice_list) == 2:
                    if class_idx == 0:  # background
                        dice_data.append(0.0)
                    else:  # live (1) or dead (2)
                        dice_data.append(dice_list[class_idx - 1])
                else:  # 有3个元素
                    dice_data.append(dice_list[class_idx] if class_idx < len(dice_list) else 0.0)
            
            data_to_plot = [iou_data, dice_data]
            if class_idx < len(axes) and len(data_to_plot[0]) > 0:  # 确保有数据
                bp = axes[class_idx].boxplot(data_to_plot, labels=['IoU', 'Dice'], patch_artist=True, widths=0.6)
                for patch, color in zip(bp['boxes'], [self.palette[0], self.palette[4]]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                
                axes[class_idx].set_ylabel('Score', fontsize=12, fontweight='bold')
                axes[class_idx].set_title(f'{class_name} Metrics Distribution', fontsize=14, fontweight='bold')
                axes[class_idx].grid(True, alpha=0.3, axis='y', linestyle='--')
                axes[class_idx].spines['top'].set_visible(False)
                axes[class_idx].spines['right'].set_visible(False)
                axes[class_idx].set_ylim([0, 1])
                axes[class_idx].set_xticks([1, 2])
                axes[class_idx].set_xticklabels(['IoU', 'Dice'])
            else:
                # 如果没有数据，显示空图
                axes[class_idx].text(0.5, 0.5, 'No data', ha='center', va='center', 
                                   transform=axes[class_idx].transAxes)
                axes[class_idx].set_title(f'{class_name} Metrics Distribution', fontsize=14, fontweight='bold')
                axes[class_idx].set_ylim([0, 1])
                axes[class_idx].set_xticks([1, 2])
                axes[class_idx].set_xticklabels(['IoU', 'Dice'])
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_boxplot')
        plt.close()
        
        # 7. 并排柱状图（Live vs Dead性能对比）- 修复索引问题
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for metric_idx, metric_name in enumerate(['IoU', 'Dice']):
            # 正确获取live和dead的数据，处理两种数据格式：[live, dead] 或 [background, live, dead]
            live_data = []
            dead_data = []
            for m in models:
                metric_list = results[m].get(metric_name.lower(), [0, 0])
                if len(metric_list) == 2:
                    # 格式是 [live, dead]
                    live_data.append(metric_list[0])
                    dead_data.append(metric_list[1])
                elif len(metric_list) >= 3:
                    # 格式是 [background, live, dead]
                    live_data.append(metric_list[1])
                    dead_data.append(metric_list[2])
                else:
                    # 默认值
                    live_data.append(0.0)
                    dead_data.append(0.0)
            
            x = np.arange(len(models))
            width = 0.35  # 减小宽度以便并排显示
            
            # 使用并排柱状图而不是堆叠图，更清晰地比较
            bars1 = axes[metric_idx].bar(x - width/2, live_data, width, label='Live Cells', 
                                         color=self.palette[0], alpha=0.8, edgecolor='black', linewidth=1.5)
            bars2 = axes[metric_idx].bar(x + width/2, dead_data, width, label='Dead Cells',
                                        color=self.palette[4], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.01:  # 只显示大于0.01的值
                        axes[metric_idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                             f'{height:.3f}', ha='center', va='bottom', 
                                             fontsize=8, fontweight='bold')
            
            axes[metric_idx].set_ylabel(f'{metric_name} Score', fontsize=12, fontweight='bold')
            axes[metric_idx].set_title(f'{metric_name} - Live vs Dead Cells Comparison', fontsize=14, fontweight='bold')
            axes[metric_idx].set_xticks(x)
            axes[metric_idx].set_xticklabels(models, rotation=45, ha='right')
            axes[metric_idx].legend(fontsize=10, frameon=True, shadow=True, loc='upper right')
            axes[metric_idx].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[metric_idx].spines['top'].set_visible(False)
            axes[metric_idx].spines['right'].set_visible(False)
            axes[metric_idx].set_ylim([0, max(max(live_data) if live_data else [0], max(dead_data) if dead_data else [0]) * 1.15])
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_stacked')
        plt.close()
        
        # 8. 散点图对比（IoU vs Dice）
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, model in enumerate(models):
            # 使用CSV中的实际值，不重新计算
            mean_iou = results[model].get('mean_iou', results[model].get('accuracy', 0))
            mean_dice = results[model].get('mean_dice', 0)
            ax.scatter(mean_iou, mean_dice, s=200, alpha=0.7, color=self.palette[i % len(self.palette)],
                      edgecolors='black', linewidth=2, label=model)
            ax.annotate(model, (mean_iou, mean_dice), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Mean IoU', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Dice', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance: IoU vs Dice', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 添加对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_scatter')
        plt.close()
        
        # 9. 性能排名图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        metrics_to_rank = [
            ('Mean IoU', lambda r: r.get('mean_iou', r.get('accuracy', 0))),  # 使用CSV中的mIoU
            ('Mean Dice', lambda r: r.get('mean_dice', 0)),  # 使用CSV中的mDice
            ('Accuracy', lambda r: r.get('accuracy', 0)),
            ('Dead Cells IoU', lambda r: (r.get('iou', [0, 0, 0])[2] if len(r.get('iou', [0, 0, 0])) > 2 else (r.get('iou', [0, 0])[1] if len(r.get('iou', [0, 0])) > 1 else 0.0)))  # 正确处理格式
        ]
        
        for idx, (metric_name, metric_func) in enumerate(metrics_to_rank):
            model_scores = [(model, metric_func(results[model])) for model in models]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            sorted_models = [m[0] for m in model_scores]
            sorted_scores = [m[1] for m in model_scores]
            
            colors = [self.palette[i % len(self.palette)] for i in range(len(sorted_models))]
            bars = axes[idx].barh(sorted_models, sorted_scores, alpha=0.8, color=colors, 
                                 edgecolor='black', linewidth=1.5)
            axes[idx].set_xlabel(metric_name, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric_name} Ranking', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x', linestyle='--')
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            axes[idx].set_xlim([0, max(sorted_scores) * 1.1 if max(sorted_scores) > 0 else 1])
            
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                axes[idx].text(score + max(sorted_scores)*0.01, bar.get_y() + bar.get_height()/2,
                             f'{score:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_ranking')
        plt.close()
        
        # 10. 综合性能得分（加权平均）
        fig, ax = plt.subplots(figsize=(12, 6))
        # 计算综合得分：Mean IoU * 0.4 + Mean Dice * 0.4 + Accuracy * 0.2
        # 使用CSV中的实际值，不重新计算
        composite_scores = []
        for model in models:
            mean_iou = results[model].get('mean_iou', results[model].get('accuracy', 0))  # 使用CSV中的mIoU
            mean_dice = results[model].get('mean_dice', 0)  # 使用CSV中的mDice
            accuracy = results[model].get('accuracy', 0)  # 使用CSV中的"细胞活力准确率"
            composite = mean_iou * 0.4 + mean_dice * 0.4 + accuracy * 0.2
            composite_scores.append(composite)
        
        colors = [self.palette[i % len(self.palette)] for i in range(len(models))]
        bars = ax.bar(models, composite_scores, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Composite Performance Score\n(Mean IoU×0.4 + Mean Dice×0.4 + Accuracy×0.2)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, max(composite_scores) * 1.15 if max(composite_scores) > 0 else 1])
        
        for i, (bar, score) in enumerate(zip(bars, composite_scores)):
            ax.text(bar.get_x() + bar.get_width()/2., score + max(composite_scores)*0.02,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_composite')
        plt.close()
        
        # 11. 类别平衡性分析（Live vs Dead性能差异）
        fig, ax = plt.subplots(figsize=(12, 6))
        balance_scores = []  # 越小越好，表示两个类别性能更平衡
        for model in models:
            # 正确处理数据格式：[background, live, dead]
            iou_list = results[model].get('iou', [0, 0, 0])
            if len(iou_list) == 2:
                iou_list = [0.0] + iou_list
            live_iou = iou_list[1] if len(iou_list) > 1 else 0.0
            dead_iou = iou_list[2] if len(iou_list) > 2 else 0.0
            balance = abs(live_iou - dead_iou)  # 性能差异
            balance_scores.append(balance)
        
        colors = [self.palette[i % len(self.palette)] for i in range(len(models))]
        bars = ax.bar(models, balance_scores, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Performance Gap (|Live IoU - Dead IoU|)', fontsize=12, fontweight='bold')
        ax.set_title('Class Balance Analysis\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (bar, score) in enumerate(zip(bars, balance_scores)):
            ax.text(bar.get_x() + bar.get_width()/2., score + max(balance_scores)*0.02,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison_balance')
        plt.close()
        
        # 12. 详细指标表格可视化
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for model in models:
            # 正确处理数据格式：[background, live, dead]
            iou_list = results[model].get('iou', [0, 0, 0])
            dice_list = results[model].get('dice', [0, 0, 0])
            if len(iou_list) == 2:
                iou_list = [0.0] + iou_list
            if len(dice_list) == 2:
                dice_list = [0.0] + dice_list
            acc = results[model].get('accuracy', 0)
            mean_iou = results[model].get('mean_iou', acc)
            mean_dice = results[model].get('mean_dice', 0)
            table_data.append([
                model,
                f"{acc:.4f}",
                f"{iou_list[1]:.4f}" if len(iou_list) > 1 else "0.0000",  # Live IoU
                f"{iou_list[2]:.4f}" if len(iou_list) > 2 else "0.0000",  # Dead IoU
                f"{mean_iou:.4f}",  # Mean IoU (使用CSV值)
                f"{dice_list[1]:.4f}" if len(dice_list) > 1 else "0.0000",  # Live Dice
                f"{dice_list[2]:.4f}" if len(dice_list) > 2 else "0.0000",  # Dead Dice
                f"{mean_dice:.4f}"  # Mean Dice (使用CSV值)
            ])
        
        columns = ['Model', 'Accuracy', 'Live IoU', 'Dead IoU', 'Mean IoU', 
                  'Live Dice', 'Dead Dice', 'Mean Dice']
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', 
                        loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        ax.set_title('Detailed Model Performance Table', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        self._save_figure(fig, 'comparison_table')
        plt.close()
        
        print(f"已生成12+张模型对比可视化图！")

    def load_evaluation_results(self) -> pd.DataFrame:
        """从evaluation_results.csv加载评估结果
        
        Returns:
            DataFrame包含所有模型的评估指标
        """
        csv_path = os.path.join(self.save_dir, 'evaluation_results.csv')
        if not os.path.exists(csv_path):
            print(f"警告: 未找到evaluation_results.csv文件: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"已从evaluation_results.csv加载评估结果: {len(df)}个模型")
        return df
    
    def plot_comprehensive_comparison_from_csv(self):
        """从evaluation_results.csv加载数据并生成可视化
        
        此函数确保可视化使用的数据与evaluation_results.csv完全一致
        """
        df = self.load_evaluation_results()
        if df.empty:
            print("错误: 无法加载evaluation_results.csv，跳过可视化")
            return
        
        # 转换DataFrame为可视化所需的格式
        # 确保所有数据完全来自CSV文件，不使用任何计算值
        results = {}
        for _, row in df.iterrows():
            model_name = row['模型']
            # 直接从CSV读取所有值，确保完全一致
            bg_iou = row.get('语义分割-背景 IoU', 0.0) if '语义分割-背景 IoU' in df.columns else 0.0
            bg_dice = row.get('语义分割-背景 Dice', 0.0) if '语义分割-背景 Dice' in df.columns else 0.0
            live_iou = row.get('语义分割-活细胞 IoU', 0.0) if '语义分割-活细胞 IoU' in df.columns else 0.0
            dead_iou = row.get('语义分割-死细胞 IoU', 0.0) if '语义分割-死细胞 IoU' in df.columns else 0.0
            live_dice = row.get('语义分割-活细胞 Dice', 0.0) if '语义分割-活细胞 Dice' in df.columns else 0.0
            dead_dice = row.get('语义分割-死细胞 Dice', 0.0) if '语义分割-死细胞 Dice' in df.columns else 0.0
            mean_iou = row.get('语义分割 mIoU', 0.0) if '语义分割 mIoU' in df.columns else 0.0
            mean_dice = row.get('语义分割 mDice', 0.0) if '语义分割 mDice' in df.columns else 0.0
            
            results[model_name] = {
                'iou': [
                    float(bg_iou) if pd.notna(bg_iou) else 0.0,  # background (0)
                    float(live_iou) if pd.notna(live_iou) else 0.0,  # live (1)
                    float(dead_iou) if pd.notna(dead_iou) else 0.0   # dead (2)
                ],
                'dice': [
                    float(bg_dice) if pd.notna(bg_dice) else 0.0,  # background (0)
                    float(live_dice) if pd.notna(live_dice) else 0.0,  # live (1)
                    float(dead_dice) if pd.notna(dead_dice) else 0.0   # dead (2)
                ],
                'mean_iou': float(mean_iou) if pd.notna(mean_iou) else 0.0,  # 从CSV直接读取，不使用计算值
                'mean_dice': float(mean_dice) if pd.notna(mean_dice) else 0.0,  # 从CSV直接读取，不使用计算值
                'accuracy': float(row.get('细胞活力准确率', 0.0)) if '细胞活力准确率' in df.columns and pd.notna(row.get('细胞活力准确率', 0.0)) else 0.0,  # 使用CSV中的"细胞活力准确率"
                'live_cell_acc': float(row.get('活细胞检测准确率 (Precision)', 0.0)) if '活细胞检测准确率 (Precision)' in df.columns and pd.notna(row.get('活细胞检测准确率 (Precision)', 0.0)) else 0.0,
                'dead_cell_acc': float(row.get('死细胞检测准确率 (Precision)', 0.0)) if '死细胞检测准确率 (Precision)' in df.columns and pd.notna(row.get('死细胞检测准确率 (Precision)', 0.0)) else 0.0,
                'live_cell_recall': float(row.get('活细胞召回率 (Recall)', 0.0)) if '活细胞召回率 (Recall)' in df.columns and pd.notna(row.get('活细胞召回率 (Recall)', 0.0)) else 0.0,
                'dead_cell_recall': float(row.get('死细胞召回率 (Recall)', 0.0)) if '死细胞召回率 (Recall)' in df.columns and pd.notna(row.get('死细胞召回率 (Recall)', 0.0)) else 0.0,
                'viability_acc': float(row.get('细胞活力准确率', 0.0)) if '细胞活力准确率' in df.columns and pd.notna(row.get('细胞活力准确率', 0.0)) else 0.0,
                'instance_live_iou': float(row.get('实例分割-活细胞 IoU', 0.0)) if '实例分割-活细胞 IoU' in df.columns and pd.notna(row.get('实例分割-活细胞 IoU', 0.0)) else 0.0,
                'instance_dead_iou': float(row.get('实例分割-死细胞 IoU', 0.0)) if '实例分割-死细胞 IoU' in df.columns and pd.notna(row.get('实例分割-死细胞 IoU', 0.0)) else 0.0,
                'bbox_map': float(row.get('bbox mAP', 0.0)) if 'bbox mAP' in df.columns and pd.notna(row.get('bbox mAP', 0.0)) else 0.0,
                'segm_map': float(row.get('segm mAP', 0.0)) if 'segm mAP' in df.columns and pd.notna(row.get('segm mAP', 0.0)) else 0.0
            }
        
        # 使用转换后的数据生成可视化
        self.plot_comprehensive_comparison(results)


    def plot_roc_curves(self, probs_all, masks_true, model_name: str):
        """Plot ROC curves for each class
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        Probs format: [background_prob, live_prob, dead_prob] for each pixel
        """
        # 为3个类别创建3个子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for class_idx in range(len(self.class_names)):  # 0=background, 1=live, 2=dead
            # Collect probabilities and labels for this class
            y_true_binary = []
            y_scores = []

            for probs, mask_true in zip(probs_all, masks_true):
                # Filter out ignore index
                valid_mask = (mask_true != 255)
                if valid_mask.sum() == 0:
                    continue

                # Flatten and filter
                # probs shape: [3, H, W] where [0]=background, [1]=live, [2]=dead
                if probs.shape[0] > class_idx:
                    prob_flat = probs[class_idx].flatten()[valid_mask.flatten()]
                    true_flat = (mask_true.flatten()[valid_mask.flatten()] == class_idx).astype(int)

                    y_scores.extend(prob_flat)
                    y_true_binary.extend(true_flat)

            if len(y_scores) == 0:
                continue

            y_true_binary = np.array(y_true_binary)
            y_scores = np.array(y_scores)

            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)

            axes[class_idx].plot(fpr, tpr, color=self.palette[class_idx],
                               linewidth=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[class_idx].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            axes[class_idx].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            axes[class_idx].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            axes[class_idx].set_title(f'{self.class_names[class_idx]} ROC Curve',
                                    fontsize=14, fontweight='bold')
            axes[class_idx].legend(loc='lower right', frameon=True, shadow=True)
            axes[class_idx].grid(True, alpha=0.3, linestyle='--')
            axes[class_idx].spines['top'].set_visible(False)
            axes[class_idx].spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_roc_curves')
        plt.close()

    def plot_pr_curves(self, probs_all, masks_true, model_name: str):
        """Plot Precision-Recall curves for each class
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        Probs format: [background_prob, live_prob, dead_prob] for each pixel
        """
        # 为3个类别创建3个子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for class_idx in range(len(self.class_names)):  # 0=background, 1=live, 2=dead
            # Collect probabilities and labels for this class
            y_true_binary = []
            y_scores = []

            for probs, mask_true in zip(probs_all, masks_true):
                # Filter out ignore index
                valid_mask = (mask_true != 255)
                if valid_mask.sum() == 0:
                    continue

                # Flatten and filter
                # probs shape: [3, H, W] where [0]=background, [1]=live, [2]=dead
                if probs.shape[0] > class_idx:
                    prob_flat = probs[class_idx].flatten()[valid_mask.flatten()]
                    true_flat = (mask_true.flatten()[valid_mask.flatten()] == class_idx).astype(int)

                    y_scores.extend(prob_flat)
                    y_true_binary.extend(true_flat)

            if len(y_scores) == 0:
                continue

            y_true_binary = np.array(y_true_binary)
            y_scores = np.array(y_scores)

            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            avg_precision = average_precision_score(y_true_binary, y_scores)

            axes[class_idx].plot(recall, precision, color=self.palette[class_idx],
                               linewidth=2.5, label=f'PR curve (AP = {avg_precision:.3f})')
            axes[class_idx].set_xlabel('Recall', fontsize=12, fontweight='bold')
            axes[class_idx].set_ylabel('Precision', fontsize=12, fontweight='bold')
            axes[class_idx].set_title(f'{self.class_names[class_idx]} PR Curve',
                                    fontsize=14, fontweight='bold')
            axes[class_idx].legend(loc='lower left', frameon=True, shadow=True)
            axes[class_idx].grid(True, alpha=0.3, linestyle='--')
            axes[class_idx].spines['top'].set_visible(False)
            axes[class_idx].spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_pr_curves')
        plt.close()

    def plot_class_wise_metrics(self, history: Dict, model_name: str):
        """绘制每个类别的指标变化（修复：只显示2个类别）"""
        if 'val_iou' not in history or len(history['val_iou']) == 0:
            return

        val_iou = np.array(history['val_iou'])
        val_dice = np.array(history['val_dice'])

        # 只显示2个类别：活细胞和死细胞
        num_classes = min(len(self.class_names), val_iou.shape[1])
        fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 6))
        if num_classes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for class_idx in range(num_classes):
            ax = axes[class_idx]
            epochs = range(1, len(val_iou) + 1)

            # 处理NaN值
            iou_values = val_iou[:, class_idx]
            dice_values = val_dice[:, class_idx]
            
            # 过滤NaN值
            valid_iou = ~np.isnan(iou_values)
            valid_dice = ~np.isnan(dice_values)

            if valid_iou.any():
                ax.plot(np.array(epochs)[valid_iou], iou_values[valid_iou], 
                       marker='o', linewidth=2.5, color=self.palette[class_idx], 
                       label='IoU', markersize=6)
            if valid_dice.any():
                ax.plot(np.array(epochs)[valid_dice], dice_values[valid_dice], 
                       marker='s', linewidth=2.5, color=self.palette[class_idx + 4], 
                       label='Dice', markersize=6)

            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name} - {self.class_names[class_idx]} Metrics',
                        fontsize=14, fontweight='bold')
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim([0, 1])  # 设置y轴范围为[0, 1]

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_class_wise_metrics')
        plt.close()

    def plot_learning_rate_schedule(self, lr_history: List[float], model_name: str):
        """Plot learning rate schedule"""
        if not lr_history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(lr_history) + 1)

        ax.plot(epochs, lr_history, marker='o', linewidth=2.5,
               color=self.palette[0], markersize=6)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_lr_schedule')
        plt.close()

    def plot_per_image_metrics(self, masks_true, masks_pred, model_name: str):
        """Plot per-image metrics distribution"""
        if not masks_true or not masks_pred:
            return

        # Calculate metrics for each image
        ious = []
        dices = []
        accs = []

        for mask_true, mask_pred in zip(masks_true, masks_pred):
            # Filter out ignore index
            valid_mask = (mask_true != 255)
            if valid_mask.sum() == 0:
                continue

            true_valid = mask_true[valid_mask]
            pred_valid = mask_pred[valid_mask]

            # Accuracy
            acc = (true_valid == pred_valid).mean()
            accs.append(acc)

            # IoU and Dice for each class
            class_ious = []
            class_dices = []
            for class_id in range(len(self.class_names)):
                true_class = (true_valid == class_id)
                pred_class = (pred_valid == class_id)

                intersection = (true_class & pred_class).sum()
                union = (true_class | pred_class).sum()

                if union > 0:
                    iou = intersection / union
                    dice = 2 * intersection / (true_class.sum() + pred_class.sum())
                    class_ious.append(iou)
                    class_dices.append(dice)

            if class_ious:
                ious.append(np.mean(class_ious))
                dices.append(np.mean(class_dices))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # IoU distribution
        axes[0].hist(ious, bins=20, color=self.palette[0], alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(ious), color=self.palette[8], linestyle='--',
                       linewidth=2, label=f'Mean = {np.mean(ious):.3f}')
        axes[0].set_xlabel('IoU Score', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{model_name} - IoU Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(frameon=True, shadow=True)
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Dice distribution
        axes[1].hist(dices, bins=20, color=self.palette[4], alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(dices), color=self.palette[8], linestyle='--',
                       linewidth=2, label=f'Mean = {np.mean(dices):.3f}')
        axes[1].set_xlabel('Dice Score', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{model_name} - Dice Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(frameon=True, shadow=True)
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        # Accuracy distribution
        axes[2].hist(accs, bins=20, color=self.palette[6], alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(accs), color=self.palette[8], linestyle='--',
                       linewidth=2, label=f'Mean = {np.mean(accs):.3f}')
        axes[2].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[2].set_title(f'{model_name} - Accuracy Distribution', fontsize=14, fontweight='bold')
        axes[2].legend(frameon=True, shadow=True)
        axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_per_image_metrics')
        plt.close()


    def plot_sample_predictions_grid(self, images, masks_true, masks_pred,
                                    filenames, model_name: str, num_samples=16):
        """Plot a grid of sample predictions"""
        num_samples = min(num_samples, len(images))
        rows = int(np.ceil(num_samples / 4))

        fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            row = i // 4
            col = i % 4

            # Original image with overlay
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)

            if isinstance(masks_pred[i], torch.Tensor):
                mask_pred_np = masks_pred[i].cpu().numpy()
            else:
                mask_pred_np = masks_pred[i]

            mask_pred_colored = self.mask_to_color(mask_pred_np)
            overlay = (img * 0.5 + mask_pred_colored * 0.5)
            overlay = np.clip(overlay, 0, 1)

            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'Sample {i+1}', fontsize=10, fontweight='bold')
            axes[row, col].axis('off')

        # Hide empty subplots
        for i in range(num_samples, rows * 4):
            row = i // 4
            col = i % 4
            axes[row, col].axis('off')

        plt.suptitle(f'{model_name} - Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_sample_grid')
        plt.close()

    def plot_error_analysis(self, masks_true, masks_pred, model_name: str):
        """绘制误差分析图（英文标签，中文注释）
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 计算误差 - 3个类别：背景(0)、活细胞(1)、死细胞(2)
        num_classes = 3
        all_errors = []
        class_errors = [[] for _ in range(num_classes)]

        for mask_true, mask_pred in zip(masks_true, masks_pred):
            # 只统计有效区域（忽略255）
            valid_mask = (mask_true != 255)
            if valid_mask.sum() > 0:
                error_map = ((mask_true != mask_pred) & valid_mask).astype(float)
                all_errors.append(error_map.sum() / valid_mask.sum())

                # 按类别统计误差
                for class_idx in range(num_classes):
                    class_mask = (mask_true == class_idx) & valid_mask
                    if class_mask.sum() > 0:
                        class_error = ((mask_true[class_mask] != mask_pred[class_mask])).sum() / class_mask.sum()
                        class_errors[class_idx].append(class_error)

        # Overall error distribution
        axes[0, 0].hist(all_errors, bins=20, color=self.palette[0], alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(all_errors), color=self.palette[8], linestyle='--',
                          linewidth=2, label=f'Mean = {np.mean(all_errors):.3f}')
        axes[0, 0].set_xlabel('Error Rate', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Overall Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend(frameon=True, shadow=True)
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)

        # Class-wise error rates - 显示3个类别
        class_error_means = [np.mean(errors) if errors else 0 for errors in class_errors]
        x = np.arange(num_classes)
        bars = axes[0, 1].bar(x, class_error_means, color=self.palette[:num_classes], alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Mean Error Rate', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Class-wise Error Rates', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.class_names, rotation=15, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)

        # Add value labels on bars
        for bar, value in zip(bars, class_error_means):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Error boxplot by class - 显示3个类别
        # 只显示有数据的类别
        valid_errors = [errors for errors in class_errors if errors]
        valid_labels = [self.class_names[i] for i, errors in enumerate(class_errors) if errors]
        valid_indices = [i for i, errors in enumerate(class_errors) if errors]
        
        if valid_errors:
            bp = axes[1, 0].boxplot(valid_errors,
                                    labels=valid_labels,
                                patch_artist=True)
            for patch, idx in zip(bp['boxes'], valid_indices):
                patch.set_facecolor(self.palette[idx])
            patch.set_alpha(0.7)
            axes[1, 0].set_xticklabels(valid_labels, rotation=15, ha='right')
        else:
            # 如果没有数据，显示空图
            axes[1, 0].text(0.5, 0.5, 'No error data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        axes[1, 0].set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Error Rate Distribution by Class', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)

        # Confusion between classes - 统计3个类别
        confusion_counts = np.zeros((num_classes, num_classes))
        for mask_true, mask_pred in zip(masks_true, masks_pred):
            # 只统计有效区域（忽略255）
            valid_mask = (mask_true != 255)
            for true_class in range(num_classes):
                for pred_class in range(num_classes):
                    mask = (mask_true == true_class) & (mask_pred == pred_class) & valid_mask
                    confusion_counts[true_class, pred_class] += mask.sum()

        # Normalize by row
        confusion_norm = confusion_counts / (confusion_counts.sum(axis=1, keepdims=True) + 1e-10)

        im = axes[1, 1].imshow(confusion_norm, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xticks(np.arange(num_classes))
        axes[1, 1].set_yticks(np.arange(num_classes))
        axes[1, 1].set_xticklabels(self.class_names, rotation=15, ha='right')
        axes[1, 1].set_yticklabels(self.class_names)
        axes[1, 1].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('True Class', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Normalized Confusion Heatmap', fontsize=14, fontweight='bold')

        # Add text annotations - 显示3个类别
        for i in range(num_classes):
            for j in range(num_classes):
                text = axes[1, 1].text(j, i, f'{confusion_norm[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontweight='bold')

        plt.colorbar(im, ax=axes[1, 1], label='Proportion')

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_error_analysis')
        plt.close()

    def save_training_history_csv(self, history: Dict, model_name: str):
        """Save training history to CSV"""
        epochs = range(1, len(history['train_loss']) + 1)

        data = {
            'Epoch': epochs,
            'Train_Loss': history['train_loss'],
            'Val_Loss': history['val_loss'],
            'Train_Acc': history['train_acc'],
            'Val_Acc': history['val_acc'],
        }

        # Add IoU and Dice for each class
        if 'val_iou' in history and len(history['val_iou']) > 0:
            val_iou = np.array(history['val_iou'])
            val_dice = np.array(history['val_dice'])

            for i, class_name in enumerate(self.class_names):
                data[f'{class_name}_IoU'] = val_iou[:, i]
                data[f'{class_name}_Dice'] = val_dice[:, i]

        df = pd.DataFrame(data)
        save_path = os.path.join(self.save_dir, f'{model_name}_training_history.csv')
        df.to_csv(save_path, index=False)
        print(f"Training history CSV saved: {save_path}")

        return df

    def plot_learning_rate_schedule(self, history: Dict, model_name: str):
        """Plot learning rate schedule"""
        if 'learning_rate' not in history or len(history['learning_rate']) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['learning_rate']) + 1)

        ax.plot(epochs, history['learning_rate'], linewidth=2.5, color=self.palette[3], marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_learning_rate')
        plt.close()

    def plot_gradient_flow(self, model, model_name: str):
        """Plot gradient flow through the network"""
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())

        if not layers:
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color=self.palette[0], label='Max Gradient')
        ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color=self.palette[4], label='Mean Gradient')
        ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        ax.set_xticks(range(0, len(ave_grads), max(1, len(ave_grads)//20)))
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers)//20))], rotation=45, ha='right', fontsize=8)
        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom=-0.001, top=max(max_grads)*1.2)
        ax.set_xlabel('Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Magnitude', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Gradient Flow', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_gradient_flow')
        plt.close()

    def plot_feature_importance(self, masks_true, masks_pred, images, model_name: str):
        """Plot feature importance heatmap"""
        # 计算每个像素位置的预测准确性
        correct_pixels = (np.array(masks_pred) == np.array(masks_true)).astype(float)

        # 平均所有图像的准确性
        avg_accuracy_map = np.mean(correct_pixels, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy heatmap
        im1 = axes[0].imshow(avg_accuracy_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_title('Spatial Accuracy Heatmap', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Accuracy', fraction=0.046, pad=0.04)

        # Error heatmap
        error_map = 1 - avg_accuracy_map
        im2 = axes[1].imshow(error_map, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Spatial Error Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Error Rate', fraction=0.046, pad=0.04)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_spatial_analysis')
        plt.close()

    def plot_class_distribution(self, masks_true, masks_pred, model_name: str):
        """Plot class distribution comparison
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # True distribution - 3个类别：背景(0)、活细胞(1)、死细胞(2)
        num_classes = 3
        true_counts = []
        pred_counts = []
        for i in range(num_classes):
            true_count = sum((mask == i).sum() for mask in masks_true)
            pred_count = sum((mask == i).sum() for mask in masks_pred)
            true_counts.append(true_count)
            pred_counts.append(pred_count)

        x = np.arange(num_classes)
        width = 0.35

        bars1 = axes[0].bar(x - width/2, true_counts, width, label='Ground Truth',
                           color=self.palette[0], alpha=0.8, edgecolor='black')
        bars2 = axes[0].bar(x + width/2, pred_counts, width, label='Prediction',
                           color=self.palette[4], alpha=0.8, edgecolor='black')

        axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Pixel Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, rotation=15, ha='right')
        axes[0].legend(fontsize=10, frameon=True, shadow=True)
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Percentage distribution
        true_pct = np.array(true_counts) / (sum(true_counts) + 1e-6) * 100
        pred_pct = np.array(pred_counts) / (sum(pred_counts) + 1e-6) * 100

        bars3 = axes[1].bar(x - width/2, true_pct, width, label='Ground Truth',
                           color=self.palette[1], alpha=0.8, edgecolor='black')
        bars4 = axes[1].bar(x + width/2, pred_pct, width, label='Prediction',
                           color=self.palette[5], alpha=0.8, edgecolor='black')

        axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Class Distribution Percentage', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names, rotation=15, ha='right')
        axes[1].legend(fontsize=10, frameon=True, shadow=True)
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_class_distribution')
        plt.close()


    def plot_boundary_accuracy(self, masks_true, masks_pred, model_name: str):
        """Plot boundary detection accuracy"""
        from scipy.ndimage import binary_dilation, binary_erosion

        boundary_ious = []
        interior_ious = []

        for mask_true, mask_pred in zip(masks_true, masks_pred):
            # 计算边界
            for class_id in range(len(self.class_names)):
                true_mask = (mask_true == class_id)
                pred_mask = (mask_pred == class_id)

                if true_mask.sum() == 0:
                    continue

                # 边界 = 膨胀 - 腐蚀
                true_boundary = binary_dilation(true_mask) & ~binary_erosion(true_mask)
                pred_boundary = binary_dilation(pred_mask) & ~binary_erosion(pred_mask)

                # 内部 = 腐蚀
                true_interior = binary_erosion(true_mask, iterations=2)
                pred_interior = binary_erosion(pred_mask, iterations=2)

                # 计算IoU
                if true_boundary.sum() > 0:
                    boundary_iou = (true_boundary & pred_boundary).sum() / (true_boundary | pred_boundary).sum()
                    boundary_ious.append(boundary_iou)

                if true_interior.sum() > 0:
                    interior_iou = (true_interior & pred_interior).sum() / (true_interior | pred_interior).sum()
                    interior_ious.append(interior_iou)

        fig, ax = plt.subplots(figsize=(10, 6))

        data = [boundary_ious, interior_ious]
        labels = ['Boundary', 'Interior']

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], [self.palette[0], self.palette[4]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

        # 设置中位线颜色
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Boundary vs Interior Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 添加均值标注
        means = [np.mean(d) for d in data]
        for i, (mean, label) in enumerate(zip(means, labels)):
            ax.text(i+1, mean, f'μ={mean:.3f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_boundary_accuracy')
        plt.close()

    def plot_size_based_performance(self, masks_true, masks_pred, model_name: str):
        """Plot performance based on object size"""
        from scipy.ndimage import label as connected_components

        size_ranges = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, float('inf'))]
        range_labels = ['Tiny\n(0-100)', 'Small\n(100-500)', 'Medium\n(500-1K)', 'Large\n(1K-5K)', 'Huge\n(5K+)']

        ious_by_size = [[] for _ in range(len(size_ranges))]

        for mask_true, mask_pred in zip(masks_true, masks_pred):
            for class_id in range(len(self.class_names)):
                true_mask = (mask_true == class_id).astype(int)
                pred_mask = (mask_pred == class_id).astype(int)

                # 连通组件分析
                labeled_true, num_true = connected_components(true_mask)

                for obj_id in range(1, num_true + 1):
                    obj_mask = (labeled_true == obj_id)
                    obj_size = obj_mask.sum()

                    # 找到对应的预测
                    obj_pred = pred_mask[obj_mask]
                    iou = obj_pred.sum() / obj_size if obj_size > 0 else 0

                    # 分配到对应的大小范围
                    for i, (min_size, max_size) in enumerate(size_ranges):
                        if min_size <= obj_size < max_size:
                            ious_by_size[i].append(iou)
                            break

        fig, ax = plt.subplots(figsize=(12, 6))

        # 过滤空数据
        valid_data = []
        valid_labels = []
        valid_colors = []
        for i, (data, label) in enumerate(zip(ious_by_size, range_labels)):
            if data:
                valid_data.append(data)
                valid_labels.append(label)
                valid_colors.append(self.palette[i % len(self.palette)])

        if valid_data:
            bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], valid_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

            for median in bp['medians']:
                median.set_color('red')
                median.set_linewidth(2)

        ax.set_xlabel('Object Size (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Performance by Object Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_size_performance')
        plt.close()

    def plot_calibration_curve(self, probs_all, masks_true, model_name: str):
        """Plot calibration curve (reliability diagram)
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        """
        if probs_all is None or len(probs_all) == 0:
            return

        # 3个类别：背景、活细胞、死细胞
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        n_bins = 10

        for class_id, class_name in enumerate(self.class_names):
            # 收集该类别的概率和真实标签
            class_probs = []
            class_labels = []

            for probs, mask_true in zip(probs_all, masks_true):
                if probs.shape[0] > class_id:
                    class_prob = probs[class_id].flatten()
                    class_label = (mask_true == class_id).flatten()
                    class_probs.extend(class_prob)
                    class_labels.extend(class_label)

            class_probs = np.array(class_probs)
            class_labels = np.array(class_labels)

            # 计算校准曲线
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_accs = []
            bin_confs = []
            bin_counts = []

            for i in range(n_bins):
                mask = (class_probs >= bin_edges[i]) & (class_probs < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_acc = class_labels[mask].mean()
                    bin_conf = class_probs[mask].mean()
                    bin_accs.append(bin_acc)
                    bin_confs.append(bin_conf)
                    bin_counts.append(mask.sum())
                else:
                    bin_accs.append(0)
                    bin_confs.append(bin_centers[i])
                    bin_counts.append(0)

            # 绘制校准曲线
            axes[0].plot(bin_confs, bin_accs, marker='o', linewidth=2.5, markersize=8,
                        label=class_name, color=self.palette[class_id])

        # 完美校准线
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        axes[0].set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        axes[0].set_title('Calibration Curve', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10, frameon=True, shadow=True)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # 置信度分布
        for class_id, class_name in enumerate(self.class_names):
            class_probs = []
            for probs in probs_all:
                if probs.shape[0] > class_id:
                    class_probs.extend(probs[class_id].flatten())

            axes[1].hist(class_probs, bins=50, alpha=0.6, label=class_name,
                        color=self.palette[class_id], edgecolor='black')

        axes[1].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10, frameon=True, shadow=True)
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_calibration')
        plt.close()



    def create_paper_figures(self, images, masks_true, masks_pred, model_name: str, filenames=None):
        """Create publication-quality figures for research papers
        四列：预处理前的原图、预处理后的图、真实标注、预测标注
        """

        # Figure 1: Multi-panel comparison (4 samples, 4 columns)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        for i in range(min(4, len(images))):
            # 第1列：预处理前的原图（直接从data目录加载）
            original_img = None
            if filenames and i < len(filenames):
                # 直接从data目录加载原始图像
                try:
                    from PIL import Image
                    import os
                    # 直接从data目录获取原图（跨平台兼容路径）
                    img_path = os.path.join('data', filenames[i])
                    if os.path.exists(img_path):
                        original_img = np.array(Image.open(img_path).convert('RGB'))
                        original_img = original_img.astype(np.float32) / 255.0
                    else:
                        print(f"警告: 未找到原图文件: {img_path}，使用反归一化近似")
                except Exception as e:
                    print(f"警告: 加载原图失败: {e}，使用反归一化近似")
                    pass  # 如果加载失败，使用反归一化近似
            
            # 如果无法加载原图，使用反归一化来近似预处理前的图像
            if original_img is None:
                # images是CHW格式的预处理后图像（已经归一化）
                img_preprocessed = images[i]
                if isinstance(img_preprocessed, np.ndarray):
                    # 如果是CHW格式，转换为HWC
                    if len(img_preprocessed.shape) == 3 and img_preprocessed.shape[0] == 3:
                        img_preprocessed = img_preprocessed.transpose(1, 2, 0)
                    
                    # 反归一化：恢复预处理前的图像
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    
                    if img_preprocessed.dtype != np.float32:
                        img_preprocessed = img_preprocessed.astype(np.float32)
                    
                    # 反归一化得到预处理前的图像（近似）
                    original_img = img_preprocessed * std + mean
                    original_img = np.clip(original_img, 0, 1)
            
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title('预处理前的原图', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')

            # 第2列：预处理后的图（归一化后的图像，需要反归一化以便显示）
            img_preprocessed = images[i]
            if isinstance(img_preprocessed, np.ndarray):
                # 如果是CHW格式，转换为HWC
                if len(img_preprocessed.shape) == 3 and img_preprocessed.shape[0] == 3:
                    img_preprocessed = img_preprocessed.transpose(1, 2, 0)
                
                # 图像已经通过Normalize()处理，需要反归一化以便显示
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                if img_preprocessed.dtype != np.float32:
                    img_preprocessed = img_preprocessed.astype(np.float32)
                
                # 反归一化以便显示（预处理后的图像，但显示时需要反归一化）
                img_for_display = img_preprocessed * std + mean
                img_for_display = np.clip(img_for_display, 0, 1)
            else:
                img_for_display = img_preprocessed
            
            axes[i, 1].imshow(img_for_display)
            axes[i, 1].set_title('预处理后的图', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')

            # 第3列：真实标注
            if isinstance(masks_true[i], torch.Tensor):
                mask_true_np = masks_true[i].cpu().numpy()
            else:
                mask_true_np = np.array(masks_true[i])
            mask_true_colored = self._colorize_mask(mask_true_np)
            axes[i, 2].imshow(mask_true_colored)
            axes[i, 2].set_title('真实标注', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')

            # 第4列：预测标注
            if isinstance(masks_pred[i], torch.Tensor):
                mask_pred_np = masks_pred[i].cpu().numpy()
            else:
                mask_pred_np = np.array(masks_pred[i])
            mask_pred_colored = self._colorize_mask(mask_pred_np)
            axes[i, 3].imshow(mask_pred_colored)
            axes[i, 3].set_title('预测标注', fontsize=12, fontweight='bold')
            axes[i, 3].axis('off')

        # Add legend (3个类别：背景、活细胞、死细胞)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.class_colors[0], edgecolor='black', label=self.class_names[0]),
            Patch(facecolor=self.class_colors[1], edgecolor='black', label=self.class_names[1]),
            Patch(facecolor=self.class_colors[2], edgecolor='black', label=self.class_names[2])
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3,
                  fontsize=12, frameon=True, shadow=True, bbox_to_anchor=(0.5, 0.98))

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self._save_figure(fig, f'{model_name}_paper_fig1_comparison')
        plt.close()

        # Figure 2: Overlay visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()

        for i in range(min(4, len(images))):
            # 正确处理图像格式和反归一化
            img_preprocessed = images[i]
            if isinstance(img_preprocessed, np.ndarray):
                # 如果是CHW格式，转换为HWC
                if len(img_preprocessed.shape) == 3 and img_preprocessed.shape[0] == 3:
                    img_preprocessed = img_preprocessed.transpose(1, 2, 0)
                
                # 图像已经通过Normalize()处理，需要反归一化以便显示
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                if img_preprocessed.dtype != np.float32:
                    img_preprocessed = img_preprocessed.astype(np.float32)
                
                # 反归一化以便显示
                img = img_preprocessed * std + mean
                img = np.clip(img, 0, 1)
            else:
                img = img_preprocessed
            if img.max() > 1:
                img = img / 255.0

            # Create overlay
            overlay = img.copy()
            if isinstance(masks_pred[i], torch.Tensor):
                mask_pred = masks_pred[i].cpu().numpy()
            else:
                mask_pred = np.array(masks_pred[i])

            # Overlay predictions with transparency (0=background, 1=live, 2=dead)
            for class_id in range(len(self.class_names)):
                mask_class = (mask_pred == class_id)
                color = np.array(self.class_colors[class_id][:3])  # 已经是归一化的0-1范围
                for c in range(3):
                    overlay[:, :, c][mask_class] = overlay[:, :, c][mask_class] * 0.5 + color[c] * 0.5

            axes[i].imshow(overlay)
            axes[i].set_title(f'Sample {i+1} - Prediction Overlay', fontsize=12, fontweight='bold')
            axes[i].axis('off')

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_paper_fig2_overlay')
        plt.close()

        # Figure 3: Error visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()

        for i in range(min(4, len(images))):
            # Create error map
            if isinstance(masks_true[i], torch.Tensor):
                mask_true_np = masks_true[i].cpu().numpy()
            else:
                mask_true_np = np.array(masks_true[i])
            if isinstance(masks_pred[i], torch.Tensor):
                mask_pred_np = masks_pred[i].cpu().numpy()
            else:
                mask_pred_np = np.array(masks_pred[i])
            error_map = (mask_true_np != mask_pred_np).astype(float)

            # 正确处理图像格式和反归一化
            img_preprocessed = images[i]
            if isinstance(img_preprocessed, np.ndarray):
                # 如果是CHW格式，转换为HWC
                if len(img_preprocessed.shape) == 3 and img_preprocessed.shape[0] == 3:
                    img_preprocessed = img_preprocessed.transpose(1, 2, 0)
                
                # 图像已经通过Normalize()处理，需要反归一化以便显示
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                if img_preprocessed.dtype != np.float32:
                    img_preprocessed = img_preprocessed.astype(np.float32)
                
                # 反归一化以便显示
                img = img_preprocessed * std + mean
                img = np.clip(img, 0, 1)
            else:
                img = img_preprocessed
            if img.max() > 1:
                img = img / 255.0

            axes[i].imshow(img, alpha=0.7)
            im = axes[i].imshow(error_map, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            axes[i].set_title(f'Sample {i+1} - Error Map', fontsize=12, fontweight='bold')
            axes[i].axis('off')

        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                           fraction=0.05, pad=0.05, aspect=30)
        cbar.set_label('Error (Red = Incorrect)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_paper_fig3_errors')
        plt.close()

        # Figure 4: Detailed single sample analysis
        if len(images) > 0:
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

            idx = 0
            img = images[idx].transpose(1, 2, 0)
            if img.max() > 1:
                img = img / 255.0

            # Original
            ax1 = fig.add_subplot(gs[:, 0])
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # Ground truth
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(self._colorize_mask(masks_true[idx]))
            ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
            ax2.axis('off')

            # Prediction
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(self._colorize_mask(masks_pred[idx]))
            ax3.set_title('Prediction', fontsize=14, fontweight='bold')
            ax3.axis('off')

            # Error map
            ax4 = fig.add_subplot(gs[0, 3])
            error_map = (masks_true[idx] != masks_pred[idx]).astype(float)
            ax4.imshow(error_map, cmap='Reds', vmin=0, vmax=1)
            ax4.set_title('Error Map', fontsize=14, fontweight='bold')
            ax4.axis('off')

            # Class-wise breakdown (0=background, 1=live, 2=dead)
            ax5 = fig.add_subplot(gs[1, 1])
            live_mask = (masks_pred[idx] == 1)  # 类别1是活细胞
            ax5.imshow(live_mask, cmap='Greens', vmin=0, vmax=1)
            ax5.set_title(f'{self.class_names[1]}', fontsize=12, fontweight='bold')
            ax5.axis('off')

            ax6 = fig.add_subplot(gs[1, 2])
            dead_mask = (masks_pred[idx] == 2)  # 类别2是死细胞
            ax6.imshow(dead_mask, cmap='Reds', vmin=0, vmax=1)
            ax6.set_title(f'{self.class_names[2]}', fontsize=12, fontweight='bold')
            ax6.axis('off')

            # Statistics
            ax7 = fig.add_subplot(gs[1, 3])
            ax7.axis('off')

            # Calculate metrics for this sample (0=background, 1=live, 2=dead)
            iou_background = self._calculate_iou(masks_true[idx] == 0, masks_pred[idx] == 0)
            iou_live = self._calculate_iou(masks_true[idx] == 1, masks_pred[idx] == 1)
            iou_dead = self._calculate_iou(masks_true[idx] == 2, masks_pred[idx] == 2)
            accuracy = (masks_true[idx] == masks_pred[idx]).mean()

            stats_text = f"Sample Metrics:\n\n"
            stats_text += f"Accuracy: {accuracy:.3f}\n\n"
            stats_text += f"{self.class_names[0]} IoU: {iou_background:.3f}\n"
            stats_text += f"{self.class_names[1]} IoU: {iou_live:.3f}\n"
            stats_text += f"{self.class_names[2]} IoU: {iou_dead:.3f}\n\n"
            stats_text += f"Background: {(masks_pred[idx] == 0).sum()} px\n"
            stats_text += f"Live Cells: {(masks_pred[idx] == 1).sum()} px\n"
            stats_text += f"Dead Cells: {(masks_pred[idx] == 2).sum()} px\n"

            ax7.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            self._save_figure(fig, f'{model_name}_paper_fig4_detailed')
            plt.close()

    def _calculate_iou(self, mask1, mask2):
        """Calculate IoU between two binary masks"""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return intersection / union if union > 0 else 0.0

    def _colorize_mask(self, mask):
        """将mask转换为RGB彩色图像用于可视化
        Mask format: 0=background, 1=live, 2=dead, 255=unlabeled
        """
        # 确保mask是numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = np.array(mask, dtype=np.int64)
        
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.float32)

        # 处理3个有效类别（0=background, 1=live, 2=dead），忽略255
        for class_id in range(len(self.class_names)):  # 0, 1, 2
            mask_class = (mask == class_id)
            color = np.array(self.class_colors[class_id][:3])  # 已经是归一化的0-1范围
            for c in range(3):
                colored[:, :, c][mask_class] = color[c]

        return colored
    
    def plot_cell_count_comparison(self, comparison_data: List[Dict], model_name: str):
        """绘制每张图像的细胞数量和细胞活力对比图
        
        Args:
            comparison_data: 包含每张图像对比数据的列表，每个元素包含：
                - filename: 图像文件名
                - gt_live_count: 实际活细胞数
                - gt_dead_count: 实际死细胞数
                - gt_viability: 实际细胞活力 (%)
                - pred_live_count: 预测活细胞数
                - pred_dead_count: 预测死细胞数
                - pred_viability: 预测细胞活力 (%)
            model_name: 模型名称
        """
        if not comparison_data:
            print("警告: 没有对比数据可显示")
            return
        
        # 提取数据
        filenames = [d['filename'] for d in comparison_data]
        gt_live = [d['gt_live_count'] for d in comparison_data]
        gt_dead = [d['gt_dead_count'] for d in comparison_data]
        gt_viability = [d['gt_viability'] for d in comparison_data]
        pred_live = [d['pred_live_count'] for d in comparison_data]
        pred_dead = [d['pred_dead_count'] for d in comparison_data]
        pred_viability = [d['pred_viability'] for d in comparison_data]
        
        # 创建图表
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 活细胞数量对比（柱状图）
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(filenames))
        width = 0.35
        ax1.bar(x - width/2, gt_live, width, label='实际', alpha=0.8, color=self.palette[0])
        ax1.bar(x + width/2, pred_live, width, label='预测', alpha=0.8, color=self.palette[4])
        ax1.set_xlabel('图像编号', fontsize=12, fontweight='bold')
        ax1.set_ylabel('活细胞数量', fontsize=12, fontweight='bold')
        ax1.set_title('活细胞数量对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'#{i+1}' for i in range(len(filenames))], rotation=45, ha='right')
        ax1.legend(frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. 死细胞数量对比（柱状图）
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x - width/2, gt_dead, width, label='实际', alpha=0.8, color=self.palette[0])
        ax2.bar(x + width/2, pred_dead, width, label='预测', alpha=0.8, color=self.palette[4])
        ax2.set_xlabel('图像编号', fontsize=12, fontweight='bold')
        ax2.set_ylabel('死细胞数量', fontsize=12, fontweight='bold')
        ax2.set_title('死细胞数量对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'#{i+1}' for i in range(len(filenames))], rotation=45, ha='right')
        ax2.legend(frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. 细胞活力对比（柱状图）
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(x - width/2, gt_viability, width, label='实际', alpha=0.8, color=self.palette[0])
        ax3.bar(x + width/2, pred_viability, width, label='预测', alpha=0.8, color=self.palette[4])
        ax3.set_xlabel('图像编号', fontsize=12, fontweight='bold')
        ax3.set_ylabel('细胞活力 (%)', fontsize=12, fontweight='bold')
        ax3.set_title('细胞活力对比', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'#{i+1}' for i in range(len(filenames))], rotation=45, ha='right')
        ax3.legend(frameon=True, shadow=True)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_ylim([0, 105])
        
        # 4. 活细胞数量散点图（预测 vs 实际）
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(gt_live, pred_live, s=100, alpha=0.7, color=self.palette[0], edgecolors='black', linewidth=1.5)
        # 添加对角线
        max_val = max(max(gt_live) if gt_live else 0, max(pred_live) if pred_live else 0)
        if max_val > 0:
            ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='理想线')
        ax4.set_xlabel('实际活细胞数量', fontsize=12, fontweight='bold')
        ax4.set_ylabel('预测活细胞数量', fontsize=12, fontweight='bold')
        ax4.set_title('活细胞数量：预测 vs 实际', fontsize=14, fontweight='bold')
        ax4.legend(frameon=True, shadow=True)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        # 添加R值
        if len(gt_live) > 1:
            correlation = np.corrcoef(gt_live, pred_live)[0, 1]
            ax4.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. 死细胞数量散点图（预测 vs 实际）
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(gt_dead, pred_dead, s=100, alpha=0.7, color=self.palette[4], edgecolors='black', linewidth=1.5)
        max_val = max(max(gt_dead) if gt_dead else 0, max(pred_dead) if pred_dead else 0)
        if max_val > 0:
            ax5.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='理想线')
        ax5.set_xlabel('实际死细胞数量', fontsize=12, fontweight='bold')
        ax5.set_ylabel('预测死细胞数量', fontsize=12, fontweight='bold')
        ax5.set_title('死细胞数量：预测 vs 实际', fontsize=14, fontweight='bold')
        ax5.legend(frameon=True, shadow=True)
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        # 添加R值
        if len(gt_dead) > 1:
            correlation = np.corrcoef(gt_dead, pred_dead)[0, 1]
            ax5.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax5.transAxes,
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. 细胞活力散点图（预测 vs 实际）
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(gt_viability, pred_viability, s=100, alpha=0.7, color=self.palette[6], edgecolors='black', linewidth=1.5)
        ax6.plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.5, label='理想线')
        ax6.set_xlabel('实际细胞活力 (%)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('预测细胞活力 (%)', fontsize=12, fontweight='bold')
        ax6.set_title('细胞活力：预测 vs 实际', fontsize=14, fontweight='bold')
        ax6.set_xlim([0, 105])
        ax6.set_ylim([0, 105])
        ax6.legend(frameon=True, shadow=True)
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        # 添加R值
        if len(gt_viability) > 1:
            correlation = np.corrcoef(gt_viability, pred_viability)[0, 1]
            ax6.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax6.transAxes,
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 7. 详细对比表格
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        # 准备表格数据
        table_data = []
        for i, data in enumerate(comparison_data):
            table_data.append([
                f"#{i+1}",
                data['filename'][:20] + '...' if len(data['filename']) > 20 else data['filename'],
                f"{data['gt_live_count']}",
                f"{data['pred_live_count']}",
                f"{data['gt_dead_count']}",
                f"{data['pred_dead_count']}",
                f"{data['gt_viability']:.2f}%",
                f"{data['pred_viability']:.2f}%",
                f"{data['live_error']:+d}",
                f"{data['dead_error']:+d}",
                f"{data['viability_error']:+.2f}%"
            ])
        
        columns = ['编号', '图像文件名', '实际活细胞', '预测活细胞', '实际死细胞', '预测死细胞',
                  '实际活力', '预测活力', '活细胞误差', '死细胞误差', '活力误差']
        
        table = ax7.table(cellText=table_data, colLabels=columns, cellLoc='center',
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置表头样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式（根据误差着色）
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
                else:
                    table[(i, j)].set_facecolor('white')
                # 误差列使用颜色编码
                if j == 8:  # 活细胞误差
                    error = comparison_data[i-1]['live_error']
                    if abs(error) <= 2:
                        table[(i, j)].set_facecolor('#90EE90')  # 浅绿色 - 误差小
                    elif abs(error) <= 5:
                        table[(i, j)].set_facecolor('#FFE4B5')  # 浅黄色 - 误差中等
                    else:
                        table[(i, j)].set_facecolor('#FFB6C1')  # 浅红色 - 误差大
                elif j == 9:  # 死细胞误差
                    error = comparison_data[i-1]['dead_error']
                    if abs(error) <= 2:
                        table[(i, j)].set_facecolor('#90EE90')
                    elif abs(error) <= 5:
                        table[(i, j)].set_facecolor('#FFE4B5')
                    else:
                        table[(i, j)].set_facecolor('#FFB6C1')
                elif j == 10:  # 活力误差
                    error = abs(comparison_data[i-1]['viability_error'])
                    if error <= 5:
                        table[(i, j)].set_facecolor('#90EE90')
                    elif error <= 10:
                        table[(i, j)].set_facecolor('#FFE4B5')
                    else:
                        table[(i, j)].set_facecolor('#FFB6C1')
        
        ax7.set_title(f'{model_name} - 每张图像的细胞数量和细胞活力详细对比', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.suptitle(f'{model_name} - 实际 vs 预测对比分析', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        self._save_figure(fig, f'{model_name}_cell_count_comparison')
        plt.close()
        
        # 保存为CSV文件
        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(self.save_dir, f'{model_name}_cell_count_comparison.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"细胞数量对比数据已保存到: {csv_path}")
