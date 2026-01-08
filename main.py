import torch
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_eval import train_and_evaluate
import numpy as np

# 配置国内镜像源和AutoDL学术加速（适配autodl等云平台）
try:
    from setup_environment import setup_mirrors
    setup_mirrors()
except ImportError:
    # 如果没有setup_environment.py，使用基础配置
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '600')  # 10分钟超时
    
    # 尝试启用AutoDL加速
    if os.path.exists('/etc/network_turbo'):
        try:
            import subprocess
            result = subprocess.run(
                'bash -c "source /etc/network_turbo && env | grep proxy"',
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.splitlines():
                    if '=' in line:
                        var, value = line.split('=', 1)
                        os.environ[var] = value
                print("已启用AutoDL学术资源加速服务")
        except:
            pass

# 设置中文字体 - 尝试多种中文字体
import matplotlib.font_manager as fm
import warnings

# 查找系统中可用的中文字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 
                 'STSong', 'STHeiti', 'STKaiti', 'STFangsong']
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 找到第一个可用的中文字体
font_found = False
for font in chinese_fonts:
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font]
        font_found = True
        print(f"使用中文字体: {font}")
        break

# 如果没有找到中文字体，使用默认设置
if not font_found:
    # Linux云平台通常没有中文字体，使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    print("提示: 未找到中文字体，使用系统默认字体")

plt.rcParams['axes.unicode_minus'] = False
# 禁用matplotlib的字体警告（如果字体不可用，不影响功能）
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', message='.*missing from font.*')

# 设置matplotlib日志级别，抑制字体警告
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def main():
    """主函数"""
    # 配置
    data_dir = 'data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 清理GPU缓存
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU内存已清理")
    
    import argparse
    parser = argparse.ArgumentParser(description='细胞活力检测系统')
    parser.add_argument('--mode', type=str, default='train_eval', 
                       choices=['train', 'eval', 'train_eval', 'visualize'],
                       help='运行模式: train(仅训练), eval(仅评估), train_eval(训练+评估), visualize(仅可视化)')
    parser.add_argument('--regenerate-predictions', action='store_true',
                       help='可视化模式下，是否重新生成预测结果（需要模型和数据）')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['segnet', 'unet', 'enhanced_unet', 'fcn', 'pspnet', 'linknet'],
                       help='要处理的模型列表')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    args = parser.parse_args()
    
    num_epochs = args.epochs
    models = args.models
    mode = args.mode
    
    print(f"运行模式: {mode}")
    print(f"使用设备: {device}")
    print(f"数据目录: {data_dir}")
    print(f"模型列表: {models}")
    print(f"训练轮数: {num_epochs}\n")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练和评估所有模型
    all_results = {}
    
    for model_name in models:
        try:
            print(f"\n{'='*80}")
            print(f"开始处理模型: {model_name}")
            print(f"{'='*80}\n")
            
            # 可视化模式
            if mode == 'visualize':
                from train_eval import visualize_model
                visualize_model(
                    model_name=model_name,
                    data_dir=data_dir,
                    device=device,
                    checkpoint_path=None,
                    regenerate_predictions=args.regenerate_predictions
                )
                # 可视化模式不返回结果
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
            else:
                # 在其他模式处理前彻底清理GPU缓存
                if device == 'cuda':
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # 尝试重置CUDA状态
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except:
                            pass
                    except RuntimeError as e:
                        if 'CUDA' in str(e) or 'cuda' in str(e).lower():
                            print(f"警告: CUDA清理失败: {e}")
                            print("尝试重置CUDA设备...")
                            try:
                                # 尝试重置CUDA
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                            except:
                                print("无法重置CUDA，继续使用CPU")
                                device = 'cpu'
                        else:
                            raise
                
                from train_eval import train_model, evaluate_model
                
                checkpoint_path = None
                # 训练模式
                if mode in ['train', 'train_eval']:
                    checkpoint_path = train_model(
                        model_name=model_name,
                        data_dir=data_dir,
                        device=device,
                        num_epochs=num_epochs,
                        skip_training=False
                    )
                
                # 评估模式
                if mode in ['eval', 'train_eval']:
                    results = evaluate_model(
                        model_name=model_name,
                        data_dir=data_dir,
                        device=device,
                        checkpoint_path=checkpoint_path
                    )
                else:
                    # 仅训练模式，返回空结果
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
            
            all_results[model_name] = results
            
            print(f"\n{model_name} 评估结果:")
            print(f"  语义分割 mIoU: {results['sem_mean_iou']:.4f}")
            print(f"  语义分割 mDice: {results['sem_mean_dice']:.4f}")
            print(f"  语义分割-活细胞 IoU: {results.get('sem_live_iou', 0.0):.4f}")
            print(f"  语义分割-死细胞 IoU: {results.get('sem_dead_iou', 0.0):.4f}")
            print(f"  实例分割-活细胞 IoU: {results['live_iou']:.4f}")
            print(f"  实例分割-死细胞 IoU: {results['dead_iou']:.4f}")
            print(f"  活细胞检测准确率 (Precision): {results['live_precision']:.4f}")
            print(f"  死细胞检测准确率 (Precision): {results['dead_precision']:.4f}")
            print(f"  活细胞召回率 (Recall): {results.get('live_recall', 0.0):.4f}")
            print(f"  死细胞召回率 (Recall): {results.get('dead_recall', 0.0):.4f}")
            print(f"  细胞活力准确率: {results['viability_accuracy']:.4f}")
            print(f"  bbox mAP: {results['bbox_mAP']:.4f}")
            print(f"  segm mAP: {results['segm_mAP']:.4f}")
            
        except Exception as e:
            print(f"模型 {model_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {
                'sem_mean_iou': 0.0,
                'sem_mean_dice': 0.0,
                'live_iou': 0.0,
                'dead_iou': 0.0,
                'live_precision': 0.0,
                'dead_precision': 0.0,
                'viability_accuracy': 0.0,
                'bbox_mAP': 0.0,
                'segm_mAP': 0.0
            }
    
    # 保存结果
    with open('results/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 创建结果DataFrame（包含语义分割和实例分割指标）
    df_data = []
    for model_name, results in all_results.items():
        df_data.append({
            '模型': model_name,
            '语义分割 mIoU': results['sem_mean_iou'],
            '语义分割 mDice': results['sem_mean_dice'],
            '语义分割-背景 IoU': results.get('sem_background_iou', 0.0),  # 背景IoU
            '语义分割-背景 Dice': results.get('sem_background_dice', 0.0),  # 背景Dice
            '语义分割-活细胞 IoU': results.get('sem_live_iou', 0.0),  # 语义分割层面的活细胞IoU
            '语义分割-死细胞 IoU': results.get('sem_dead_iou', 0.0),  # 语义分割层面的死细胞IoU
            '语义分割-活细胞 Dice': results.get('sem_live_dice', 0.0),  # 语义分割层面的活细胞Dice
            '语义分割-死细胞 Dice': results.get('sem_dead_dice', 0.0),  # 语义分割层面的死细胞Dice
            '实例分割-活细胞 IoU': results['live_iou'],  # 实例分割层面的活细胞IoU
            '实例分割-死细胞 IoU': results['dead_iou'],  # 实例分割层面的死细胞IoU
            '活细胞检测准确率 (Precision)': results['live_precision'],
            '死细胞检测准确率 (Precision)': results['dead_precision'],
            '活细胞召回率 (Recall)': results.get('live_recall', 0.0),
            '死细胞召回率 (Recall)': results.get('dead_recall', 0.0),
            '细胞活力准确率': results['viability_accuracy'],
            'bbox mAP': results['bbox_mAP'],
            'segm mAP': results['segm_mAP']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv('results/evaluation_results.csv', index=False, encoding='utf-8-sig')
    
    # 可视化结果（使用Visualizer类）
    from visualization import Visualizer
    
    visualizer = Visualizer(save_dir='results')
    
    # 生成模型对比可视化
    print("\n生成模型对比可视化...")
    try:
        # 转换结果格式以适配可视化函数（3个类别：background, live, dead）
        viz_results = {}
        for model_name, results in all_results.items():
            # 获取background IoU（如果存在），否则设为0
            bg_iou = results.get('sem_background_iou', 0.0)
            viz_results[model_name] = {
                'iou': [
                    bg_iou,  # background (0)
                    results.get('sem_live_iou', 0.0),  # live (1)
                    results.get('sem_dead_iou', 0.0)   # dead (2)
                ],
                'dice': [
                    results.get('sem_background_dice', 0.0),  # background (0)
                    results.get('sem_live_dice', 0.0),  # live (1)
                    results.get('sem_dead_dice', 0.0)   # dead (2)
                ],
                'accuracy': results.get('sem_mean_iou', 0.0),  # 使用mIoU作为accuracy的近似
                'live_cell_acc': results.get('live_precision', 0.0),
                'dead_cell_acc': results.get('dead_precision', 0.0),
                'viability_acc': results.get('viability_accuracy', 0.0)
            }
        
        # 生成全面的模型对比可视化（从evaluation_results.csv读取数据）
        visualizer.plot_comprehensive_comparison_from_csv()
        
    except Exception as e:
        print(f"警告: 模型对比可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 原有的简单可视化
    visualize_results(all_results)
    
    print("\n" + "="*80)
    print("所有模型评估完成！")
    print("结果已保存到 results/ 目录")
    print("="*80)


def visualize_results(all_results: dict):
    """可视化评估结果"""
    
    # 1. 模型对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
    
    models = list(all_results.keys())
    
    # 子图1: IoU指标
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    live_ious = [all_results[m]['live_iou'] for m in models]
    dead_ious = [all_results[m]['dead_iou'] for m in models]
    ax1.bar(x - width/2, live_ious, width, label='活细胞 IoU', alpha=0.8)
    ax1.bar(x + width/2, dead_ious, width, label='死细胞 IoU', alpha=0.8)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('IoU')
    ax1.set_title('活细胞 vs 死细胞 IoU')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 检测准确率
    ax2 = axes[0, 1]
    live_precisions = [all_results[m]['live_precision'] for m in models]
    dead_precisions = [all_results[m]['dead_precision'] for m in models]
    ax2.bar(x - width/2, live_precisions, width, label='活细胞检测准确率', alpha=0.8)
    ax2.bar(x + width/2, dead_precisions, width, label='死细胞检测准确率', alpha=0.8)
    ax2.set_xlabel('模型')
    ax2.set_ylabel('准确率')
    ax2.set_title('活细胞 vs 死细胞检测准确率')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: mAP指标
    ax3 = axes[1, 0]
    bbox_maps = [all_results[m]['bbox_mAP'] for m in models]
    segm_maps = [all_results[m]['segm_mAP'] for m in models]
    ax3.bar(x - width/2, bbox_maps, width, label='bbox mAP', alpha=0.8)
    ax3.bar(x + width/2, segm_maps, width, label='segm mAP', alpha=0.8)
    ax3.set_xlabel('模型')
    ax3.set_ylabel('mAP')
    ax3.set_title('bbox mAP vs segm mAP')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 细胞活力准确率
    ax4 = axes[1, 1]
    viability_accs = [all_results[m]['viability_accuracy'] for m in models]
    bars = ax4.bar(x, viability_accs, alpha=0.8, color='green')
    ax4.set_xlabel('模型')
    ax4.set_ylabel('准确率')
    ax4.set_title('细胞活力准确率')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 详细对比表格
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['模型', '语义分割\nmIoU', '语义分割\nmDice', '活细胞\nIoU', '死细胞\nIoU',
               '活细胞检测\n准确率', '死细胞检测\n准确率', '细胞活力\n准确率', 'bbox\nmAP', 'segm\nmAP']
    
    for model_name in models:
        r = all_results[model_name]
        table_data.append([
            model_name,
            f"{r['sem_mean_iou']:.4f}",
            f"{r['sem_mean_dice']:.4f}",
            f"{r['live_iou']:.4f}",
            f"{r['dead_iou']:.4f}",
            f"{r['live_precision']:.4f}",
            f"{r['dead_precision']:.4f}",
            f"{r['viability_accuracy']:.4f}",
            f"{r['bbox_mAP']:.4f}",
            f"{r['segm_mAP']:.4f}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('模型详细性能对比表', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('results/detailed_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n可视化结果已保存到 results/ 目录")


if __name__ == '__main__':
    main()

