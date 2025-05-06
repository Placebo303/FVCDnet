"""
FVCD-net 模型评估与验证工具
用于评估模型性能、可视化重建结果和分析边缘保留效果
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义正确的解析函数
def load_validation_results(metrics_file):
    """
    从验证指标文件加载结果
    """
    results = []
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:  # 确保行有足够的数据
            filename = parts[0]
            # 正确提取PSNR和SSIM值
            psnr_part = [p for p in parts if 'PSNR:' in p]
            ssim_part = [p for p in parts if 'SSIM:' in p]
            
            if psnr_part and ssim_part:
                psnr = float(psnr_part[0].split(':')[1])
                ssim = float(ssim_part[0].split(':')[1])
                results.append({'filename': filename, 'psnr': psnr, 'ssim': ssim})
    
    df = pd.DataFrame(results)
    return df

def analyze_performance_statistics(df):
    """
    分析性能统计数据并生成图表
    """
    # 确保DataFrame有正确的列
    if 'psnr' not in df.columns or 'ssim' not in df.columns:
        print(f"警告: DataFrame中没有找到预期的列。可用列: {df.columns.tolist()}")
        return {}
        
    # 计算总体统计信息
    stats = {
        "总样本数": len(df),
        "平均PSNR": df['psnr'].mean(),
        "平均SSIM": df['ssim'].mean(),
        "PSNR范围": f"{df['psnr'].min():.2f} - {df['psnr'].max():.2f}",
        "SSIM范围": f"{df['ssim'].min():.4f} - {df['ssim'].max():.4f}",
        "PSNR标准差": df['psnr'].std(),
        "SSIM标准差": df['ssim'].std()
    }
    
    # 输出统计信息
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 创建性能分布直方图
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['psnr'], kde=True, bins=20)
    plt.title('PSNR分布')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('样本数量')
    plt.axvline(df['psnr'].mean(), color='red', linestyle='--', 
                label=f'平均值: {df["psnr"].mean():.2f}')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(df['ssim'], kde=True, bins=20)
    plt.title('SSIM分布')
    plt.xlabel('SSIM')
    plt.ylabel('样本数量')
    plt.axvline(df['ssim'].mean(), color='red', linestyle='--', 
                label=f'平均值: {df["ssim"].mean():.4f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('performance_distribution.png', dpi=300)
    plt.show()
    
    # 保存结果到Excel文件以便进一步分析
    df.to_excel('validation_results.xlsx', index=False)
    
    return stats

# 使用这些函数分析您的数据
metrics_file = '/notebooks/Code/DL_net/data_npy/Mito_View3_[LF160_SF160_WF480]/LR/SR_[Mito]_view3_L5c126b8_reallasttry/validation_metrics.txt'

# 1. 加载验证结果
results_df = load_validation_results(metrics_file)

# 检查是否成功加载数据
print(f"加载了 {len(results_df)} 行数据")
if len(results_df) > 0:
    print(f"数据列: {results_df.columns.tolist()}")
    print(results_df.head())  # 显示前5行确认数据格式正确

# 2. 分析性能统计
stats = analyze_performance_statistics(results_df)


def compare_with_baselines(df, baseline_results=None):
    """
    与基线方法进行比较
    """
    if baseline_results is None:
        # 默认的基线结果，需要替换为实际数据
        baseline_results = {
            'Bicubic插值': {'psnr': 32.1, 'ssim': 0.843},
            'SRCNN': {'psnr': 34.5, 'ssim': 0.865},
            # 添加更多基线...
        }
    
    # 将当前模型结果添加到对比中
    model_comparison = pd.DataFrame(baseline_results).T
    model_comparison.loc['FVCD-net'] = {
        'psnr': df['psnr'].mean(), 
        'ssim': df['ssim'].mean()
    }
    
    print("模型性能对比:")
    print(model_comparison)
    
    # 创建性能对比柱状图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=model_comparison.index, y=model_comparison['psnr'])
    plt.title('不同模型PSNR对比')
    plt.ylabel('PSNR (dB)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=model_comparison.index, y=model_comparison['ssim'])
    plt.title('不同模型SSIM对比')
    plt.ylabel('SSIM')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()
    
    return model_comparison


def visualize_sample(gt_img, pred_img, sample_id, psnr, ssim):
    """
    可视化单个样本的原始图像和重建结果
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gt_img, cmap='gray' if len(gt_img.shape) == 2 else None)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_img, cmap='gray' if len(pred_img.shape) == 2 else None)
    plt.title(f'重建结果 (PSNR: {psnr:.2f}, SSIM: {ssim:.4f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sample_{sample_id}_visualization.png', dpi=300)
    plt.show()


def visualize_sample_difference(gt_img, pred_img, sample_id, psnr, ssim):
    """
    可视化样本差异图
    """
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(gt_img, cmap='gray' if len(gt_img.shape) == 2 else None)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_img, cmap='gray' if len(pred_img.shape) == 2 else None)
    plt.title(f'重建结果 (PSNR: {psnr:.2f}, SSIM: {ssim:.4f})')
    plt.axis('off')
    
    # 计算差异图
    diff = np.abs(gt_img - pred_img)
    
    plt.subplot(1, 3, 3)
    # 设置颜色范围，确保小差异也可见
    plt.imshow(diff, cmap='hot', norm=Normalize(vmin=0, vmax=max(0.1, diff.max())))
    plt.title('差异图')
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sample_{sample_id}_difference.png', dpi=300)
    plt.show()


def analyze_edge_detail(gt_img, pred_img, sample_id):
    """
    分析边缘和细节重建效果
    """
    # 使用Canny边缘检测
    gt_edges = canny(gt_img if len(gt_img.shape) == 2 else np.mean(gt_img, axis=2), sigma=2)
    pred_edges = canny(pred_img if len(pred_img.shape) == 2 else np.mean(pred_img, axis=2), sigma=2)
    
    # 计算边缘一致性
    edge_consistency = np.sum(gt_edges & pred_edges) / np.sum(gt_edges | pred_edges) if np.sum(gt_edges | pred_edges) > 0 else 0
    
    # 可视化边缘检测结果
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(gt_img, cmap='gray' if len(gt_img.shape) == 2 else None)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(pred_img, cmap='gray' if len(pred_img.shape) == 2 else None)
    plt.title('重建结果')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(gt_edges, cmap='gray')
    plt.title('Ground Truth边缘')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(pred_edges, cmap='gray')
    plt.title('重建结果边缘')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'边缘一致性: {edge_consistency:.4f}', fontsize=14, y=1.05)
    plt.savefig(f'edge_analysis_{sample_id}.png', dpi=300)
    plt.show()
    
    return edge_consistency


def find_best_worst_samples(df, n=5):
    """
    找出性能最佳和最差的样本
    """
    best_samples = df.nlargest(n, 'psnr')
    worst_samples = df.nsmallest(n, 'psnr')
    median_idx = len(df) // 2
    median_samples = df.iloc[median_idx-n//2:median_idx+n//2+1]
    
    print("最佳性能样本:")
    print(best_samples)
    print("\n最差性能样本:")
    print(worst_samples)
    print("\n中等性能样本:")
    print(median_samples)
    
    return {
        'best': best_samples,
        'worst': worst_samples,
        'median': median_samples
    }


def load_sample_images(sample_names, data_dir):
    """
    加载样本图像（GT和预测结果）
    注意: 需要根据实际存储格式修改
    """
    images = {}
    for name in sample_names:
        # 根据实际路径和文件格式调整
        gt_path = os.path.join(data_dir, 'gt', name)
        pred_path = os.path.join(data_dir, 'pred', name)
        
        # 根据实际文件格式加载图像
        # 例如，如果是numpy数组:
        gt_img = np.load(gt_path)
        pred_img = np.load(pred_path)
        
        # 或者如果是图像文件:
        # from PIL import Image
        # gt_img = np.array(Image.open(gt_path))
        # pred_img = np.array(Image.open(pred_path))
        
        images[name] = {'gt': gt_img, 'pred': pred_img}
    
    return images


if __name__ == "__main__":
    # 示例用法
    metrics_file = 'validation_metrics.txt'
    results_df = load_validation_results(metrics_file)
    analyze_performance_statistics(results_df)