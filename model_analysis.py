"""
FVCD-net 模型分析与调试工具
用于分析训练过程、损失曲线、模型表现和过拟合问题
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_training_log(log_file):
    """
    加载训练日志数据
    可根据实际日志格式调整
    """
    try:
        # 尝试直接读取CSV格式日志
        log_df = pd.read_csv(log_file)
        return log_df
    except:
        # 如果不是标准CSV格式，尝试自定义解析
        epochs = []
        train_loss = []
        val_loss = []
        
        with open(log_file, 'r') as f:
            for line in f:
                # 根据日志格式解析，这里需要根据实际日志格式调整
                if 'Epoch' in line and 'Loss' in line:
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if 'Epoch' in part:
                            epoch_num = int(parts[i+1].strip(':,'))
                            epochs.append(epoch_num)
                        if 'Loss:' in part:
                            loss_value = float(parts[i+1])
                            train_loss.append(loss_value)
                        # 根据需要解析其他指标
        
        return pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss else None
        })


def plot_training_loss(log_df):
    """
    绘制训练过程中的损失曲线
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    if 'train_loss' in log_df.columns:
        plt.plot(log_df['epoch'], log_df['train_loss'], 'b-', label='训练损失')
    
    # 如果有验证损失则一并绘制
    if 'val_loss' in log_df.columns and not log_df['val_loss'].isna().all():
        plt.plot(log_df['epoch'], log_df['val_loss'], 'r--', label='验证损失')
    
    plt.title('训练过程损失曲线')
    plt.xlabel('训练轮次(Epoch)')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('training_loss.png', dpi=300)
    plt.show()
    
    # 如果有足够的数据，分析损失下降趋势
    if len(log_df) > 1 and 'train_loss' in log_df.columns:
        train_loss = log_df['train_loss'].values
        loss_decrease_rate = [(train_loss[i] - train_loss[i+1])/train_loss[i] 
                             if train_loss[i] != 0 else 0 
                             for i in range(len(train_loss)-1)]
        
        plt.figure(figsize=(12, 4))
        plt.plot(log_df['epoch'].values[1:], loss_decrease_rate)
        plt.title('每个Epoch的损失下降率')
        plt.xlabel('Epoch')
        plt.ylabel('损失下降率')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('loss_decrease_rate.png', dpi=300)
        plt.show()


def analyze_loss_components(component_losses):
    """
    分析各损失组件的相对贡献
    component_losses: 包含各损失组件均值的字典
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制柱状图
    plt.bar(component_losses.keys(), component_losses.values())
    plt.title('模型各损失组件贡献')
    plt.ylabel('平均损失值')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 在每个柱状图上标注百分比
    total = sum(component_losses.values())
    for i, (key, value) in enumerate(component_losses.items()):
        plt.text(i, value + 0.001*total, f'{value/total*100:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('loss_components.png', dpi=300)
    plt.show()
    
    return {k: {'value': v, 'percentage': v/total*100} for k, v in component_losses.items()}


def plot_loss_components_over_time(loss_components_df):
    """
    绘制各损失组件随时间的变化
    loss_components_df: 包含各epoch各损失组件的DataFrame
    """
    plt.figure(figsize=(14, 6))
    
    # 除了'epoch'列外的所有列被视为损失组件
    components = [col for col in loss_components_df.columns if col != 'epoch']
    
    for component in components:
        plt.plot(loss_components_df['epoch'], loss_components_df[component], label=component)
    
    plt.title('各损失组件随训练过程的变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('loss_components_over_time.png', dpi=300)
    plt.show()


def analyze_performance_by_group(results_df, group_by='image_type'):
    """
    按图像类型分析性能表现
    默认使用文件名第一部分作为分组依据
    """
    # 添加图像分类标签
    if group_by not in results_df.columns:
        results_df[group_by] = results_df['filename'].apply(
            lambda x: x.split('_')[0] if '_' in x else x.split('.')[0]
        )
    
    # 按图像类型分组分析性能
    group_performance = results_df.groupby(group_by).agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'ssim': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("不同类型图像的性能:")
    print(group_performance)
    
    # 可视化不同类型图像的性能
    plt.figure(figsize=(14, 6))
    sns.boxplot(x=group_by, y='psnr', data=results_df)
    plt.title(f'不同{group_by}的PSNR分布')
    plt.xlabel(group_by)
    plt.ylabel('PSNR (dB)')
    plt.xticks(rotation=45 if len(results_df[group_by].unique()) > 10 else 0)
    plt.savefig(f'performance_by_{group_by}_psnr.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(x=group_by, y='ssim', data=results_df)
    plt.title(f'不同{group_by}的SSIM分布')
    plt.xlabel(group_by)
    plt.ylabel('SSIM')
    plt.xticks(rotation=45 if len(results_df[group_by].unique()) > 10 else 0)
    plt.savefig(f'performance_by_{group_by}_ssim.png', dpi=300)
    plt.show()
    
    # 找出每种类型中表现最好和最差的样本
    best_worst_by_type = {}
    for img_type in results_df[group_by].unique():
        type_df = results_df[results_df[group_by] == img_type]
        best = type_df.nlargest(1, 'psnr')
        worst = type_df.nsmallest(1, 'psnr')
        
        print(f"\n类型 {img_type} 表现最好的样本: {best['filename'].values[0]}, "
              f"PSNR={best['psnr'].values[0]:.2f}, SSIM={best['ssim'].values[0]:.4f}")
        print(f"类型 {img_type} 表现最差的样本: {worst['filename'].values[0]}, "
              f"PSNR={worst['psnr'].values[0]:.2f}, SSIM={worst['ssim'].values[0]:.4f}")
        
        best_worst_by_type[img_type] = {'best': best, 'worst': worst}
    
    return group_performance, best_worst_by_type


def analyze_overfitting(train_metrics, val_metrics):
    """
    分析是否存在过拟合问题
    train_metrics, val_metrics: 包含训练和验证指标的DataFrame
    """
    # 确保两个DataFrame索引相同
    common_epochs = set(train_metrics['epoch']).intersection(set(val_metrics['epoch']))
    train_data = train_metrics[train_metrics['epoch'].isin(common_epochs)]
    val_data = val_metrics[val_metrics['epoch'].isin(common_epochs)]
    
    # 按epoch排序
    train_data = train_data.sort_values('epoch')
    val_data = val_data.sort_values('epoch')
    
    # 绘制训练集和验证集上的性能对比
    plt.figure(figsize=(16, 12))
    
    # 损失对比
    plt.subplot(2, 2, 1)
    plt.plot(train_data['epoch'], train_data['loss'], 'b-', label='训练集损失')
    plt.plot(val_data['epoch'], val_data['loss'], 'r--', label='验证集损失')
    plt.title('训练集与验证集损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # PSNR对比
    if 'psnr' in train_data.columns and 'psnr' in val_data.columns:
        plt.subplot(2, 2, 2)
        plt.plot(train_data['epoch'], train_data['psnr'], 'b-', label='训练集PSNR')
        plt.plot(val_data['epoch'], val_data['psnr'], 'r--', label='验证集PSNR')
        plt.title('训练集与验证集PSNR对比')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # SSIM对比
    if 'ssim' in train_data.columns and 'ssim' in val_data.columns:
        plt.subplot(2, 2, 3)
        plt.plot(train_data['epoch'], train_data['ssim'], 'b-', label='训练集SSIM')
        plt.plot(val_data['epoch'], val_data['ssim'], 'r--', label='验证集SSIM')
        plt.title('训练集与验证集SSIM对比')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # 损失差异
    plt.subplot(2, 2, 4)
    loss_diff = train_data['loss'] - val_data['loss']
    plt.plot(train_data['epoch'], loss_diff)
    plt.title('训练集与验证集损失差异')
    plt.xlabel('Epoch')
    plt.ylabel('Loss差异')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300)
    plt.show()
    
    # 分析过拟合风险
    last_epochs = min(10, len(loss_diff))
    recent_diff_mean = loss_diff.iloc[-last_epochs:].mean()
    diff_trend = loss_diff.iloc[-last_epochs:].is_monotonic_increasing
    
    overfitting_risk = "低"
    if recent_diff_mean > 0.1 * val_data['loss'].iloc[-1]:
        overfitting_risk = "中"
    if recent_diff_mean > 0.3 * val_data['loss'].iloc[-1] or diff_trend:
        overfitting_risk = "高"
    
    print(f"过拟合风险评估: {overfitting_risk}")
    print(f"最近{last_epochs}个epoch的训练-验证损失差异均值: {recent_diff_mean:.4f}")
    print(f"差异是否持续增长: {'是' if diff_trend else '否'}")
    
    return {
        'risk_level': overfitting_risk,
        'recent_diff_mean': recent_diff_mean,
        'is_diff_increasing': diff_trend
    }


if __name__ == "__main__":
    # 示例用法
    log_file = 'training_log.csv'  # 替换为实际的日志文件
    
    # 加载并分析训练日志
    log_df = load_training_log(log_file)
    plot_training_loss(log_df)
    
    # 分析损失组件（假设数据）
    component_losses = {
        'SR_loss': 0.002,
        'Recon_loss': 0.0015,
        'Edge_loss': 0.0005
    }
    analyze_loss_components(component_losses)