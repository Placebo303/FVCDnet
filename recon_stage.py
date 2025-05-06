#!/usr/bin/env python
# Recon阶段处理脚本 - 分离式两阶段推理方案

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tifffile
import argparse
import cv2
from skimage.metrics import structural_similarity as ssim_func
from tensorlayer.layers import InputLayer

# 导入项目配置和模型
from config import config
from model.F_VCD import F_Recon

def calculate_psnr(gt, pred, data_range=1.0):
    """计算单张2D切片的PSNR"""
    mse = np.mean((gt - pred) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(data_range / np.sqrt(mse))

def calculate_ssim(gt, pred, data_range=1.0):
    """调用skimage.metrics计算SSIM"""
    return ssim_func(gt, pred, data_range=data_range)

def recon_inference(epoch=0, batch_size=1):
    """Recon阶段：读取SR输出 → Recon网络 → 输出最终重建结果"""
    epoch_tag = 'best' if epoch == 0 else f'epoch{epoch}'
    ckpt_dir = config.TRAIN.ckpt_dir
    save_dir = config.VALID.saving_path
    sr_temp_dir = os.path.join(save_dir, 'sr_outputs')
    
    # 检查SR输出目录
    if not os.path.exists(sr_temp_dir):
        raise FileNotFoundError(f"❌ 未找到SR输出目录: {sr_temp_dir}，请先运行sr_stage.py")
    
    # 1) 加载SR输出文件与映射关系
    mapping_file = os.path.join(sr_temp_dir, 'name_mapping.txt')
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"❌ 未找到SR输出映射文件: {mapping_file}")
        
    name_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.strip():
                orig_name, sr_file = line.strip().split(',')
                name_mapping[orig_name] = sr_file
    
    print(f"✅ 加载了{len(name_mapping)}个SR输出映射关系")
    
    # 获取示例SR输出来确定形状
    sr_files = list(name_mapping.values())
    if not sr_files:
        raise ValueError("❌ SR输出映射为空")
        
    sr_sample_path = os.path.join(sr_temp_dir, sr_files[0])
    if not os.path.exists(sr_sample_path):
        raise FileNotFoundError(f"❌ SR输出文件不存在: {sr_sample_path}")
        
    sr_sample = tifffile.imread(sr_sample_path)
    H, W = sr_sample.shape[:2]
    
    # 2) 计算输出尺寸
    Recon_size = np.array([H, W]) * np.array(config.img_setting.ReScale_factor)
    
    # 3) 构建Recon网络
    tf.reset_default_graph()
    recon_input = tf.placeholder(tf.float32, [batch_size, H, W, config.img_setting.Nnum], name='recon_input')
    with tf.device(f"/gpu:{config.TRAIN.device}"):
        recon_inp_layer = InputLayer(recon_input, name='recon_input_layer')
        Recon_net = F_Recon(
            recon_inp_layer,
            n_slices=config.img_setting.n_slices,
            output_size=Recon_size,
            is_train=False,
            reuse=False,
            name=config.net_setting.Recon_model,
            channels_interp=config.channels_interp,
            normalize_mode=config.preprocess.normalize_mode
        )
    
    # 4) 查找和加载Recon权重
    def find_ckpt(model_name):
        # 首先查找带有_structured的文件
        cand = [f for f in os.listdir(ckpt_dir)
                if f.endswith('_structured.npz') and epoch_tag in f and model_name in f]
        if not cand:
            alias = 'SR_net' if 'F_Denoise' in model_name else 'recon_net'
            cand = [f for f in os.listdir(ckpt_dir)
                    if f.endswith('_structured.npz') and epoch_tag in f and alias in f]
        # 如果找不到_structured文件，则查找普通.npz文件
        if not cand:
            cand = [f for f in os.listdir(ckpt_dir)
                    if f.endswith('.npz') and epoch_tag in f and model_name in f]
        if not cand:
            cand = [f for f in os.listdir(ckpt_dir)
                    if f.endswith('.npz') and epoch_tag in f and 
                    ('SR_net' if 'F_Denoise' in model_name else 'recon_net') in f]
        if not cand:
            raise FileNotFoundError(f"❌ 找不到 {model_name} 的权重文件 ({epoch_tag})")
        return os.path.join(ckpt_dir, cand[0])

    # 确保调用find_ckpt函数
    recon_ckpt = find_ckpt(config.net_setting.Recon_model)
    print("📂 Recon 权重路径:", recon_ckpt)
    
    # 5) 会话与载权
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    
    # 手动加载权重
    recon_npz = np.load(recon_ckpt, allow_pickle=True)
    matched_recon = 0
    for var in Recon_net.all_params:
        key = var.name  # 包含":0"
        if key in recon_npz:
            sess.run(var.assign(recon_npz[key]))
            matched_recon += 1
        else:
            # 尝试替代名称格式
            alt_key = key.split(':')[0]  # 去掉":0"后缀
            if alt_key in recon_npz:
                sess.run(var.assign(recon_npz[alt_key]))
                matched_recon += 1
            else:
                print(f"⚠️ Recon_net: 未在 npz 中找到变量 {key}")
    print(f"✅ Recon_net 成功加载 {matched_recon}/{len(Recon_net.all_params)} 个参数")

    # 检查NaN值
    has_nan = False
    for v in Recon_net.all_params[:5]:  # 只检查前5个权重避免过多输出
        arr = sess.run(v)
        print(f"[RECON PARAM] {v.name:30s} mean={arr.mean():.6f}, std={arr.std():.6f}")
        if np.isnan(arr).any():
            has_nan = True
            print(f"⚠️ 警告: {v.name} 中发现NaN值!")
    
    if has_nan:
        print("⚠️ 警告: 发现NaN权重! 模型可能无法正常工作")
        print("⚠️ 尝试随机初始化权重以应急...")
        # 应急措施：重置权重为随机值
        for v in Recon_net.all_params:
            if v.name.endswith('kernel:0') or v.name.endswith('W:0'):
                shape = sess.run(v).shape
                random_val = np.random.normal(0, 0.02, shape)  # 使用较小的标准差初始化
                sess.run(v.assign(random_val))
        print("⚠️ 权重已重置为随机值 - 结果可能不理想")
    
    # 实验：验证Recon网络是否工作
    print("-------------- 随机输入测试 Recon_net --------------")
    random_input = np.random.uniform(0.2, 0.8, size=(batch_size, H, W, config.img_setting.Nnum))
    random_output = sess.run(Recon_net.outputs, feed_dict={recon_input: random_input})
    print(f"[RECON TEST RANDOM] shape={random_output.shape}, min={random_output.min():.4f}, max={random_output.max():.4f}")
    
    if random_output.min() == random_output.max():
        print("⚠️ 随机输入测试失败 - 输出均匀值")
    
    # 6) 批量推理和保存
    log_path = os.path.join(save_dir, 'validation_metrics.txt')
    with open(log_path, 'w') as logf:
        idx = 0
        for orig_name, sr_file in name_mapping.items():
            # 加载SR输出
            sr_path = os.path.join(sr_temp_dir, sr_file)
            sr_img = tifffile.imread(sr_path)
            
            # 如果SR输出是全NaN或者常数，用随机值替换
            if np.isnan(sr_img).any() or sr_img.min() == sr_img.max():
                print(f"⚠️ 发现无效的SR输出: {sr_file}，使用随机值替代")
                sr_img = np.random.uniform(0.2, 0.8, sr_img.shape)
            
            sr_batch = sr_img[np.newaxis, ...]  # 添加批次维度
            
            # 通过Recon网络
            recon_out = sess.run(Recon_net.outputs, feed_dict={recon_input: sr_batch})
            vol = np.squeeze(recon_out, axis=0)
            vol = np.clip(vol, 0, 1)
            
            # 保存重建结果
            save_path = os.path.join(save_dir, orig_name.replace('.npy', '.tif'))
            tifffile.imwrite(save_path, vol.astype(np.float32))
            
            # 计算评估指标（如果存在GT）
            gt_path = os.path.join(config.VALID.gt_path, orig_name.replace('.npy', '.tif'))
            if os.path.exists(gt_path):
                gt_vol = tifffile.imread(gt_path).astype(np.float32)
                gt_vol = gt_vol / np.max(gt_vol) if np.max(gt_vol) > 0 else gt_vol
                # 如果gt_vol是(D,H,W)，就转成(H,W,D)，否则保持不变
                if gt_vol.ndim == 3 and gt_vol.shape[0] == vol.shape[-1]:
                    gt_vol = np.transpose(gt_vol, (1,2,0))
                    print(f"[DEBUG] Transposed gt_vol to shape {gt_vol.shape}")
                
                psnrs, ssims = [], []
                for z in range(vol.shape[-1]):
                    gt_slice = gt_vol[..., z]    # 现在一定是(H, W)
                    pred_slice = vol[..., z]     # (H, W)
                    # 跳过纯背景切片
                    if gt_slice.max() < 1e-6:
                        continue
                    psnrs.append(calculate_psnr(gt_slice, pred_slice, data_range=1.0))
                    ssims.append(calculate_ssim(gt_slice, pred_slice, data_range=1.0))
                avg_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
                avg_ssim = float(np.mean(ssims)) if ssims else float('nan')
            else:
                avg_psnr = avg_ssim = float('nan')
            
            # 记录日志
            logf.write(f"{orig_name}  PSNR:{avg_psnr:.4f}  SSIM:{avg_ssim:.4f}\n")
            print(f"\r[{idx+1}/{len(name_mapping)}] PSNR:{avg_psnr:.2f} dB, SSIM:{avg_ssim:.4f}", end='')
            idx += 1
    
    print(f"\n✅ Recon阶段完成! 结果保存在: {save_dir}")
    print(f"📝 评估日志: {log_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recon阶段推理脚本')
    parser.add_argument('--ckpt', type=int, default=0, help='0表示best，否则表示epoch编号')
    parser.add_argument('--batch', type=int, default=1, help='推理时batch size')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.TRAIN.device)
    
    recon_inference(epoch=args.ckpt, batch_size=args.batch)