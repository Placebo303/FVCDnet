import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tifffile
from skimage.metrics import structural_similarity as ssim_func
from skimage.transform import resize
from config import config                                  # 配置文件路径 & 参数
from model.F_VCD import F_Denoise, F_Recon
from tensorlayer.layers import InputLayer
from utils import normalize_percentile
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.TRAIN.device)

def read_valid_npy_images(path):
    img_list = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    if not img_list:
        raise FileNotFoundError(f"❌ 没有找到任何 .npy 验证数据: {path}")
    imgs = []
    for fn in img_list:
        data = np.load(os.path.join(path, fn)).astype(np.float32)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        imgs.append(normalize_percentile(data))  # 使用与训练时一致的 normalize_percentile 归一化方法
    imgs = np.stack(imgs, axis=0)  # [N, H, W, V]
    print(f"✅ 加载 {len(img_list)} 张验证图, shape = {imgs.shape}")
    return imgs, img_list

def calculate_psnr(gt, pred, data_range=1.0):
    """计算单张 2D 切片的 PSNR"""
    mse = np.mean((gt - pred) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(data_range / np.sqrt(mse))

def calculate_ssim(gt, pred, data_range=1.0):
    """调用 skimage.metrics 计算 SSIM"""
    return ssim_func(gt, pred, data_range=data_range)

def infer(epoch=0, batch_size=1):
    """主流程：构图→载权重→批量推理→计算指标并保存"""
    epoch_tag = 'best' if epoch == 0 else f'epoch{epoch}'
    ckpt_dir   = config.TRAIN.ckpt_dir
    valid_path = config.VALID.lf2d_path
    save_dir   = config.VALID.saving_path
    os.makedirs(save_dir, exist_ok=True)

    # 1) 读取验证集
    valid_imgs, names = read_valid_npy_images(valid_path)
    H, W = valid_imgs.shape[1], valid_imgs.shape[2]
    SR_size    = np.array([H, W]) * config.img_setting.sr_factor
    Recon_size = SR_size * np.array(config.img_setting.ReScale_factor)

    # 2) 构建分离的网络图 - 避免嵌套作用域问题
    tf.reset_default_graph()
    
    # === 第一阶段：SR网络 ===
    sr_input = tf.placeholder(tf.float32, [batch_size, H, W, config.img_setting.Nnum], name='sr_input')
    with tf.device(f"/gpu:{config.TRAIN.device}"):
        # 注意：这里不再创建额外的作用域，直接使用F_Denoise内部的作用域
        sr_input_layer = InputLayer(sr_input, name='input_layer')
        sr_net = F_Denoise(
            sr_input_layer, output_size=SR_size,
            angRes=config.img_setting.Nnum,
            sr_factor=config.img_setting.sr_factor,
            reuse=False, name=config.net_setting.SR_model,
            channels_interp=config.channels_interp,
            normalize_mode=config.preprocess.normalize_mode
        )
        sr_output = sr_net.outputs  # 记录SR输出张量
    
    # === 第二阶段：Recon网络（完全独立） ===
    recon_input = tf.placeholder(tf.float32, [batch_size, H, W, config.img_setting.Nnum], name='recon_input')
    with tf.device(f"/gpu:{config.TRAIN.device}"):
        # 同样不添加额外作用域
        recon_input_layer = InputLayer(recon_input, name='recon_input_layer')
        recon_net = F_Recon(
            recon_input_layer,
            n_slices=config.img_setting.n_slices,
            output_size=Recon_size,
            is_train=False,
            reuse=False,
            name=config.net_setting.Recon_model,
            channels_interp=config.channels_interp,
            normalize_mode=config.preprocess.normalize_mode
        )
        recon_output = recon_net.outputs  # 记录Recon输出张量
    
    # 3) 查找权重文件
    # 这部分保持不变...
    
    # 4) 会话与载权
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    
    # 加载SR_net权重 - 修复变量名匹配逻辑
    sr_npz = np.load(sr_ckpt, allow_pickle=True)
    # 现在从all_params获取变量而不是用作用域来收集
    sr_vars = sr_net.all_params
    matched_sr = 0
    for var in sr_vars:
        # 从var.name中去除:0后缀
        key = var.name.split(':')[0]
        if key in sr_npz:
            sess.run(var.assign(sr_npz[key]))
            matched_sr += 1
        else:
            # 尝试使用备选名称格式
            alt_key = key.replace('F_Denoise/F_Denoise/', 'F_Denoise/')
            if alt_key in sr_npz:
                sess.run(var.assign(sr_npz[alt_key]))
                matched_sr += 1
            else:
                print(f"⚠️ SR_net: 未在npz中找到变量 {key} 或 {alt_key}")
    print(f"✅ SR_net 成功加载 {matched_sr}/{len(sr_vars)} 个参数")

    # 加载Recon_net权重 - 使用相同的修复逻辑
    recon_npz = np.load(recon_ckpt, allow_pickle=True)
    recon_vars = recon_net.all_params
    matched_recon = 0
    for var in recon_vars:
        key = var.name.split(':')[0]
        if key in recon_npz:
            sess.run(var.assign(recon_npz[key]))
            matched_recon += 1
        else:
            alt_key = key.replace('F_Recon/F_Recon/', 'F_Recon/')
            if alt_key in recon_npz:
                sess.run(var.assign(recon_npz[alt_key]))
                matched_recon += 1
            else:
                print(f"⚠️ Recon_net: 未在npz中找到变量 {key} 或 {alt_key}")
    print(f"✅ Recon_net 成功加载 {matched_recon}/{len(recon_vars)} 个参数")
    
    # 打印权重前5个参数来确认是否正确加载
    print("-------------- 调试 Recon_net 参数 --------------")
    for i, v in enumerate(recon_vars[:5]):
        arr = sess.run(v)
        print(f"[RECON PARAM {i}] {v.name:30s} mean={arr.mean():.6f}, std={arr.std():.6f}")

    # 5) 两阶段推理过程
    # 5.1) SR阶段 - 处理LF输入
    sr_result = sess.run(sr_output, feed_dict={sr_input: valid_imgs[0:1]})
    print(f"[SR STAGE] sr_result shape={sr_result.shape}, min={sr_result.min():.4f}, max={sr_result.max():.4f}")
    
    # 保存SR输出
    sr_save_path = os.path.join(save_dir, 'sr_out.tif')
    tifffile.imwrite(sr_save_path, sr_result[0])
    
    # 5.2) Recon阶段 - 将SR结果作为输入
    # 注：此处先resize回原始尺寸
    H0, W0 = valid_imgs.shape[1], valid_imgs.shape[2]
    sr_resized = cv2.resize(
        sr_result[0], (W0, H0), 
        interpolation=cv2.INTER_LINEAR
    )[None, ...]  # 增加批次维度
    
    # 随机测试输入
    random_input = np.random.uniform(0.2, 0.8, size=(1, H0, W0, config.img_setting.Nnum))
    random_result = sess.run(recon_output, feed_dict={recon_input: random_input})
    print(f"[RECON RANDOM] shape={random_result.shape}, min={random_result.min():.4f}, max={random_result.max():.4f}")
    
    # 真实SR输出作为输入
    recon_result = sess.run(recon_output, feed_dict={recon_input: sr_resized})
    print(f"[RECON STAGE] recon_result shape={recon_result.shape}, min={recon_result.min():.4f}, max={recon_result.max():.4f}")
    
    # 保存Recon输出
    recon_save_path = os.path.join(save_dir, 'recon_out.tif')
    tifffile.imwrite(recon_save_path, recon_result[0])
    print(f"📂 保存图像: SR={sr_save_path}, Recon={recon_save_path}")

    # 6) 批量推理、保存 tif、计算并记录 PSNR/SSIM
    log_path = os.path.join(save_dir, 'validation_metrics.txt')
    with open(log_path, 'w') as logf:
        for idx in range(len(valid_imgs)):
            batch = valid_imgs[idx:idx+batch_size]
            out = sess.run(Recon_net.outputs, feed_dict={t_image: batch})
            vol = np.squeeze(out, axis=0)
            vol = np.clip(vol, 0, 1)

            # 保存重建体数据
            save_p = os.path.join(save_dir, names[idx].replace('.npy', '.tif'))
            tifffile.imwrite(save_p, vol.astype(np.float32))

            # 读取 GT stack 并归一化
            gt_p = os.path.join(config.VALID.gt_path, names[idx].replace('.npy', '.tif'))
            if os.path.exists(gt_p):
                gt_vol = tifffile.imread(gt_p).astype(np.float32)
                gt_vol = gt_vol / np.max(gt_vol)
                # —— 新增：如果 gt_vol 是 (D,H,W)，就转成 (H,W,D)，否则保持不变 —— 
                if gt_vol.ndim == 3 and gt_vol.shape[0] == vol.shape[-1]:
                    gt_vol = np.transpose(gt_vol, (1,2,0))
                    print(f"[DEBUG] Transposed gt_vol to shape {gt_vol.shape}")

                psnrs, ssims = [], []
                for z in range(vol.shape[-1]):
                    gt_slice   = gt_vol[..., z]    # 现在一定是 (H, W)
                    pred_slice = vol[...,   z]     # (H, W)
                    # 跳过纯背景切片
                    if gt_slice.max() < 1e-6:
                        continue
                    psnrs.append(calculate_psnr( gt_slice, pred_slice, data_range=1.0))
                    ssims.append(calculate_ssim(gt_slice, pred_slice, data_range=1.0))
                avg_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
                avg_ssim = float(np.mean(ssims)) if ssims else float('nan')
            else:
                avg_psnr = avg_ssim = float('nan')

            logf.write(f"{names[idx]}  PSNR:{avg_psnr:.4f}  SSIM:{avg_ssim:.4f}\n")
            print(f"\r[{idx+1}/{len(valid_imgs)}] PSNR:{avg_psnr:.2f} dB, SSIM:{avg_ssim:.4f}", end='')

    print(f"\n✅ 推理完成，日志保存在：{log_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='F-VCD 推理与评估脚本')
    parser.add_argument('-c', '--ckpt',  type=int, default=0, help='0 表示 best，否则表示 epoch 编号')
    parser.add_argument('-b', '--batch', type=int, default=1, help='推理时 batch size')
    args = parser.parse_args()
    infer(epoch=args.ckpt, batch_size=args.batch)
