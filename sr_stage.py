#!/usr/bin/env python
# SR阶段处理脚本 - 分离式两阶段推理方案

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tifffile
import argparse
import cv2
from tensorlayer.layers import InputLayer

# 导入项目配置和模型
from config import config
from model.F_VCD import F_Denoise
from utils import normalize_percentile

def read_valid_npy_images(path):
    img_list = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    if not img_list:
        raise FileNotFoundError(f"❌ 没有找到任何 .npy 验证数据: {path}")
    imgs = []
    names = []
    for fn in img_list:
        data = np.load(os.path.join(path, fn)).astype(np.float32)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        imgs.append(normalize_percentile(data))  # 使用与训练时一致的normalize_percentile归一化方法
        names.append(fn)
    imgs = np.stack(imgs, axis=0)  # [N, H, W, V]
    print(f"✅ 加载 {len(img_list)} 张验证图, shape = {imgs.shape}")
    return imgs, names

def sr_inference(epoch=0, batch_size=1):
    """SR阶段：LF输入 → SR网络 → 保存SR输出"""
    epoch_tag = 'best' if epoch == 0 else f'epoch{epoch}'
    ckpt_dir = config.TRAIN.ckpt_dir
    valid_path = config.VALID.lf2d_path
    save_dir = config.VALID.saving_path
    os.makedirs(save_dir, exist_ok=True)
    sr_temp_dir = os.path.join(save_dir, 'sr_outputs')
    os.makedirs(sr_temp_dir, exist_ok=True)

    # 1) 读取验证数据
    valid_imgs, names = read_valid_npy_images(valid_path)
    H, W = valid_imgs.shape[1], valid_imgs.shape[2]
    SR_size = np.array([H, W]) * config.img_setting.sr_factor

    # 2) 构建SR网络
    tf.reset_default_graph()
    t_image = tf.placeholder(tf.float32, [batch_size, H, W, config.img_setting.Nnum], name='t_LFP')
    with tf.device(f"/gpu:{config.TRAIN.device}"):
        inp_layer = InputLayer(t_image, name='input_layer')
        SR_net = F_Denoise(
            inp_layer, output_size=SR_size,
            angRes=config.img_setting.Nnum,
            sr_factor=config.img_setting.sr_factor,
            reuse=False, name=config.net_setting.SR_model,
            channels_interp=config.channels_interp,
            normalize_mode=config.preprocess.normalize_mode
        )
    
    # 3) 查找和加载SR权重
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

    # 关键修复：确保调用find_ckpt函数并赋值给sr_ckpt
    sr_ckpt = find_ckpt(config.net_setting.SR_model)
    print("📂 SR 权重路径:", sr_ckpt)

    # 4) 会话与载权
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    # 修改：手动加载权重，不使用TensorLayer的辅助函数
    sr_npz = np.load(sr_ckpt, allow_pickle=True)
    matched_sr = 0
    for var in SR_net.all_params:
        key = var.name  # 包含":0"
        if key in sr_npz:
            sess.run(var.assign(sr_npz[key]))
            matched_sr += 1
        else:
            # 尝试替代名称格式
            alt_key = key.split(':')[0]  # 去掉":0"后缀
            if alt_key in sr_npz:
                sess.run(var.assign(sr_npz[alt_key]))
                matched_sr += 1
            else:
                print(f"⚠️ SR_net: 未在 npz 中找到变量 {key}")
    print(f"✅ SR_net 成功加载 {matched_sr}/{len(SR_net.all_params)} 个参数")

    # 检查NaN值
    has_nan = False
    for v in SR_net.all_params[:5]:  # 只检查前5个权重避免过多输出
        arr = sess.run(v)
        print(f"[SR PARAM] {v.name:30s} mean={arr.mean():.6f}, std={arr.std():.6f}")
        if np.isnan(arr).any():
            has_nan = True
            print(f"⚠️ 警告: {v.name} 中发现NaN值!")
    
    if has_nan:
        print("⚠️ 警告: 发现NaN权重! 模型可能无法正常工作")
    
    # 5) 批量推理和保存
    name_mapping_file = os.path.join(sr_temp_dir, 'name_mapping.txt')
    with open(name_mapping_file, 'w') as f:
        for idx in range(len(valid_imgs)):
            # 处理一批次数据
            batch = valid_imgs[idx:idx+batch_size]
            sr_out = sess.run(SR_net.outputs, feed_dict={t_image: batch})
            
            # 保存SR输出
            sr_filename = f'sr_output_{idx:04d}.tif'
            sr_save_path = os.path.join(sr_temp_dir, sr_filename)
            tifffile.imwrite(sr_save_path, sr_out[0])
            
            # 记录映射关系
            f.write(f"{names[idx]},{sr_filename}\n")
            
            # 显示进度
            print(f"\r处理SR {idx+1}/{len(valid_imgs)}", end='')
    
    print(f"\n✅ SR阶段完成! 输出保存在: {sr_temp_dir}")
    print(f"📝 映射文件: {name_mapping_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR阶段推理脚本')
    parser.add_argument('--ckpt', type=int, default=0, help='0表示best，否则表示epoch编号')
    parser.add_argument('--batch', type=int, default=1, help='推理时batch size')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.TRAIN.device)
    
    sr_inference(epoch=args.ckpt, batch_size=args.batch)