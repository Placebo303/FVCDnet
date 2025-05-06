import os
import numpy as np
import imageio
from utils import get_and_rearrange3d, normalize_percentile
from config import config

# 设置原始数据目录（你的Stack、HR、LR文件夹）
original_root = os.path.join(os.path.dirname(__file__), 'data', 'Mito_View3_[LF160_SF160_WF480]')
# 设置保存npy的新目录
npy_root = os.path.join(os.path.dirname(__file__), 'data_npy', 'Mito_View3_[LF160_SF160_WF480]')
os.makedirs(os.path.join(npy_root, 'Stack'), exist_ok=True)
os.makedirs(os.path.join(npy_root, 'HR'), exist_ok=True)
os.makedirs(os.path.join(npy_root, 'LR'), exist_ok=True)

print("✅ 目录创建完毕")

# 定义处理函数
def convert_folder(sub_folder, save_sub_folder, read_type=None):
    src_path = os.path.join(original_root, sub_folder)
    save_path = os.path.join(npy_root, save_sub_folder)

    file_list = sorted([f for f in os.listdir(src_path) if f.endswith('.tif') or f.endswith('.tiff')])

    for idx, fname in enumerate(file_list):
        print(f"[{sub_folder}] Processing {idx+1}/{len(file_list)}: {fname}")
        data = get_and_rearrange3d(fname, src_path, normalize_fn=normalize_percentile,
                                   read_type=read_type, angRes=config.img_setting.Nnum)
        save_name = fname.replace('.tif', '.npy').replace('.tiff', '.npy')
        np.save(os.path.join(save_path, save_name), data.astype(np.float32))

# 开始批量处理
convert_folder('Stack', 'Stack')
convert_folder('HR', 'HR', read_type=config.preprocess.SynView_type)
convert_folder('LR', 'LR', read_type=config.preprocess.LFP_type)

print("🎯 全部数据转换完成，保存在:", npy_root)