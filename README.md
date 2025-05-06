# F-VCD Network (Fourier Light Field Microscopy View Channel Depth)

[![GitHub license](https://img.shields.io/github/license/Placebo303/FVCDnet.svg)](https://github.com/Placebo303/FVCDnet/blob/main/LICENSE)

# F-VCD Network (Optimized by Kai Yang, based on Fei Lab's original F-VCD)

**F-VCD** is a deep learning-based approach for microscopy image restoration and enhancement in the Fourier domain. This repository contains the official implementation of the F-VCD network as described in our paper.

## 🧠 Overview

F-VCD-net is designed for **Fourier Light Field Microscopy (FLFM)** image restoration. By leveraging Fourier domain priors and a novel **View-Channel-Depth (VCD)** architecture, it achieves high-quality reconstructions of biological structures, enabling tasks such as:

- Super-resolution  
- Denoising  
- Deconvolution  
- Artifact removal  

## 🛠️ Environment Setup

You can use **conda** or **miniforge** to create the required environment.

### Option 1: Environment from `.yml` file

```bash
git clone https://github.com/Placebo303/FVCDnet.git
cd FVCDnet

conda env create -f f_vcd_environment.yml
conda activate FVCD-net
```

### Option 2: Manual installation

```bash
conda create -n FVCD-net python=3.7
conda activate FVCD-net

conda install -c anaconda -c conda-forge -c simpleitk \
  imageio=2.4.1 scipy=1.2.0 scikit-image=0.14.1 tensorflow-gpu=1.14.0 numpy=1.15.4

pip install easydict==1.9
```

### Dependencies

- Python 3.7  
- TensorFlow-GPU 1.14.0  
- NumPy 1.15.4  
- SciPy 1.2.0  
- scikit-image 0.14.1  
- imageio 2.4.1  
- easydict 1.9  

## 📈 Training

To train the F-VCD network, open and run the `FVCD-training.ipynb` notebook. It includes:

1. Environment initialization  
2. Data preparation  
3. Model training  
4. Performance evaluation  

## 🚀 Usage

F-VCD can be applied to a wide range of microscopy image restoration tasks. Detailed examples are provided in the notebook, including:

- Single-image super-resolution  
- Low-light denoising  
- Volumetric reconstruction  

## 📚 Citation

```bibtex
@article{feilab-fvcd,
  title={F-VCD: A Fourier Light Field Microscopy View Channel Depth Network for Microscopy Image Restoration},
  author={Fei Lab},
  journal={TBD},
  year={2023}
}
```

## 📄 License

This project is licensed under the [MIT License](https://github.com/Placebo303/FVCDnet/blob/main/LICENSE).

## 🙏 Acknowledgments

- Original implementation by: feilab-hust/F-VCD

## 👤 Author Contributions

This repository includes practical adaptations and minor improvements by **Kai Yang (Placebo303)**, based on the original implementation by Fei Lab.

The key modifications are:
- [✓] Adapted and debugged the training and inference pipeline for stable execution on the Paperspace cloud platform
- [✓] Improved data loading efficiency by converting `.tif` datasets to `.npy` format
- [✓] Restructured the inference process into two stages to avoid image parameter mismatch failures
- [✓] Fixed several compatibility issues related to environment setup and file paths

For academic reference to this modified version, please cite as:

```bibtex
@misc{yangkai-fvcd,
  author = {Kai Yang},
  title = {F-VCDnet: Practical Adaptation for Fourier Light Field Microscopy on Cloud Platforms},
  year = {2025},
  note = {Adapted version based on Fei Lab's F-VCD. GitHub: https://github.com/Placebo303/FVCDnet}
}

## 📬 Contact

For questions, bug reports, or contributions, feel free to open an issue or contact the repository owner at micheal_yangkai@foxmail.com

_Last updated: 2025-05-06 by Placebo303_


---

# F-VCD 网络（傅里叶光场显微成像视角频道深度网络）

[![GitHub license](https://img.shields.io/github/license/Placebo303/FVCDnet.svg)](https://github.com/Placebo303/FVCDnet/blob/main/LICENSE)

# F-VCD 网络（由Placebo303优化，基于 Fei Lab 的原始 F-VCD）

**F-VCD** 是一种基于深度学习的方法，用于在傅里叶域中对显微图像进行恢复与增强。本仓库包含了我们论文中所描述的 F-VCD 网络的官方实现。

## 🧠 概述

F-VCD 网络专为 **傅里叶光场显微成像（FLFM）** 图像恢复设计，结合傅里叶域先验和新颖的 **视角-通道-深度（VCD）** 网络结构，实现对生物图像的高质量重建，支持以下任务：

- 超分辨率重建  
- 去噪处理  
- 去卷积  
- 去伪影  

## 🛠️ 环境配置

您可以通过 **conda** 或 **miniforge** 创建运行环境。

### 方式一：使用 `.yml` 文件配置环境

```bash
git clone https://github.com/Placebo303/FVCDnet.git
cd FVCDnet

conda env create -f f_vcd_environment.yml
conda activate FVCD-net
```

### 方式二：手动安装依赖项

```bash
conda create -n FVCD-net python=3.7
conda activate FVCD-net

conda install -c anaconda -c conda-forge -c simpleitk \
  imageio=2.4.1 scipy=1.2.0 scikit-image=0.14.1 tensorflow-gpu=1.14.0 numpy=1.15.4

pip install easydict==1.9
```

### 依赖项列表

- Python 3.7  
- TensorFlow-GPU 1.14.0  
- NumPy 1.15.4  
- SciPy 1.2.0  
- scikit-image 0.14.1  
- imageio 2.4.1  
- easydict 1.9  

## 📈 网络训练

要训练 F-VCD 网络，请运行 `FVCD-training.ipynb` 笔记本，内容包括：

1. 环境初始化  
2. 数据准备  
3. 模型训练  
4. 性能评估  

## 🚀 使用说明

F-VCD 可应用于多种显微图像恢复任务，详细示例请参考笔记本，包括：

- 单幅图像超分辨率重建  
- 低照度图像去噪  
- 三维体重建  

## 📚 引用格式

```bibtex
@article{feilab-fvcd,
  title={F-VCD: A Fourier Light Field Microscopy View Channel Depth Network for Microscopy Image Restoration},
  author={Fei Lab},
  journal={TBD},
  year={2023}
}
```

## 📄 授权协议

本项目采用 [MIT 协议](https://github.com/Placebo303/FVCDnet/blob/main/LICENSE)。

## 🙏 致谢

- 原始实现：feilab-hust/F-VCD  

## 👤 作者贡献

本仓库在 Fei Lab 的原始 F-VCD 实现基础上，由 **Kai Yang（GitHub ID: Placebo303）** 进行了实际部署和部分适配优化。

主要修改包括：
- [✓] 将训练与推理流程适配到 Paperspace 云平台，修复运行过程中出现的兼容性问题  
- [✓] 为提高数据加载效率，将原始 `.tif` 图像数据预处理为 `.npy` 格式  
- [✓] 将推理流程拆分为两步执行，避免因图像参数不匹配导致中断  
- [✓] 修复部分环境配置与路径处理相关的问题，提升整体稳定性  

如需在学术中引用本修改版本，可使用以下格式：

```bibtex
@misc{yangkai-fvcd,
  author = {Kai Yang},
  title = {F-VCDnet: Practical Adaptation for Fourier Light Field Microscopy on Cloud Platforms},
  year = {2025},
  note = {Adapted version based on Fei Lab's F-VCD. GitHub: https://github.com/Placebo303/FVCDnet}
}

## 📬 联系方式

如有问题或建议，欢迎在 GitHub 上提交 Issue 或使用邮件地址：micheal_yangkai@foxmail.com联系仓库维护者

_最后更新于：2025 年 5 月 6 日，由 Placebo303_
