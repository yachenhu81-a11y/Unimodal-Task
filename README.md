# 单模态任务：草图识别与可控生成

> **上海交通大学 CS3308 机器学习课程期末项目**

本项目是一个集草图识别、生成与评估于一体的完整实验框架。在草图识别任务中，我们基于QuickDraw-414k数据集，设计了结合草图图像与序列表示的多模态融合卷积神经网络模型，在测试集上达到了 74.75%​ 的准确率。在草图生成任务中，我们实现了基于变分自编码器(VAE)的可控生成模型，支持重建和潜在空间插值，并提出了新的实例级评估指标。

开发人员：胡雅晨(https://github.com/yachenhu81-a11y) & 肖嘉熠(https://github.com/pigeon-rat) & 蒋铎鸣(https://github.com/www123wwwww)

## 核心特性
高效草图识别：采用多模态（图像+序列）融合的CNN模型，显著提升了复杂草图的分类性能。

可控草图生成：基于VAE实现草图重建与潜在空间平滑插值，生成过程可控可解释。

新颖评估体系：提出实例级评估指标，从多个维度量化生成草图的质量。

完整实验验证：识别任务准确率达74.75%，重建任务平均得分0.898，深入分析了模型的有效性与局限性。

## 项目结构概览
```text
Unimodal_Task/
│
├── data/
│   ├── QuickDraw414k/             # [需下载] 原始数据集 (.npy, .png)
│   └── generated_results/         # 模型生成的草图结果
│
├── checkpoints/                   # [需下载] 预训练模型权重 (.pth)
│   ├── classifier_npy_best.pth    # 序列单模态模型
│   ├── classifier_img_best.pth    # 图片单模态模型
│   └── classifier_dual_best.pth   # 双模态融合模型 (最终模型)
│
├── scripts_recognition/           # === 任务 A: 识别 ===
│   ├── train_npy.py               # 训练序列分支 (Transformer)
│   ├── train_img.py               # 训练图像分支 (ResNet)
│   ├── train_dual.py              # 训练双模态融合网络
│   └── eval_generation.py         # 使用识别模型评估生成质量
│
├── scripts_generation/            # === 任务 B: 生成 ===
│   ├── sketch_rnn.py              # 训练 VAE 生成模型
│   ├── reconstruct_samples.py     # 执行重建任务
│   ├── interpolation.py           # 执行插值任务
│   ├── insmetric.py               # 计算实例级评估指标
│   └── test_sketch/               # 生成模型权重存放处
│
├── requirements.txt               # 依赖列表
└── README.md                      # 项目说明

```
## 快速开始
### 1. 环境配置
克隆本仓库并安装所需依赖
```
git clone https://github.com/yachenhu81-a11y/Unimodal-Task.git
cd Unimodal-Task
pip install -r requirements.txt
```
### 2. 数据准备
将 QuickDraw-414k 数据集文件置于 data/QuickDraw414k/目录下。

### 3. 模型库
由于 GitHub 文件大小限制，预训练模型权重请通过以下链接下载，并将草图识别模型放入 `checkpoints/` 目录，草图生成模型放入`scripts_generation/test_sketch/` 目录

分享内容: classifier_npy_best.pth

链接: https://pan.sjtu.edu.cn/web/share/6d8407098dfdbe95ea5fe22498196d49, 提取码: 1h3w

## 使用指南

### 任务A ：草图识别

*1. 训练单模态基线**

```bash
# 训练序列模型 (Transformer)
python scripts_recognition/train_npy.py

# 训练图像模型 (ResNet-18)
python scripts_recognition/train_img.py
```

**2. 训练双模态融合模型**
依赖于单模态预训练权重，进行 Late Fusion 微调：
```bash
python scripts_recognition/train_dual.py
```

### 任务 B: 草图生成 (Generation)

**1. 训练生成模型**

```bash
python scripts_generation/sketch_rnn.py
```

**2. 草图重建 (Reconstruction)**

```bash
# 将测试集草图编码后重新解码
python scripts_generation/reconstruct_samples.py
```

**3. 潜在空间插值 (Interpolation)**
生成从一个草图渐变到另一个草图的序列：

```bash
python scripts_generation/interpolation.py
```

**4. 评估生成质量**

```bash
# 计算实例级几何指标 (Instance-level Metrics)
python scripts_generation/insmetric.py
```
