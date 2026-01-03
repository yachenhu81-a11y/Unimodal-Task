# 单模态任务：草图识别与可控生成

本项目围绕上海交通大学 CS3308：机器学习课程的期末项目，专注于单模态任务——草图识别与可控草图生成。在草图识别任务中，我们基于QuickDraw-414k数据集，设计了结合草图图像与序列表示的多模态融合卷积神经网络模型，在测试集上达到了 74.75%​ 的准确率。在草图生成任务中，我们实现了基于变分自编码器(VAE)的可控生成模型，支持重建和潜在空间插值，并提出了新的实例级评估指标。项目全部源代码已公开。

开发人员：胡雅晨(https://github.com/yachenhu81-a11y) & 肖嘉熠() & 蒋铎鸣()
GitHub仓库：https://github.com/yachenhu81-a11y/Unimodal-Task
核心特性
高效草图识别：采用多模态（图像+序列）融合的CNN模型，显著提升了复杂草图的分类性能。
可控草图生成：基于VAE实现草图重建与潜在空间平滑插值，生成过程可控可解释。
新颖评估体系：提出实例级评估指标，从多个维度量化生成草图的质量。
完整实验验证：识别任务准确率达74.75%，重建任务平均得分0.898，深入分析了模型的有效性与局限性。

## 项目结构概览
```
Unimodal_Task/
│
├── data/
│   ├── QuickDraw414k/               # 原始数据集 (.npy, .png)
│   └── generated_results/           # 生成的草图
│
├── checkpoints/                     # 预训练模型 (.pth文件)
│   ├── classifier_npy_best.pth      # 序列单模态模型
│   ├── classifier_img_best.pth      # 图片单模态模型
│   ├── classifier_dual_best.pth     # 双模态融合模型
│   └── ...                         # 生成模型
│
├── logs/                            # 训练日志
│
├── scripts_recognition/             # A. 识别任务脚本
│   ├── utils.py
│   ├── train_npy.py                 # 训练序列单模态
│   ├── train_img.py                 # 训练图片单模态
│   ├── train_dual.py                # 训练双模态
│   └── eval_generation.py
│
├── scripts_generation/              # B. 生成任务脚本
│   ├── interpolation/               # 插值生成结果
│   ├── reconstruction/              # 重建生成结果
│   ├── insmetric.py                 # 实例级评估标准
│   ├── inspect_samples.py           # 检查原始图片
│   ├── interpolation.py             # 执行插值生成
│   ├── reconstruct_samples.py        # 执行图片重建
│   ├── sketch_rnn.py                # 训练生成模型
│   └── test_sketch/                 # 生成模型检查点
│       ├── decoder_epoch_80000.pth
│       └── encoder_epoch_80000.pth
│
├── README.md                        # 项目说明文档
└── requirements.txt                 # 项目依赖列表
```
快速开始
1. 环境配置
克隆本仓库并安装所需依赖。
git clone https://github.com/yachenhu81-a11y/Unimodal-Task.git
cd Unimodal-Task
pip install -r requirements.txt
2. 数据准备
将 QuickDraw-414k 数据集文件置于 data/QuickDraw414k/目录下。
3. 运行识别任务
训练或评估草图识别模型。
## 训练识别模型
先运行 train_npy.py 或 train_img.py 训练单模态模型。
```
python scripts_recognition/train_npy.py
```
```
python scripts_recognition/train_img.py
```
得到模型classifier_npy_best.pth 单模态图片模型，classifier_img_best.pth单模态序列模型后可运行 train_dual.py 训练双模态模型。
```
python scripts_recognition/train_dual.py
```
## 评估准确率
python scripts_recognition/eval_generation.py          
4. 运行生成任务
进行草图重建或插值生成。
## 训练生成模型
python scripts_generation/sketch_rnn.py
## 草图重建示例
python scripts_generation/reconstruct_samples.py
## 潜在空间插值生成
python scripts_generation/interpolation.py
5. 评估生成质量
使用我们提出的实例级指标评估生成结果。
python scripts_generation/insmetric.py
