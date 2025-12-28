# Unimodal-Task
## 文件结构
```
Unimodal_Task/
│
├── data/
│   ├── QuickDraw414k/               # A. 原始数据集 (.npy, .png)
│   └── generated_results/           # B. 生成的草图
│
├── checkpoints/                     # 模型仓库(存放训练好的 .pth)
│   ├── classifier_npy_best.pth      # A. 单模态图片模型
│   ├── classifier_img_best.pth      # A. 单模态序列模型
│   ├── classifier_dual_best.pth     # A. 双模态模型
│   └──                              # B. 生成模型
│
├── logs/                            # 【日志中心】存放训练日志 txt
│   ├── train_seq_{timestamp}.txt
│   └── train_dual_{timestamp}.txt
│
│
├── scripts_recognition/             # A. 识别任务脚本
│   ├── utils.py                     # 通用工具
│   ├── train_npy.py                 # 训练序列单模态
│   ├── train_img.py                 # 训练图片单模态
│   ├── train_dual.py                # 训练双模态
│   ├── eval_generation.py           # 评估生成质量
│
├── scripts_generation/              # B. 生成任务脚本
│
├── README.md                        # 项目说明书
└── requirements.txt                 # 依赖库列表
```
