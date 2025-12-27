# Unimodal-Task
## 文件结构
```
Unimodal_Task/
│
├── data/
│   ├── QuickDraw414k/             # A.原始数据集 (.npy, .png)
│   └── generated_results/         # B.生成的草图
│
├── checkpoints/                   # 模型仓库(存放训练好的 .pth)
│   ├── classifier_seq.pth         # A.单模态模型
│   ├── classifier_dual.pth        # A.双模态模型
│   └──                            # B.生成模型
│
├── logs/                          # 【日志中心】存放训练日志 txt
│   ├── train_seq_20251227.txt
│   └── train_dual_20251227.txt
│
│
├── scripts_recognition/           # A.识别任务脚本
│   ├── train_unimodal.py          # 训练单模态
│   ├── train_dual.py              # 训练双模态
│   ├── eval_generation.py         # 评估生成质量
│
├── scripts_generation/            # B.生成任务脚本
│
├── README.md                      # 项目说明书
└── requirements.txt               # 依赖库列表
```
