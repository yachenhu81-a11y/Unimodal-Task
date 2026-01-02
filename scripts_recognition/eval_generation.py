import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

# 导入工具包
from utils import SketchResNet, SketchTransformer, strokes_to_5stroke

# =================配置区域=================
# 1. 待评估的文件夹 (生成结果存放处)
GEN_ROOT = '../data/generated_results' 

# 2. 原始数据文件夹 (仅用于获取类别列表)
TRAIN_DATA_ROOT = '../data/QuickDraw414k/picture_files/train'

# 3. 模型路径 (退两层 checkpoints)
CKPT_SEQ  = '../checkpoints/classifier_npy_best.pth'
CKPT_IMG  = '../checkpoints/classifier_img_best.pth'
CKPT_DUAL = '../checkpoints/classifier_dual_final.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================

# --- 双模态网络定义 ---
class LateFusionNetwork(nn.Module):
    def __init__(self, num_classes, seq_model, img_model):
        super(LateFusionNetwork, self).__init__()
        self.seq_encoder = seq_model
        if hasattr(self.seq_encoder, 'classifier'): self.seq_encoder.classifier = nn.Identity()
        self.img_encoder = img_model
        if hasattr(self.img_encoder, 'backbone'): self.img_encoder.backbone.fc = nn.Identity()
        elif hasattr(self.img_encoder, 'fc'): self.img_encoder.fc = nn.Identity()

        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, seq, img, padding_mask):
        feat_seq = self.seq_encoder(seq, src_key_padding_mask=padding_mask)
        feat_img = self.img_encoder(img)
        return self.fusion_head(torch.cat((feat_seq, feat_img), dim=1))

def get_classes():
    """获取所有类别名称"""
    if not os.path.exists(TRAIN_DATA_ROOT):
        raise FileNotFoundError(f"需要原始数据目录来获取类别列表: {TRAIN_DATA_ROOT}")
    return sorted([d for d in os.listdir(TRAIN_DATA_ROOT) if os.path.isdir(os.path.join(TRAIN_DATA_ROOT, d))])

def load_models(num_classes):
    """一次性加载所有模型到显存，避免重复加载"""
    print("正在加载所有模型...")
    
    # 1. 序列模型
    model_seq = SketchTransformer(num_classes).to(DEVICE)
    model_seq.load_state_dict(torch.load(CKPT_SEQ, map_location=DEVICE))
    model_seq.eval()
    
    # 2. 图片模型
    model_img = SketchResNet(num_classes).to(DEVICE)
    model_img.load_state_dict(torch.load(CKPT_IMG, map_location=DEVICE))
    model_img.eval()
    
    # 3. 双模态模型
    seq_base = SketchTransformer(num_classes)
    img_base = SketchResNet(num_classes)
    model_dual = LateFusionNetwork(num_classes, seq_base, img_base).to(DEVICE)
    model_dual.load_state_dict(torch.load(CKPT_DUAL, map_location=DEVICE))
    model_dual.eval()
    
    print("模型加载完毕！")
    return model_seq, model_img, model_dual

def preprocess_img(path):
    """读取并处理图片"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    try:
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"图片损坏: {path}")
        return None

def preprocess_seq(path):
    """读取并处理序列"""
    try:
        raw = np.load(path, allow_pickle=True, encoding='latin1')
        if raw.ndim == 0: raw = raw.item()
        seq = strokes_to_5stroke(raw, max_len=196)
        return torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"NPY损坏: {path}")
        return None

def main():
    if not os.path.exists(GEN_ROOT):
        print(f"错误：找不到生成结果文件夹: {GEN_ROOT}")
        return

    # 1. 准备工作
    classes = get_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    # 加载三个模型
    model_seq, model_img, model_dual = load_models(num_classes)

    print(f"\n开始评估: {GEN_ROOT}")
    print("="*60)

    total_correct = 0
    total_samples = 0
    
    # 2. 遍历每个类别文件夹
    # 假设结构: data/generated_results/cat/xxx.png
    gen_classes = sorted([d for d in os.listdir(GEN_ROOT) if os.path.isdir(os.path.join(GEN_ROOT, d))])
    
    for cls_name in gen_classes:
        if cls_name not in class_to_idx:
            print(f"跳过未知类别: {cls_name}")
            continue
            
        cls_dir = os.path.join(GEN_ROOT, cls_name)
        target_label = class_to_idx[cls_name]
        
        # 3. 分组文件：将同名的 .npy 和 .png 配对
        file_groups = defaultdict(dict)
        for f in os.listdir(cls_dir):
            base_name, ext = os.path.splitext(f)
            if ext == '.npy':
                file_groups[base_name]['npy'] = os.path.join(cls_dir, f)
            elif ext == '.png':
                file_groups[base_name]['png'] = os.path.join(cls_dir, f)
        
        if len(file_groups) == 0:
            continue

        cls_correct = 0
        cls_total = 0
        
        # 4. 对该类别下的所有样本进行推理
        # 使用 tqdm 显示进度条
        pbar = tqdm(file_groups.items(), desc=f"评估 [{cls_name}]", leave=False)
        
        for base_name, paths in pbar:
            npy_path = paths.get('npy')
            png_path = paths.get('png')
            
            output = None
            
            with torch.no_grad():
                # === 策略选择 ===
                
                # 情况 A: 双模态
                if npy_path and png_path:
                    s_data = preprocess_seq(npy_path)
                    i_data = preprocess_img(png_path)
                    if s_data is not None and i_data is not None:
                        mask = (s_data.abs().sum(dim=-1) == 0)
                        output = model_dual(s_data, i_data, mask)
                
                # 情况 B: 只有图片
                elif png_path and not npy_path:
                    i_data = preprocess_img(png_path)
                    if i_data is not None:
                        output = model_img(i_data)
                
                # 情况 C: 只有序列
                elif npy_path and not png_path:
                    s_data = preprocess_seq(npy_path)
                    if s_data is not None:
                        mask = (s_data.abs().sum(dim=-1) == 0)
                        output = model_seq(s_data, src_key_padding_mask=mask)
            
            if output is not None:
                pred = torch.argmax(output, dim=1).item()
                if pred == target_label:
                    cls_correct += 1
                cls_total += 1
        
        # 统计该类别
        if cls_total > 0:
            acc = 100 * cls_correct / cls_total
            print(f"类别: {cls_name:<15} | 样本数: {cls_total:<4} | 识别准确率: {acc:.2f}%")
            total_correct += cls_correct
            total_samples += cls_total
            
    # 5. 最终汇总
    print("="*60)
    if total_samples > 0:
        final_acc = 100 * total_correct / total_samples
        print(f"总体评估完成!")
        print(f"总样本数: {total_samples}")
        print(f"总体识别准确率: {final_acc:.2f}%")
    else:
        print("未找到任何有效样本。")

if __name__ == '__main__':
    main()

