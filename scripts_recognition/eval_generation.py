import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms, models

# ==========================================
# 0. 配置中心
# ==========================================
SEQ_MODEL_NAME = 'classifier_npy_best.pth'          # 序列单模态模型
IMG_MODEL_NAME = 'classifier_img_best.pth'  # 图片单模态模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 类定义 (保持与训练一致)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1), :].unsqueeze(0)

class SketchTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=5, d_model=256, nhead=8, num_layers=4):
        super(SketchTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src); x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask).mean(dim=1)
        return self.classifier(output)

class ThickenLines(object):
    def __init__(self, radius=1): self.radius = radius
    def __call__(self, img): return img.filter(ImageFilter.MaxFilter(size=self.radius*2+1))

class SketchResNet(nn.Module):
    def __init__(self, num_classes):
        super(SketchResNet, self).__init__()
        self.backbone = models.resnet18(weights=None) # From Scratch
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x): return self.backbone(x)

# ==========================================
# 2. 数据处理工具
# ==========================================
def strokes_to_5stroke(data, max_len=200):
    if not isinstance(data, np.ndarray) or data.shape[1] != 4: return np.zeros((1, 5), dtype=np.float32)
    coords = data[:, 0:2]; deltas = np.zeros_like(coords); deltas[0] = coords[0]; deltas[1:] = coords[1:] - coords[:-1]
    pens = data[:, 2:4]; result = np.hstack((deltas, pens, np.zeros((len(data), 1))))
    result = np.vstack((result, [[0, 0, 0, 0, 1]])); result = result.astype(np.float32); result[:, 0:2] /= 255.0
    if len(result) > max_len: result = result[:max_len]; result[-1, 2:] = [0, 0, 1]
    return result

# ==========================================
# 3. 评测主逻辑
# ==========================================
def judge_flexible():
    # 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current_dir)
    ckpt_dir = os.path.join(root, 'checkpoints')
    train_dir = os.path.join(root, 'data', 'QuickDraw414k', 'coordinate_files', 'train')
    gen_dir = os.path.join(root, 'data', 'generated_results')

    print("⚖️  启动灵活裁判 (Flexible Mode)...")

    # 1. 建立索引
    if not os.path.exists(train_dir): return print("❌ 找不到训练集路径")
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)

    # 2. 加载模型
    # (A) 序列模型
    model_seq = SketchTransformer(num_classes).to(DEVICE)
    seq_path = os.path.join(ckpt_dir, SEQ_MODEL_NAME)
    if os.path.exists(seq_path):
        model_seq.load_state_dict(torch.load(seq_path, map_location=DEVICE))
        model_seq.eval()
        print(f"✅ 序列模型就绪: {SEQ_MODEL_NAME}")
    else:
        print(f"⚠️ 序列模型未找到，将跳过 NPY 评测")
        model_seq = None

    # (B) 图片模型
    model_img = SketchResNet(num_classes).to(DEVICE)
    img_path = os.path.join(ckpt_dir, IMG_MODEL_NAME)
    if os.path.exists(img_path):
        model_img.load_state_dict(torch.load(img_path, map_location=DEVICE))
        model_img.eval()
        print(f"✅ 图片模型就绪: {IMG_MODEL_NAME}")
    else:
        print(f"⚠️ 图片模型未找到，将跳过 PNG 评测")
        model_img = None

    # 图片预处理 (对应 From Scratch 训练)
    img_transform = transforms.Compose([
        ThickenLines(radius=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print("\n" + "="*110)
    print(f"{'文件名':<25} | {'真实标签':<15} || {'NPY预测':<15} | {'PNG预测':<15} | {'最终结论'}")
    print("="*110)

    # 统计计数器
    stats = {
        'npy_total': 0, 'npy_correct': 0,
        'png_total': 0, 'png_correct': 0,
        'both_total': 0, 'both_correct': 0
    }

    if not os.path.exists(gen_dir): return print("❌ 生成目录不存在")

    for cls_name in os.listdir(gen_dir):
        if cls_name not in class_to_idx: continue
        cls_dir = os.path.join(gen_dir, cls_name)
        label_idx = class_to_idx[cls_name]
        
        # 获取所有无后缀的文件名
        basenames = set([os.path.splitext(f)[0] for f in os.listdir(cls_dir)])

        for base in basenames:
            npy_path = os.path.join(cls_dir, base + '.npy')
            png_path = os.path.join(cls_dir, base + '.png')
            
            has_npy = os.path.exists(npy_path)
            has_png = os.path.exists(png_path)
            
            if not has_npy and not has_png: continue

            # --- 1. 跑序列模型 ---
            seq_pred_name = "---"
            is_seq_right = None
            if has_npy and model_seq:
                try:
                    raw = np.load(npy_path, allow_pickle=True, encoding='latin1')
                    if raw.ndim == 0: raw = raw.item()
                    seq = strokes_to_5stroke(raw)
                    seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
                    padding_mask = (seq_tensor.abs().sum(dim=-1) == 0)
                    with torch.no_grad():
                        out = model_seq(seq_tensor, src_key_padding_mask=padding_mask)
                        pred = out.argmax(1).item()
                        seq_pred_name = idx_to_class[pred]
                        is_seq_right = (pred == label_idx)
                        stats['npy_total'] += 1
                        if is_seq_right: stats['npy_correct'] += 1
                except: seq_pred_name = "Error"

            # --- 2. 跑图片模型 ---
            img_pred_name = "---"
            is_img_right = None
            if has_png and model_img:
                try:
                    img = Image.open(png_path).convert('RGB')
                    img_tensor = img_transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = model_img(img_tensor)
                        pred = out.argmax(1).item()
                        img_pred_name = idx_to_class[pred]
                        is_img_right = (pred == label_idx)
                        stats['png_total'] += 1
                        if is_img_right: stats['png_correct'] += 1
                except: img_pred_name = "Error"

            # --- 3. 生成结论 ---
            tag = ""
            
            # 情况 A: 只有 NPY
            if has_npy and not has_png:
                if is_seq_right: tag = "✅ NPY正确"
                else: tag = "❌ NPY错误"
            
            # 情况 B: 只有 PNG
            elif not has_npy and has_png:
                if is_img_right: tag = "✅ PNG正确"
                else: tag = "❌ PNG错误"
            
            # 情况 C: 两个都有 (Both)
            elif has_npy and has_png:
                stats['both_total'] += 1
                if is_seq_right and is_img_right:
                    tag = "✅✅ 双赢 (完美)"
                    stats['both_correct'] += 1
                elif is_seq_right:
                    tag = "⚠️ 仅NPY对"
                elif is_img_right:
                    tag = "⚠️ 仅PNG对"
                else:
                    tag = "❌❌ 全错"

            print(f"{base:<25} | {cls_name:<15} || {seq_pred_name:<15} | {img_pred_name:<15} | {tag}")

    print("="*110)
    print("📊 统计报告")
    if stats['npy_total'] > 0:
        print(f"   - 序列模态准确率 (NPY): {100*stats['npy_correct']/stats['npy_total']:.2f}%  ({stats['npy_correct']}/{stats['npy_total']})")
    if stats['png_total'] > 0:
        print(f"   - 图片模态准确率 (PNG): {100*stats['png_correct']/stats['png_total']:.2f}%  ({stats['png_correct']}/{stats['png_total']})")
    if stats['both_total'] > 0:
        print(f"   - 双模态一致性准确率 (Both Correct): {100*stats['both_correct']/stats['both_total']:.2f}%")
    
    if stats['npy_total'] == 0 and stats['png_total'] == 0:
        print("   (未找到任何有效样本)")

if __name__ == '__main__':
    judge_flexible()