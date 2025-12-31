import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from collections import defaultdict
from tqdm import tqdm

# 导入工具包
from utils import SketchResNet, SketchTransformer, strokes_to_5stroke

# =================配置区域=================
GEN_ROOT = '../data/generated_results' 
TRAIN_DATA_ROOT = '../data/QuickDraw414k/picture_files/train'

# 模型路径
CKPT_SEQ  = '../checkpoints/classifier_npy_best.pth'
CKPT_IMG  = '../checkpoints/classifier_img_best.pth'
CKPT_DUAL = '../checkpoints/classifier_dual_final.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 强制颜色反转
FORCE_INVERT = True 
# =========================================

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
    if not os.path.exists(TRAIN_DATA_ROOT):
        if os.path.exists(GEN_ROOT):
            return sorted([d for d in os.listdir(GEN_ROOT) if os.path.isdir(os.path.join(GEN_ROOT, d))])
    return sorted([d for d in os.listdir(TRAIN_DATA_ROOT) if os.path.isdir(os.path.join(TRAIN_DATA_ROOT, d))])

def load_models(num_classes):
    print("⚙️  正在加载模型...")
    model_seq = SketchTransformer(num_classes).to(DEVICE).eval()
    model_seq.load_state_dict(torch.load(CKPT_SEQ, map_location=DEVICE))
    
    model_img = SketchResNet(num_classes).to(DEVICE).eval()
    model_img.load_state_dict(torch.load(CKPT_IMG, map_location=DEVICE))
    
    seq_base = SketchTransformer(num_classes)
    img_base = SketchResNet(num_classes)
    model_dual = LateFusionNetwork(num_classes, seq_base, img_base).to(DEVICE).eval()
    model_dual.load_state_dict(torch.load(CKPT_DUAL, map_location=DEVICE))
    return model_seq, model_img, model_dual

def preprocess_img(path):
    """
    读取并处理图片
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    try:
        img = Image.open(path)
        
        # 2. 强制转为灰度图 (L模式)
        # 只保留亮度
        img = img.convert('L')
        
        # 3. 二值化 (Binarization)
        # 阈值设为 240 ，只要不是接近纯白的，都算作线
        threshold = 240 
        img = img.point(lambda p: 0 if p < threshold else 255)
        
        # 4. 颜色反转逻辑 (白底黑线 -> 黑底白线)
        if FORCE_INVERT:
            img = F.invert(img)
            
        # 5. 转回 RGB
        img = img.convert('RGB')

        return transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        # print(f"图片处理出错: {path} {e}")
        return None

def preprocess_seq(path):
    try:
        raw = np.load(path, allow_pickle=True, encoding='latin1')
        if raw.ndim == 0: raw = raw.item()
        
        # (N,3) -> (N,4)
        if raw.ndim == 2 and raw.shape[1] == 3:
            coords = raw[:, 0:2]
            p = raw[:, 2]
            new_pens = np.zeros((len(p), 2), dtype=np.float32)
            lift_mask = (p > 0)
            new_pens[lift_mask, 1] = 1 
            new_pens[~lift_mask, 0] = 1 
            raw = np.hstack((coords, new_pens))

        seq = strokes_to_5stroke(raw, max_len=196)
        if np.all(seq == 0): return None
        return torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
    except Exception:
        return None

def main():
    if not os.path.exists(GEN_ROOT): return
    classes = get_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    model_seq, model_img, model_dual = load_models(num_classes)

    print(f"\n🚀 开始评估: {GEN_ROOT}")
    print("="*60)

    gen_classes = sorted([d for d in os.listdir(GEN_ROOT) if os.path.isdir(os.path.join(GEN_ROOT, d))])
    
    total_correct = 0
    total_samples = 0

    for cls_name in gen_classes:
        if cls_name not in class_to_idx: continue
        cls_dir = os.path.join(GEN_ROOT, cls_name)
        target_label = class_to_idx[cls_name]
        
        files = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
        if not files: continue

        print(f"\n📂 正在评估类别: [{cls_name}]")
        print("-" * 60)
        
        cls_correct = 0
        cls_total = 0
        
        for f in files:
            base_name = os.path.splitext(f)[0]
            png_path = os.path.join(cls_dir, f)
            npy_path = os.path.join(cls_dir, base_name + '.npy')
            
            i_data = preprocess_img(png_path)
            s_data = preprocess_seq(npy_path) if os.path.exists(npy_path) else None
            
            output = None
            mode = "Img" # 记录用了什么模态
            
            with torch.no_grad():
                # 尝试双模态
                if s_data is not None and i_data is not None:
                    mask = (s_data.abs().sum(dim=-1) == 0)
                    if not mask.all():
                        output = model_dual(s_data, i_data, mask)
                        mode = "Dual"
                    else:
                        output = model_img(i_data)
                # 降级单模态
                elif i_data is not None:
                    output = model_img(i_data)
            
            if output is not None:
                # 获取预测结果
                probs = torch.softmax(output, dim=1)
                pred_prob, pred_idx = torch.max(probs, 1)
                pred_label = pred_idx.item()
                pred_name = classes[pred_label]
                
                is_correct = (pred_label == target_label)
                if is_correct: cls_correct += 1
                cls_total += 1
                
                # === 打印单条结果 ===
                mark = "✅" if is_correct else "❌"
                # 格式: ✅ [Dual] apple_0.png : apple -> apple (0.99)
                # 格式: ❌ [Img ] apple_1.png : apple -> pear  (0.45)
                print(f"{mark} [{mode}] {f:<20} : {cls_name} -> {pred_name:<10} (Conf: {pred_prob.item():.2f})")
        
        if cls_total > 0:
            acc = 100 * cls_correct / cls_total
            print("-" * 60)
            print(f"📊 {cls_name} 准确率: {acc:.2f}% ({cls_correct}/{cls_total})")
            total_correct += cls_correct
            total_samples += cls_total

    print("="*60)
    if total_samples > 0:
        final_acc = 100 * total_correct / total_samples
        print(f"🏆 总体评估完成!")
        print(f"🎯 总体识别准确率: {final_acc:.2f}%")

if __name__ == '__main__':
    main()
