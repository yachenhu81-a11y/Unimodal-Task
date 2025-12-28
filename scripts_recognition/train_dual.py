import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
import numpy as np
from datetime import datetime

from utils import get_logger, strokes_to_5stroke, SketchTransformer, SketchResNet

# ==========================================
# 1. 融合网络
# ==========================================
class LateFusionNetwork(nn.Module):
    def __init__(self, num_classes, seq_model, img_model):
        super(LateFusionNetwork, self).__init__()
        
        # 1. 序列分支处理
        self.seq_encoder = seq_model
        if hasattr(self.seq_encoder, 'classifier'): 
            self.seq_encoder.classifier = nn.Identity()
        
        # 2. 图片分支处理
        self.img_encoder = img_model
        if hasattr(self.img_encoder, 'backbone'): 
            self.img_encoder.backbone.fc = nn.Identity()
        elif hasattr(self.img_encoder, 'fc'): 
            self.img_encoder.fc = nn.Identity()

        # 3. 融合层
        # 256 (Seq) + 512 (Img) = 768
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 512, 512),  # <--- 这里改成 256 + 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, seq, img, padding_mask):
        feat_seq = self.seq_encoder(seq, src_key_padding_mask=padding_mask)
        feat_img = self.img_encoder(img)
        
        # 拼接
        combined = torch.cat((feat_seq, feat_img), dim=1)
        
        # 最终分类
        return self.fusion_head(combined)


# ==========================================
# 2. 双模态数据集
# ==========================================
class DualModalDataset(Dataset):
    def __init__(self, data_root, mode='train', max_len=196, logger=None):
        self.max_len = max_len
        self.seq_dir = os.path.join(data_root, 'coordinate_files', mode)
        self.img_dir = os.path.join(data_root, 'picture_files', mode)
        
        all_items = os.listdir(self.seq_dir)
        self.classes = sorted([d for d in all_items if os.path.isdir(os.path.join(self.seq_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        limit = 5000 if mode == 'train' else 500
        if logger: logger.info(f"[{mode}] 对齐双模态数据 (Limit: {limit})...")

        for cls_name in self.classes:
            seq_path = os.path.join(self.seq_dir, cls_name)
            img_path = os.path.join(self.img_dir, cls_name)
            f_names = sorted(os.listdir(seq_path))[:limit]
            label = self.class_to_idx[cls_name]
            for f in f_names:
                if f.endswith('.npy'):
                    base = f[:-4]
                    self.samples.append({
                        'npy': os.path.join(seq_path, f),
                        'png': os.path.join(img_path, base + '.png'),
                        'label': label
                    })

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            raw = np.load(item['npy'], allow_pickle=True, encoding='latin1')
            if raw.ndim == 0: raw = raw.item()
            seq = strokes_to_5stroke(raw, self.max_len)
            seq = torch.from_numpy(seq)
        except: seq = torch.zeros((10, 5), dtype=torch.float32)
        
        try:
            img = Image.open(item['png']).convert('RGB')
            img = self.transform(img)
        except: img = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        return seq, img, item['label']

def collate_fn_dual(batch):
    seqs, imgs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=True, padding_value=0), torch.stack(imgs), torch.tensor(labels)

def train_fusion():
    logger, _ = get_logger('../logs', name_prefix='train_dual_finetune') 
    logger.info(f"=== 双模态训练开始: {datetime.now()} ===")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = DualModalDataset('../data/QuickDraw414k', 'train', logger=logger)
    test_ds = DualModalDataset('../data/QuickDraw414k', 'test', logger=logger)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn_dual, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn_dual, num_workers=8, pin_memory=True)

    logger.info("加载模型...")
    seq_model = SketchTransformer(len(train_ds.classes)).to(DEVICE)
    seq_model.load_state_dict(torch.load('../checkpoints/classifier_npy_best.pth', map_location=DEVICE))
    
    img_model = SketchResNet(len(train_ds.classes)).to(DEVICE)
    img_model.load_state_dict(torch.load('../checkpoints/classifier_img_best.pth', map_location=DEVICE))
    
    model = LateFusionNetwork(len(train_ds.classes), seq_model, img_model).to(DEVICE)
    
    optimizer = torch.optim.AdamW([
        {'params': model.seq_encoder.parameters(), 'lr': 1e-5}, 
        {'params': model.img_encoder.parameters(), 'lr': 1e-5},
        {'params': model.fusion_head.parameters(), 'lr': 1e-3} 
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(10): 
        model.train()
        correct = 0; total = 0; total_loss = 0
        for i, (seqs, imgs, labels) in enumerate(train_loader):
            seqs, imgs, labels = seqs.to(DEVICE), imgs.to(DEVICE), labels.to(DEVICE)
            mask = (seqs.abs().sum(dim=-1) == 0)
            
            optimizer.zero_grad()
            out = model(seqs, imgs, mask)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item(); total += labels.size(0)
            if i % 100 == 0: print(f"Epoch {epoch+1} Step {i}/{len(train_loader)} Loss: {loss.item():.4f}", end='\r')

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        test_correct = 0; test_total = 0
        with torch.no_grad():
            for seqs, imgs, labels in test_loader:
                seqs, imgs, labels = seqs.to(DEVICE), imgs.to(DEVICE), labels.to(DEVICE)
                mask = (seqs.abs().sum(dim=-1) == 0)
                out = model(seqs, imgs, mask)
                _, pred = torch.max(out, 1)
                test_correct += (pred == labels).sum().item(); test_total += labels.size(0)
        
        test_acc = 100 * test_correct / test_total
        logger.info(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Loss: {avg_loss:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '../checkpoints/classifier_dual_final.pth')
            logger.info(f"Saved Best Dual Model: {test_acc:.2f}%")

if __name__ == '__main__':
    train_fusion()
