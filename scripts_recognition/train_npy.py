import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime

# 导入工具
from utils import get_logger, strokes_to_5stroke, SketchTransformer

# ==========================================
# 1. 数据集
# ==========================================
class SketchSequenceDataset(Dataset):
    def __init__(self, data_root, mode='train', max_len=196, logger=None):
        self.max_len = max_len
        self.base_dir = os.path.join(data_root, 'coordinate_files', mode)
        if not os.path.exists(self.base_dir): return
        
        all_items = os.listdir(self.base_dir)
        self.classes = sorted([d for d in all_items if os.path.isdir(os.path.join(self.base_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        limit = 5000 if mode == 'train' else 500
        if logger: logger.info(f"[{mode}] 索引序列数据 (Limit: {limit})...")
        
        for cls_name in self.classes:
            cls_folder = os.path.join(self.base_dir, cls_name)
            files = glob.glob(os.path.join(cls_folder, '*.npy'))
            label = self.class_to_idx[cls_name]
            for f in files[:limit]: self.samples.append((f, label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        f_path, label = self.samples[idx]
        try:
            raw = np.load(f_path, allow_pickle=True, encoding='latin1')
            if raw.ndim == 0: raw = raw.item()
            seq = strokes_to_5stroke(raw, self.max_len)
            return torch.from_numpy(seq), label
        except: return torch.zeros((10, 5), dtype=torch.float32), label

def collate_fn(batch):
    seqs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=True, padding_value=0), torch.tensor(labels)

# ==========================================
# 2. 训练主程序
# ==========================================
def train_unimodal():
    # 注意：这里 name_prefix 决定了日志文件名
    logger, _ = get_logger('../logs', name_prefix='train_npy_best') 
    logger.info(f"=== 单模态序列训练开始: {datetime.now()} ===")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    
    train_ds = SketchSequenceDataset('../data/QuickDraw414k', 'train', logger=logger)
    test_ds = SketchSequenceDataset('../data/QuickDraw414k', 'test', logger=logger)
    if len(train_ds) == 0: return
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = SketchTransformer(len(train_ds.classes)).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(30):
        model.train()
        correct = 0; total = 0; total_loss = 0
        for i, (seqs, labels) in enumerate(train_loader):
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            mask = (seqs.abs().sum(dim=-1) == 0)
            
            optimizer.zero_grad()
            out = model(seqs, src_key_padding_mask=mask)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item(); total += labels.size(0)
            if i % 100 == 0: print(f"Epoch {epoch+1} Step {i}/{len(train_loader)} Loss: {loss.item():.4f}", end='\r')

        scheduler.step()
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # 验证
        model.eval()
        test_correct = 0; test_total = 0
        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                mask = (seqs.abs().sum(dim=-1) == 0)
                out = model(seqs, src_key_padding_mask=mask)
                _, pred = torch.max(out, 1)
                test_correct += (pred == labels).sum().item(); test_total += labels.size(0)
        
        test_acc = 100 * test_correct / test_total
        logger.info(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Loss: {avg_loss:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '../checkpoints/classifier_npy_best.pth')
            logger.info(f"Saved Best Model: {test_acc:.2f}%")

if __name__ == '__main__':
    train_unimodal()