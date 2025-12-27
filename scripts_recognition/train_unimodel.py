import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
from datetime import datetime

# ==========================================
# 0. 路径与日志配置
# ==========================================
def get_logger(save_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(save_dir, f'train_seq_best_{timestamp}.txt')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [] 

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s')) 
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    
    return logger, log_filename

# ==========================================
# 1. 核心工具
# ==========================================
def strokes_to_5stroke(data, max_len=200):
    if not isinstance(data, np.ndarray) or data.shape[1] != 4:
        return np.zeros((1, 5), dtype=np.float32)
    coords = data[:, 0:2] 
    deltas = np.zeros_like(coords)
    deltas[0] = coords[0] 
    deltas[1:] = coords[1:] - coords[:-1]
    pens = data[:, 2:4]
    p3 = np.zeros((len(data), 1))
    result = np.hstack((deltas, pens, p3))
    end_token = np.array([[0, 0, 0, 0, 1]])
    result = np.vstack((result, end_token))
    result = result.astype(np.float32)
    result[:, 0:2] /= 255.0 
    if len(result) > max_len:
        result = result[:max_len]
        result[-1, 2:] = [0, 0, 1]
    return result

# ==========================================
# 2. 数据集类
# ==========================================
class SketchSequenceDataset(Dataset):
    def __init__(self, data_root, mode='train', max_len=196, logger=None):
        self.max_len = max_len
        self.base_dir = os.path.join(data_root, 'coordinate_files', mode)
        
        if not os.path.exists(self.base_dir):
            if logger: logger.info(f"【严重错误】找不到数据文件夹: {self.base_dir}")
            self.samples = []
            return

        all_items = os.listdir(self.base_dir)
        self.class_names = sorted([d for d in all_items if os.path.isdir(os.path.join(self.base_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.classes = self.class_names
        
        if logger: logger.info(f"[{mode}] 发现 {len(self.classes)} 个类别...")
        
        self.samples = []
        limit = 5000 if mode == 'train' else 500 
        
        if logger: logger.info(f"[{mode}] 正在索引文件 (每类 {limit} 个)...")
        
        for cls_name in self.class_names:
            cls_folder = os.path.join(self.base_dir, cls_name)
            npy_files = glob.glob(os.path.join(cls_folder, '*.npy'))
            label_idx = self.class_to_idx[cls_name]
            for f_path in npy_files[:limit]:
                self.samples.append((f_path, label_idx))
                
        if logger: logger.info(f"[{mode}] 索引完成，共 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            raw_data = np.load(file_path, allow_pickle=True, encoding='latin1')
            if raw_data.ndim == 0: data = raw_data.item()
            else: data = raw_data
            seq = strokes_to_5stroke(data, self.max_len)
            return torch.from_numpy(seq), label
        except:
            return torch.zeros((10, 5), dtype=torch.float32), label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded_seqs, labels

# ==========================================
# 3. 模型定义 (Transformer)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class SketchTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=5, d_model=256, nhead=8, num_layers=4):
        super(SketchTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = output.mean(dim=1)
        return self.classifier(output)

# ==========================================
# 4. 训练主程序
# ==========================================
def train_unimodal():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    LOG_DIR = os.path.join(project_root, 'logs')
    CKPT_DIR = os.path.join(project_root, 'checkpoints')
    DATA_PATH = os.path.join(project_root, 'data', 'QuickDraw414k') 
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # --- 1. 初始化 ---
    logger, log_filename = get_logger(LOG_DIR)
    logger.info(f"=== 单模态训练 (Save Best) 开始: {datetime.now()} ===")
    
    BATCH_SIZE = 128 
    LR = 0.0003
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SketchSequenceDataset(DATA_PATH, mode='train', logger=logger)
    test_ds = SketchSequenceDataset(DATA_PATH, mode='test', logger=logger)
    if len(train_ds) == 0: return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    num_classes = len(train_ds.classes)
    model = SketchTransformer(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    logger.info("开始训练...")
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for i, (seqs, labels) in enumerate(train_loader):
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            padding_mask = (seqs.abs().sum(dim=-1) == 0)
            
            optimizer.zero_grad()
            outputs = model(seqs, src_key_padding_mask=padding_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if i % 100 == 0:
                logger.info(f"  Epoch {epoch+1} Step {i}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        scheduler.step()
        train_acc = 100 * correct / total_samples
        avg_loss = total_loss/len(train_loader)
        logger.info(f"Epoch {epoch+1} Train Acc: {train_acc:.2f}% | Avg Loss: {avg_loss:.4f}")
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                padding_mask = (seqs.abs().sum(dim=-1) == 0)
                outputs = model(seqs, src_key_padding_mask=padding_mask)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = 100 * test_correct / test_total
        logger.info(f"Epoch {epoch+1} Test Acc: {test_acc:.2f}%")
        

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(CKPT_DIR, 'classifier_seq_best.pth')
            torch.save(model.state_dict(), save_path)


    logger.info(f"单模态训练结束。最终最强模型保存在: {os.path.join(CKPT_DIR, 'classifier_seq_best.pth')}")

if __name__ == '__main__':
    train_unimodal()