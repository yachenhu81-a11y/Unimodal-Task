import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from datetime import datetime
import io 

# 工具
from utils import get_logger, SketchResNet

# ==========================================
# 1. 数据集
# ==========================================
class BalancedSketchDataset(Dataset):
    def __init__(self, data_root, mode='train', logger=None):
        self.base_dir = os.path.join(data_root, 'picture_files', mode)
        if not os.path.exists(self.base_dir): return
        
        all_items = os.listdir(self.base_dir)
        self.classes = sorted([d for d in all_items if os.path.isdir(os.path.join(self.base_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        limit = 5000 if mode == 'train' else 500
        
        if logger: logger.info(f"[{mode}] 正在将图片加载到内存...")
        
        count = 0
        for cls_name in self.classes:
            cls_folder = os.path.join(self.base_dir, cls_name)
            files = glob.glob(os.path.join(cls_folder, '*.png'))
            label = self.class_to_idx[cls_name]
            
            for f in files[:limit]:
                with open(f, 'rb') as img_f:
                    img_bytes = img_f.read()
                    self.samples.append((img_bytes, label))
                    count += 1
            
            if logger and count % 10000 == 0:
                print(f"  已加载 {count} 张...", end='\r')
        
        if logger: logger.info(f"\n[{mode}] 加载完成! 共 {len(self.samples)} 张。")
            
        self.img_size = 224
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        if mode == 'train': 
            self.transform.transforms.insert(2, transforms.RandomHorizontalFlip())
            self.transform.transforms.insert(3, transforms.RandomRotation(10))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, label = self.samples[idx]
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return self.transform(img), label
        except: 
            return torch.zeros((3, self.img_size, self.img_size)), label

# ==========================================
# 2. 训练主程序
# ==========================================
def train_img_balanced():
    # 日志名
    logger, _ = get_logger('../logs', name_prefix='train_img_scratch_best') 
    logger.info(f"=== 单模态图片训练开始: {datetime.now()} ===")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 128 
    
    train_ds = BalancedSketchDataset('../data/QuickDraw414k', 'train', logger)
    test_ds = BalancedSketchDataset('../data/QuickDraw414k', 'test', logger)
    
    if len(train_ds) == 0: return
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    logger.info(f"Batch Size: {BATCH_SIZE} | Image Size: 224x224")
    
    model = SketchResNet(len(train_ds.classes)).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(30):
        model.train()
        correct = 0; total = 0; total_loss = 0
        
        start_time = datetime.now()
        
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(imgs) # GPU 加粗
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item(); total += labels.size(0)
            
            if i % 50 == 0: 
                print(f"Epoch {epoch+1} Step {i}/{len(train_loader)} Loss: {loss.item():.4f}", end='\r')

        scheduler.step()
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        test_correct = 0; test_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                test_correct += (pred == labels).sum().item(); test_total += labels.size(0)
        
        test_acc = 100 * test_correct / test_total
        
        time_elapsed = datetime.now() - start_time
        logger.info(f"Epoch {epoch+1} ({time_elapsed}) | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Loss: {avg_loss:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '../checkpoints/classifier_img_best.pth')
            logger.info(f"Saved Best Model: {test_acc:.2f}%")

if __name__ == '__main__':
    train_img_balanced()
