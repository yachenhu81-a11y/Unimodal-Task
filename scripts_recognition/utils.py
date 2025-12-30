import os
import numpy as np
import torch
import torch.nn as nn
import logging
from datetime import datetime
from PIL import Image, ImageFilter
from torchvision import transforms, models

# ==========================================
# 1. 通用日志工具
# ==========================================
def get_logger(save_dir, name_prefix='train'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(save_dir, f'{name_prefix}_{timestamp}.txt')
    
    logger = logging.getLogger(name_prefix) 
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
# 2. 预处理工具
# ==========================================
class ThickenLines(object):
    def __init__(self, radius=1): self.radius = radius
    def __call__(self, img): return img.filter(ImageFilter.MaxFilter(size=self.radius*2+1))

def strokes_to_5stroke(data, max_len=200):
    if not isinstance(data, np.ndarray) or data.shape[1] != 4:
        return np.zeros((1, 5), dtype=np.float32)
    coords = data[:, 0:2] 
    deltas = np.zeros_like(coords)
    deltas[0] = coords[0] 
    deltas[1:] = coords[1:] - coords[:-1]
    pens = data[:, 2:4]
    result = np.hstack((deltas, pens, np.zeros((len(data), 1))))
    result = np.vstack((result, [[0, 0, 0, 0, 1]]))
    result = result.astype(np.float32)
    result[:, 0:2] /= 255.0 
    if len(result) > max_len:
        result = result[:max_len]
        result[-1, 2:] = [0, 0, 1]
    return result

# ==========================================
# 3. 模型定义
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
    def forward(self, x): return x + self.pe[:x.size(1), :].unsqueeze(0)

class SketchTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=5, d_model=256, nhead=8, num_layers=4):
        super(SketchTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = output.mean(dim=1)
        return self.classifier(output)

# --- 图片模型 ---
class SketchResNet(nn.Module):
    def __init__(self, num_classes):
        super(SketchResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        self.thickener = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.thickener(x)
        return self.backbone(x)
