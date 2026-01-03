import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HParams():
    def __init__(self):
        # 路径配置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.data_dir = os.path.join(self.project_root, 'data', 'QuickDraw_generation')
        self.categories = ['cat', 'apple', 'bus', 'angel', 'clock', 'pig', 'sheep', 'umbrella'] 
        self.max_seq_length = 200

hp = HParams()

def purify(strokes):
    data = []
    # 必须保留原始索引关系，不能随意丢弃，否则ID会对不上
    # 但为了保证模型能跑，我们只保留符合长度的数据
    # 为了让 ID 在两个脚本间通用，我们必须使用完全相同的 purify 逻辑
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def load_category_data(category):
    """只加载特定类别的数据用于查看"""
    path = os.path.join(hp.data_dir, f'{category}.npz')
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    
    raw = np.load(path, encoding='latin1', allow_pickle=True)['train']
    # 必须经过 purify，因为 interpolation.py 也是基于 purify 后的数据列表取索引的
    data = purify(raw)
    return data

def plot_sketches(category_name, num_samples=5):
    data = load_category_data(category_name)
    total_samples = len(data)
    
    if total_samples == 0:
        return

    # 随机挑选 5 个索引
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    print(f"Showing {num_samples} samples for '{category_name}'...")

    plt.figure(figsize=(15, 3))
    
    for i, idx in enumerate(random_indices):
        #idx = 52215
        seq = data[idx]
        
        # 转换为绝对坐标用于绘图
        # 原数据是 delta_x, delta_y
        abs_x = np.cumsum(seq[:, 0])
        abs_y = np.cumsum(seq[:, 1])
        pen_states = seq[:, 2] # 笔画结束标志
        
        split_indices = np.where(pen_states > 0)[0] + 1
        strokes = np.split(np.stack([abs_x, abs_y], axis=1), split_indices)
        
        ax = plt.subplot(1, num_samples, i+1)
        for s in strokes:
            if len(s) > 0:
                # 注意 y 轴反转以符合屏幕坐标习惯
                ax.plot(s[:, 0], -s[:, 1], color='black')
        
        ax.set_title(f"ID: {idx}") # 关键：显示ID
        ax.axis('equal')
        ax.axis('off')
        
    plt.suptitle(f"Category: {category_name}", fontsize=16)
    plt.show()

if __name__ == '__main__':
    # 修改这里来查看不同类别的图
    # 运行后会弹窗，记下你喜欢的图片的 ID
    plot_sketches('apple', num_samples=5)
    plot_sketches('pig', num_samples=5)