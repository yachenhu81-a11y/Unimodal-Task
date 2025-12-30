import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################### Hyperparameters
class HParams():
    def __init__(self):
        # 路径配置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.data_dir = os.path.join(self.project_root, 'data', 'QuickDraw_generation')
        
        # --- 修改：输出目录为 reconstruction ---
        self.output_dir = os.path.join(self.script_dir, 'reconstruction')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.categories = ['cat', 'apple', 'bus', 'angel', 'clock', 'pig', 'sheep', 'umbrella'] 
        
        # --- 模型参数 (保持一致) ---
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9 
        self.temperature = 0.4 # 重建时温度可以稍微低一点以求稳定，或者保持0.4增加多样性
        self.max_seq_length = 200
        
        self.load_epoch = 80000 

hp = HParams()

################################# Model Definition (保持不变)
class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, dropout=hp.dropout, bidirectional=True)
        self.fc_mu = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2*hp.enc_hidden_size, hp.Nz)

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            hidden = torch.zeros(2, batch_size, hp.enc_hidden_size).to(device)
            cell = torch.zeros(2, batch_size, hp.enc_hidden_size).to(device)
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        return mu, sigma_hat 

class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.fc_hc = nn.Linear(hp.Nz, 2*hp.dec_hidden_size)
        self.lstm = nn.LSTM(hp.Nz+5, hp.dec_hidden_size, dropout=hp.dropout)
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6*hp.M+3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])
        params_pen = params[-1]
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        
        len_out = 1
        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out, -1, hp.M)
        q = F.softmax(params_pen).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell

class ReconstructionModel():
    def __init__(self):
        self.encoder = EncoderRNN().to(device)
        self.decoder = DecoderRNN().to(device)
        self.encoder.eval()
        self.decoder.eval()

    def load(self, epoch):
        enc_path = os.path.join(hp.script_dir, 'test_sketch', f'encoder_epoch_{epoch}.pth')
        dec_path = os.path.join(hp.script_dir, 'test_sketch', f'decoder_epoch_{epoch}.pth')
        
        if not os.path.exists(enc_path):
            print(f"Error: Model not found at {enc_path}")
            return False
            
        self.encoder.load_state_dict(torch.load(enc_path, map_location=device))
        self.decoder.load_state_dict(torch.load(dec_path, map_location=device))
        print(f"Loaded model from epoch {epoch}")
        return True

    def generate_from_latent(self, z, scale_factor):
        """
        解码潜变量 z，并进行反归一化。
        返回格式: [N, 3] -> (Absolute_X, Absolute_Y, Pen_State)
        """
        sos = torch.tensor([0,0,1,0,0], device=device).view(1,1,-1)
        s = sos
        seq_x, seq_y, seq_z = [], [], []
        hidden_cell = None
        
        with torch.no_grad():
            for i in range(hp.max_seq_length):
                input = torch.cat([s, z.unsqueeze(0)], 2)
                self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                    self.rho_xy, self.q, hidden, cell = self.decoder(input, z, hidden_cell)
                hidden_cell = (hidden, cell)
                
                s, dx, dy, pen_down, eos = self.sample_next_state()
                
                # --- 反归一化 ---
                dx_real = dx * scale_factor
                dy_real = dy * scale_factor
                
                seq_x.append(dx_real)
                seq_y.append(dy_real)
                seq_z.append(pen_down)
                
                if eos: break
        
        # 计算绝对坐标
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        
        return np.stack([x_sample, y_sample, z_sample]).T

    def sample_next_state(self):
        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        q = self.q.data[0,0,:].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        
        mu_x = self.mu_x.data[0,0,pi_idx].cpu().item()
        mu_y = self.mu_y.data[0,0,pi_idx].cpu().item()
        sigma_x = self.sigma_x.data[0,0,pi_idx].cpu().item()
        sigma_y = self.sigma_y.data[0,0,pi_idx].cpu().item()
        rho_xy = self.rho_xy.data[0,0,pi_idx].cpu().item()
        
        mean = [mu_x, mu_y]
        sigma_x *= np.sqrt(hp.temperature)
        sigma_y *= np.sqrt(hp.temperature)
        cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
               [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
        x_val, y_val = np.random.multivariate_normal(mean, cov, 1)[0]
        
        next_state = torch.zeros(5)
        next_state[0] = x_val
        next_state[1] = y_val
        next_state[q_idx+2] = 1
        return next_state.to(device).view(1,1,-1), x_val, y_val, q_idx==1, q_idx==2

################################# Data Helper Functions
def purify(strokes):
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def prepare_data_dict():
    all_raw_data = []
    data_by_cat = {}

    print("Loading datasets to calculate scale factor...")
    for cat in hp.categories:
        path = os.path.join(hp.data_dir, f'{cat}.npz')
        if os.path.exists(path):
            raw = np.load(path, encoding='latin1', allow_pickle=True)['train']
            purified = purify(raw)
            all_raw_data.extend(purified)
            data_by_cat[cat] = purified
    
    # Global Scale Factor
    flat_data = []
    for seq in all_raw_data:
        flat_data.extend(seq[:, :2].flatten())
    scale_factor = np.std(flat_data)
    print(f"Global Scale Factor: {scale_factor:.4f}")

    # Normalize data
    norm_data_by_cat = {}
    for cat, sequences in data_by_cat.items():
        norm_seqs = []
        for seq in sequences:
            n_seq = seq.copy()
            n_seq[:, :2] /= scale_factor
            norm_seqs.append(n_seq)
        norm_data_by_cat[cat] = norm_seqs
        
    return norm_data_by_cat, scale_factor

def make_tensor_input(sequence):
    len_seq = len(sequence[:,0])
    Nmax = hp.max_seq_length
    new_seq = np.zeros((Nmax, 5))
    new_seq[:len_seq, :2] = sequence[:, :2]
    new_seq[:len_seq-1, 2] = 1 - sequence[:-1, 2]
    new_seq[:len_seq, 3] = sequence[:, 2]
    new_seq[(len_seq-1):, 4] = 1
    new_seq[len_seq-1, 2:4] = 0
    return torch.from_numpy(new_seq).float().unsqueeze(1).to(device)

def save_plot(sequence, filename, title, color='blue'):
    """
    输入: [N, 3] 绝对坐标 (x, y, pen)
    """
    abs_x = sequence[:, 0]
    abs_y = sequence[:, 1]
    pen_states = sequence[:, 2]

    split_indices = np.where(pen_states > 0)[0] + 1
    strokes = np.split(np.stack([abs_x, abs_y], axis=1), split_indices)
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    for s in strokes:
        if len(s) > 0:
            # Y轴加负号
            plt.plot(s[:, 0], -s[:, 1], color=color, linewidth=2)
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    
    save_path = os.path.join(hp.output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Image Saved: {save_path}")

def save_npy(sequence, filename):
    """
    保存为 npy 文件
    输入: [N, 3] 绝对坐标
    """
    save_path = os.path.join(hp.output_dir, filename)
    np.save(save_path, sequence)
    print(f"Data Saved: {save_path}")

################################# Main Reconstruction Logic
def reconstruct_sample(cat_name, idx):
    # 1. 准备数据和缩放因子
    # 注意：这里会重新加载数据计算scale factor，虽然有点慢但保证准确
    if not hasattr(reconstruct_sample, "data_cache"):
         reconstruct_sample.data_cache = prepare_data_dict()
    data_dict, scale_factor = reconstruct_sample.data_cache
    
    if cat_name not in data_dict:
        print(f"Error: Category {cat_name} not found.")
        return
    
    if idx >= len(data_dict[cat_name]):
        print(f"Error: Index {idx} out of range for {cat_name}.")
        return

    print(f"\nProcessing: {cat_name} (ID: {idx})")

    # 2. 获取原始数据 (Normalized)
    seq_norm = data_dict[cat_name][idx] 
    
    # 3. 转换为 Tensor 输入模型
    input_tensor = make_tensor_input(seq_norm)

    # 4. 加载模型
    model = ReconstructionModel()
    if not model.load(hp.load_epoch):
        return

    # 5. 模型推理 (Encode -> Decode)
    with torch.no_grad():
        z, _ = model.encoder(input_tensor, 1)
        # 生成重建数据 (已包含反归一化和Cumsum，结果为绝对坐标)
        recon_absolute = model.generate_from_latent(z, scale_factor)

    # 6. 处理原始数据用于保存 (Norm Delta -> Real Delta -> Real Absolute)
    def denormalize_to_absolute(norm_seq, scale):
        real_seq = norm_seq.copy()
        real_seq[:, :2] *= scale # 反归一化
        abs_x = np.cumsum(real_seq[:, 0])
        abs_y = np.cumsum(real_seq[:, 1])
        return np.stack([abs_x, abs_y, real_seq[:, 2]], axis=1)

    orig_absolute = denormalize_to_absolute(seq_norm, scale_factor)

    # 7. 保存文件 (4个文件)
    # 文件名基础
    base_name = f"{cat_name}_id{idx}"

    # A. 保存原始数据 npy (绝对坐标)
    save_npy(orig_absolute, f"original_{base_name}.npy")
    
    # B. 保存重构数据 npy (绝对坐标)
    save_npy(recon_absolute, f"recon_{base_name}.npy")
    
    # C. 保存原始图片 png
    save_plot(orig_absolute, 
              f"original_{base_name}.png", 
              f"Original: {cat_name} (ID:{idx})", 
              color='black')
              
    # D. 保存重构图片 png
    save_plot(recon_absolute, 
              f"recon_{base_name}.png", 
              f"Reconstruction: {cat_name} (ID:{idx})", 
              color='green')

    print("--- Done ---")

if __name__ == '__main__':
    # ================= 使用说明 =================
    # 在这里指定你要重建的类别和 ID
    # 程序会生成 原图PNG, 重构PNG, 原数据NPY, 重构数据NPY
    # ===========================================
    
    target_category = 'apple'
    target_id = 11913
    
    reconstruct_sample(target_category, target_id)
    
    # 你也可以一次跑多个
    # reconstruct_sample('clock', 7182)