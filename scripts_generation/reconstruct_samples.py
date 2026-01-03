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
        
        # 输出目录
        self.output_dir = os.path.join(self.script_dir, 'reconstruction')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.categories = ['cat', 'apple', 'bus', 'angel', 'clock', 'pig', 'sheep', 'umbrella'] 
        
        # --- 模型参数 ---
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9 
        self.temperature = 0.4 
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
        解码并反归一化。
        返回格式: [N, 4] -> (Absolute_X, Absolute_Y, Pen_Lift, Pen_EOS)
        """
        sos = torch.tensor([0,0,1,0,0], device=device).view(1,1,-1)
        s = sos
        seq_x, seq_y = [], []
        seq_p2, seq_p3 = [], [] # p2: lift, p3: eos
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
                # q_idx=1 -> Lift(p2), q_idx=2 -> EOS(p3)
                seq_p2.append(1 if pen_down else 0)
                seq_p3.append(1 if eos else 0)
                
                if eos: break
        
        # 计算绝对坐标
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        p2_sample = np.array(seq_p2)
        p3_sample = np.array(seq_p3)
        
        # 组合成 (N, 4)
        return np.column_stack([x_sample, y_sample, p2_sample, p3_sample])

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
            # 加载原始数据
            raw = np.load(path, encoding='latin1', allow_pickle=True)['train']
            purified = purify(raw)
            all_raw_data.extend(purified)
            data_by_cat[cat] = purified
    
    # Global Scale Factor (只使用前两列 delta_x, delta_y 计算)
    flat_data = []
    for seq in all_raw_data:
        flat_data.extend(seq[:, :2].flatten())
    scale_factor = np.std(flat_data)
    print(f"Global Scale Factor: {scale_factor:.4f}")

    # Normalize data (只归一化前两列)
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
    new_seq[:len_seq-1, 2] = 1 - sequence[:-1, 2] # p1: pen touching
    new_seq[:len_seq, 3] = sequence[:, 2]         # p2: pen lifted
    new_seq[(len_seq-1):, 4] = 1                  # p3: end of sequence (forced)
    new_seq[len_seq-1, 2:4] = 0
    return torch.from_numpy(new_seq).float().unsqueeze(1).to(device)

def save_plot(sequence, filename, title, color='white'):
    """
    输入: [N, 4] 或 [N, 3] 绝对坐标
    修改：黑底白线，去标题
    """
    abs_x = sequence[:, 0]
    abs_y = sequence[:, 1]
    pen_states = sequence[:, 2] # 使用第3列作为断笔标志

    split_indices = np.where(pen_states > 0)[0] + 1
    strokes = np.split(np.stack([abs_x, abs_y], axis=1), split_indices)
    
    # 1. 设置黑色背景的 Figure
    fig = plt.figure(figsize=(5,5), facecolor='black')
    ax = fig.add_subplot(111)
    
    for s in strokes:
        if len(s) > 0:
            # 2. 线条颜色由参数 color 控制 (默认 white)
            plt.plot(s[:, 0], -s[:, 1], color=color, linewidth=2)
    
    # 3. 删除标题
    # plt.title(title) <--- 已注释/删除

    plt.axis('equal')
    plt.axis('off')
    
    save_path = os.path.join(hp.output_dir, filename)
    # 4. 保存时确保背景色为黑色
    plt.savefig(save_path, facecolor='black') 
    plt.close()
    print(f"Image Saved: {save_path}")

def save_npy(sequence, filename):
    """
    保存为 npy 文件
    """
    save_path = os.path.join(hp.output_dir, filename)
    np.save(save_path, sequence)
    print(f"Data Saved ({sequence.shape}): {save_path}")

################################# Main Reconstruction Logic
def reconstruct_sample(cat_name, idx):
    if not hasattr(reconstruct_sample, "data_cache"):
         reconstruct_sample.data_cache = prepare_data_dict()
    data_dict, scale_factor = reconstruct_sample.data_cache
    
    if cat_name not in data_dict:
        print(f"Error: Category {cat_name} not found.")
        return
    
    if idx >= len(data_dict[cat_name]):
        print(f"Error: Index {idx} out of range.")
        return

    print(f"\nProcessing: {cat_name} (ID: {idx})")

    # 1. 获取原始数据 (归一化的)
    seq_norm = data_dict[cat_name][idx] 
    
    # 2. 模型推理 (Encode -> Decode)
    input_tensor = make_tensor_input(seq_norm)
    model = ReconstructionModel()
    if not model.load(hp.load_epoch):
        return

    with torch.no_grad():
        z, _ = model.encoder(input_tensor, 1)
        # 生成重建数据 (N, 4) [x, y, p_lift, p_eos]
        recon_absolute = model.generate_from_latent(z, scale_factor)

    # 3. 处理原始数据用于保存 (Norm Delta -> Real Absolute)
    def denormalize_to_absolute(norm_seq, scale):
        real_seq = norm_seq.copy()
        real_seq[:, :2] *= scale # 反归一化
        
        abs_x = np.cumsum(real_seq[:, 0])
        abs_y = np.cumsum(real_seq[:, 1])
        
        if norm_seq.shape[1] >= 4:
            return np.column_stack([abs_x, abs_y, real_seq[:, 2], real_seq[:, 3]])
        else:
            p_eos = np.zeros(len(abs_x))
            p_eos[-1] = 1 
            return np.column_stack([abs_x, abs_y, real_seq[:, 2], p_eos])

    orig_absolute = denormalize_to_absolute(seq_norm, scale_factor)

    # 4. 保存文件
    base_name = f"{cat_name}_id{idx}"

    save_npy(orig_absolute, f"original_{base_name}.npy")
    save_npy(recon_absolute, f"recon_{base_name}.npy")
    
    # 这里 color='white' 配合 save_plot 中的黑底，实现黑底白线
    save_plot(orig_absolute, 
              f"original_{base_name}.png", 
              f"Original: {cat_name} (ID:{idx})", 
              color='white')
              
    save_plot(recon_absolute, 
              f"recon_{base_name}.png", 
              f"Reconstruction: {cat_name} (ID:{idx})", 
              color='white')

    print("--- Done ---")

if __name__ == '__main__':
    target_category = 'apple'
    target_id = 11913
    
    reconstruct_sample(target_category, target_id)