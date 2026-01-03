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
        
        # 输出目录：interpolation
        self.output_dir = os.path.join(self.script_dir, 'interpolation')
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

################################# Model Definition
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

class InterpolationModel():
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
        生成序列并乘以 scale_factor 进行反归一化，
        确保生成的图片与 inspect_samples.py 中的原图尺度一致。
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
                
                # --- 关键修改：反归一化 ---
                dx_real = dx * scale_factor
                dy_real = dy * scale_factor
                
                seq_x.append(dx_real)
                seq_y.append(dy_real)
                seq_z.append(pen_down)
                
                if eos: break
        
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
    """
    加载数据并计算全局缩放因子 (Scale Factor)。
    返回:
        norm_data_by_cat: 归一化后的数据字典 (用于输入模型)
        scale_factor:     全局缩放因子 (用于反归一化绘图)
    """
    all_raw_data = []
    data_by_cat = {}

    print("Loading datasets to calculate normalization...")
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

    # Normalize data in the dictionary
    norm_data_by_cat = {}
    for cat, sequences in data_by_cat.items():
        norm_seqs = []
        for seq in sequences:
            n_seq = seq.copy()
            n_seq[:, :2] /= scale_factor # 归一化
            norm_seqs.append(n_seq)
        norm_data_by_cat[cat] = norm_seqs
        
    return norm_data_by_cat, scale_factor # 修改：返回 scale_factor

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
    绘制并保存图片。
    sequence: [N, 3] 数组，包含绝对坐标(x,y)或偏移量(dx,dy)？
    注意：这里传入的 sequence 应该是已经转换成绝对坐标 (cumsum) 后的数据，
    或者是原始数据列表。
    inspect_samples 逻辑：先 cumsum 变绝对坐标，再 split。
    """
    
    # 如果传入的是未 cumsum 的数据 (dx, dy)，这里需要额外判断吗？
    # generate_from_latent 返回的是已经 cumsum 过的。
    # 原始数据我们也会在 main 里处理成绝对坐标格式传入，或者在这里统一处理？
    # 为了兼容 inspect_samples 的逻辑，我们假设传入的是绝对坐标 (x, y, pen)。
    # 但 wait，inspect_samples 里是先 split 再 plot。
    
    # 我们统一约定：传入的 sequence 是 [N, 3] (x_abs, y_abs, pen_state)
    abs_x = sequence[:, 0]
    abs_y = sequence[:, 1]
    pen_states = sequence[:, 2]

    split_indices = np.where(pen_states > 0)[0] + 1
    strokes = np.split(np.stack([abs_x, abs_y], axis=1), split_indices)
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    for s in strokes:
        if len(s) > 0:
            # 加上负号，符合屏幕坐标系
            plt.plot(s[:, 0], -s[:, 1], color=color, linewidth=2)
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    
    save_path = os.path.join(hp.output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

################################# Main Interpolation Logic
def main(cat1_name, idx1, cat2_name, idx2, alpha=0.5):
    # 1. Load Data
    data_dict, scale_factor = prepare_data_dict() # 获取 scale_factor
    
    if cat1_name not in data_dict or cat2_name not in data_dict:
        print(f"Error: Categories {cat1_name} or {cat2_name} not in dataset.")
        return
    
    # 验证 ID 是否越界
    if idx1 >= len(data_dict[cat1_name]) or idx2 >= len(data_dict[cat2_name]):
        print("Error: Index out of range.")
        return

    # 2. Get specific samples by ID
    print(f"Selecting {cat1_name} ID: {idx1}")
    print(f"Selecting {cat2_name} ID: {idx2}")
    
    seq1_norm = data_dict[cat1_name][idx1] # 归一化后的数据 (用于模型输入)
    seq2_norm = data_dict[cat2_name][idx2]
    
    input1 = make_tensor_input(seq1_norm)
    input2 = make_tensor_input(seq2_norm)

    # 3. Load Model
    model = InterpolationModel()
    if not model.load(hp.load_epoch):
        return

    # 4. Encode
    with torch.no_grad():
        z1, _ = model.encoder(input1, 1) 
        z2, _ = model.encoder(input2, 1)
    
    # 5. Interpolate
    z_interp = z1 * alpha + z2 * (1 - alpha)

    # 6. Generate (Decode) & Denormalize
    # 传入 scale_factor，生成绝对坐标的大尺寸图片
    recon1 = model.generate_from_latent(z1, scale_factor)
    recon2 = model.generate_from_latent(z2, scale_factor)
    recon_interp = model.generate_from_latent(z_interp, scale_factor)

    # 7. Prepare Original Data for Plotting (Denormalize)
    # 原始数据在 data_dict 里是归一化的偏移量 (dx, dy)。
    # 我们需要：1. 乘回 scale_factor (变回真实偏移量) 2. cumsum (变回绝对坐标)
    def prepare_original_for_plot(norm_seq, scale):
        real_seq = norm_seq.copy()
        real_seq[:, :2] *= scale # 反归一化
        
        abs_x = np.cumsum(real_seq[:, 0])
        abs_y = np.cumsum(real_seq[:, 1])
        return np.stack([abs_x, abs_y, real_seq[:, 2]], axis=1)

    orig1_plot = prepare_original_for_plot(seq1_norm, scale_factor)
    orig2_plot = prepare_original_for_plot(seq2_norm, scale_factor)

    # 8. Visualization & Save
    print(f"\n--- Interpolating between {cat1_name} (ID:{idx1}) and {cat2_name} (ID:{idx2}) ---")
    
    # 保存5张图：原图A -> 重构A -> 插值 -> 重构B -> 原图B
    
    # 1. 原始图 Source A
    save_plot(orig1_plot, 
              f'0_original_{cat1_name}_id{idx1}.png', 
              f'Original: {cat1_name} (ID:{idx1})', 
              color='black')
              
    # 2. 重构图 Source A (AutoEncoder output)
    save_plot(recon1, 
              f'1_recon_{cat1_name}_id{idx1}.png', 
              f'Reconstruction: {cat1_name}', 
              color='green')
              
    # 3. 插值图 Interpolation
    save_plot(recon_interp, 
              f'2_interp_{cat1_name}_{cat2_name}.png', 
              f'Interpolation ({alpha*100}%)', 
              color='red')
              
    # 4. 重构图 Source B
    save_plot(recon2, 
              f'3_recon_{cat2_name}_id{idx2}.png', 
              f'Reconstruction: {cat2_name}', 
              color='green')

    # 5. 原始图 Source B
    save_plot(orig2_plot, 
              f'4_original_{cat2_name}_id{idx2}.png', 
              f'Original: {cat2_name} (ID:{idx2})', 
              color='black')

if __name__ == '__main__':
    # ================= 使用说明 =================
    # 1. 先运行 inspect_samples.py，记下你觉得好看的 ID。
    # 2. 将 ID 填入下方函数中。
    # ===========================================
    
    cat_id = 11913    
    clock_id = 7182   
    
    # 使用 alpha=0.5 生成中间状态
    main('apple', cat_id, 'clock', clock_id, alpha=0.4)