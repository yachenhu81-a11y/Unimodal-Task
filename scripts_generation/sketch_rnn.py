import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################### Hyperparameters & Path Config
class HParams():
    def __init__(self):
        # 获取当前脚本文件(sketch_rnn.py)所在的目录
        # 例如: E:\code\...\Unimodal-Task\scripts_generation
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 获取项目根目录 (脚本目录的上一级)
        # 例如: E:\code\...\Unimodal-Task
        self.project_root = os.path.dirname(self.script_dir)
        
        # 1. 数据路径配置: ../data/QuickDraw_generation
        self.data_dir = os.path.join(self.project_root, 'data', 'QuickDraw_generation')
        
        # 2. 输出路径配置: ./test_sketch (与脚本同级)
        self.output_dir = os.path.join(self.script_dir, 'test_sketch')
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Data Dir: {self.data_dir}")
        print(f"Output Dir: {self.output_dir}")

        # 想要训练的类别列表 (请确保这8个文件都在data目录下)
        self.categories = ['cat', 'apple', 'bus', 'angel', 'clock', 'pig', 'sheep', 'umbrella'] 
        
        # 模型参数
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200

hp = HParams()

################################# Data Loading and Preprocessing

def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes, scale_factor=None):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    if scale_factor is None:
        scale_factor = calculate_normalizing_scale_factor(strokes)
    
    normalized_data = []
    for seq in strokes:
        n_seq = seq.copy()
        n_seq[:, 0:2] /= scale_factor
        normalized_data.append(n_seq)
    return normalized_data, scale_factor

def load_dataset():
    """Load and mix data from all categories specified in hp.categories"""
    all_strokes = []
    all_labels = []

    print(f"Loading data from {hp.data_dir}...")
    
    for cat_name in hp.categories:
        file_path = os.path.join(hp.data_dir, f'{cat_name}.npz')
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}, skipping.")
            continue
            
        try:
            dataset = np.load(file_path, encoding='latin1', allow_pickle=True)
            # 这里我们只取 'train' 部分
            raw_data = dataset['train']
            
            # Step 1: Purify (filter bad sequences)
            purified_data = purify(raw_data)
            
            # Add to list
            all_strokes.extend(purified_data)
            # Extend labels list with the category name
            all_labels.extend([cat_name] * len(purified_data))
            
            print(f"Loaded {cat_name}: {len(purified_data)} samples.")
            
        except Exception as e:
            print(f"Error loading {cat_name}: {e}")

    if not all_strokes:
        raise ValueError("No data loaded! Please check your data path and category names.")

    # Step 2: Global Normalize
    # Calculate one scale factor for ALL data to maintain consistency
    print("Calculating global normalization factor...")
    norm_data, scale_factor = normalize(all_strokes)
    
    print(f"Total dataset size: {len(norm_data)}. Scale factor: {scale_factor:.4f}")
    
    return norm_data, all_labels

# Load Data Global Variables
train_data, train_labels = load_dataset()
Nmax = max_size(train_data)

############################## Batch Generation
def make_batch(batch_size):
    """
    Randomly sample a batch from the mixed dataset.
    Returns:
        batch: tensor (max_len, batch_size, 5)
        lengths: list of sequence lengths
        indices: indices of the sampled data (used for tracking labels during inference)
    """
    batch_idx = np.random.choice(len(train_data), batch_size)
    batch_sequences = [train_data[idx] for idx in batch_idx]
    
    strokes = []
    lengths = []
    
    for seq in batch_sequences:
        len_seq = len(seq[:,0])
        new_seq = np.zeros((Nmax, 5))
        new_seq[:len_seq, :2] = seq[:, :2]
        new_seq[:len_seq-1, 2] = 1 - seq[:-1, 2]
        new_seq[:len_seq, 3] = seq[:, 2]
        new_seq[(len_seq-1):, 4] = 1
        new_seq[len_seq-1, 2:4] = 0
        lengths.append(len_seq)
        strokes.append(new_seq)

    batch = torch.from_numpy(np.stack(strokes, 1)).float().to(device)
    return batch, lengths, batch_idx

################################ Adaptive LR
def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

################################# Encoder and Decoder Modules
class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, dropout=hp.dropout, bidirectional=True)
        self.fc_mu = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.train()

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
        sigma = torch.exp(sigma_hat/2.)
        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).to(device)
        z = mu + sigma*N
        return z, mu, sigma_hat

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
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])
        params_pen = params[-1]
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        
        if self.training:
            len_out = Nmax + 1
        else:
            len_out = 1
                                    
        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out, -1, hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out, -1, hp.M)
        q = F.softmax(params_pen).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell

################################ Model and Training Loop
class Model():
    def __init__(self):
        self.encoder = EncoderRNN().to(device)
        self.decoder = DecoderRNN().to(device)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

    def make_target(self, batch, lengths):
        eos = torch.stack([torch.tensor([0,0,0,0,1])]*batch.size()[1]).to(device).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(Nmax+1, batch.size()[1])
        for indice, length in enumerate(lengths):
            mask[:length, indice] = 1
        mask = mask.to(device)
        dx = torch.stack([batch.data[:,:,0]]*hp.M, 2)
        dy = torch.stack([batch.data[:,:,1]]*hp.M, 2)
        p1 = batch.data[:,:,2]
        p2 = batch.data[:,:,3]
        p3 = batch.data[:,:,4]
        p = torch.stack([p1, p2, p3], 2)
        return mask, dx, dy, p

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        
        batch, lengths, _ = make_batch(hp.batch_size)
        
        # encode:
        z, self.mu, self.sigma = self.encoder(batch, hp.batch_size)
        # create start of sequence:
        sos = torch.stack([torch.tensor([0,0,1,0,0])]*hp.batch_size).to(device).unsqueeze(0)
        batch_init = torch.cat([sos, batch], 0)
        z_stack = torch.stack([z]*(Nmax+1))
        inputs = torch.cat([batch_init, z_stack], 2)
        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        # prepare targets:
        mask, dx, dy, p = self.make_target(batch, lengths)
        
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.eta_step = 1 - (1-hp.eta_min)*hp.R
        
        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask, dx, dy, p, epoch)
        loss = LR + LKL
        
        loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.grad_clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if epoch % 100 == 0:
            print('epoch', epoch, 'loss', loss.item(), 'LR', LR.item(), 'LKL', LKL.item())
            self.encoder_optimizer = lr_decay(self.encoder_optimizer)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer)
        
        if epoch % 500 == 0:
            self.conditional_generation(epoch)
            
        if epoch % 15000 == 0 and epoch > 0:
            self.save(epoch)

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y - 2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp/norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask*torch.log(1e-5+torch.sum(self.pi * pdf, 2)))\
            /float(Nmax*hp.batch_size)
        LP = -torch.sum(p*torch.log(self.q))/float(Nmax*hp.batch_size)
        return LS + LP

    def kullback_leibler_loss(self):
        LKL = -0.5*torch.sum(1+self.sigma-self.mu**2-torch.exp(self.sigma))\
            /float(hp.Nz*hp.batch_size)
        KL_min = torch.tensor([hp.KL_min], device=device)
        return hp.wKL*self.eta_step * torch.max(LKL, KL_min)

    def save(self, epoch):
        # 保存到指定目录
        enc_path = os.path.join(hp.output_dir, f'encoder_epoch_{epoch}.pth')
        dec_path = os.path.join(hp.output_dir, f'decoder_epoch_{epoch}.pth')
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.decoder.state_dict(), dec_path)
        print(f"Model saved to {hp.output_dir}")

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation(self, epoch):
        """
        Modified to perform inference on a random sample from the dataset.
        Saves both the original (Ground Truth) and the Reconstructed image.
        """
        # 1. Randomly sample 1 real data point
        batch, lengths, idx_list = make_batch(1)
        idx = idx_list[0]
        label_name = train_labels[idx]
        
        print(f"\n[Inference] Epoch {epoch}: Processing a sketch of category: '{label_name}'")

        self.encoder.train(False)
        self.decoder.train(False)
        
        # 2. Encode
        z, _, _ = self.encoder(batch, 1)
        
        # 3. Decode
        sos = torch.tensor([0,0,1,0,0], device=device).view(1,1,-1)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        
        for i in range(Nmax):
            input = torch.cat([s, z.unsqueeze(0)], 2)
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            s, dx, dy, pen_down, eos = self.sample_next_state()
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                break
        
        # 4. Process Generated Output
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        generated_sequence = np.stack([x_sample, y_sample, z_sample]).T
        
        # 5. Process Original Input for Visualization
        original_seq_raw = batch.squeeze(1).cpu().numpy()
        orig_x = np.cumsum(original_seq_raw[:, 0], 0)
        orig_y = np.cumsum(original_seq_raw[:, 1], 0)
        orig_z = original_seq_raw[:, 3] # p2 column (pen lifted)
        original_sequence = np.stack([orig_x, orig_y, orig_z]).T

        # 6. Save Images
        make_image(original_sequence, epoch, name=f'_{label_name}_original', color='green')
        make_image(generated_sequence, epoch, name=f'_{label_name}_generated', color='blue')
        
        self.encoder.train(True)
        self.decoder.train(True)

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
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        return next_state.to(device).view(1,1,-1), x, y, q_idx==1, q_idx==2

def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False):
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def make_image(sequence, epoch, name='_output_', color='blue'):
    """
    Plot drawing with separated strokes
    sequence: [x_abs, y_abs, pen_lifted_boolean]
    """
    split_indices = np.where(sequence[:, 2] > 0)[0] + 1
    strokes = np.split(sequence, split_indices)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for s in strokes:
        if len(s) > 0:
            plt.plot(s[:, 0], -s[:, 1], color=color)
            
    plt.axis('equal')
    plt.axis('off')

    fig.canvas.draw()
    
    try:
        buffer = fig.canvas.tostring_rgb()
    except AttributeError:
        buffer = fig.canvas.buffer_rgba()
        
    width, height = fig.canvas.get_width_height()
    mode = 'RGBA' if len(buffer) == width * height * 4 else 'RGB'
    pil_image = PIL.Image.frombytes(mode, (width, height), buffer)
    
    if mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    
    save_path = os.path.join(hp.output_dir, f"{epoch}{name}.png")
    pil_image.save(save_path, "PNG")
    plt.close("all")

################################ Main Execution
if __name__=="__main__":
    model = Model()
    
    print("Start Training...")
    # 训练循环
    # for epoch in range(15001):
    #     model.train(epoch)
    
    # 训练结束后，加载保存的模型进行一次最终测试
    try:
        print("Loading final model for demonstration...")
        final_epoch = 15000
        enc_path = os.path.join(hp.output_dir, f'encoder_epoch_{final_epoch}.pth')
        dec_path = os.path.join(hp.output_dir, f'decoder_epoch_{final_epoch}.pth')
        
        if os.path.exists(enc_path) and os.path.exists(dec_path):
            model.load(enc_path, dec_path)
            model.conditional_generation(final_epoch + 1)
        else:
            print("Checkpoint not found, skipping final loading test.")
            
    except Exception as e:
        print(f"Error in loading/inference: {e}")