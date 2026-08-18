"""
Transformer 结构连接（SC）与功能连接（FC）动态分析
======================================================

数据集准备（二选一）：
  方案A（自动）: pip install datasets transformers
                 首次运行会自动从HuggingFace下载WikiText-2

  方案B（手动）: 从 https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
                 下载后解压到 ./data/wikitext-2/
                 目录结构: ./data/wikitext-2/wiki.train.tokens
                                             wiki.valid.tokens
                                             wiki.test.tokens

切换结构连接方案：
  SC_METHOD = 'linear'    → 方案一：线性骨架分解
  SC_METHOD = 'jacobian'  → 方案二：期望雅可比矩阵

作者: 自动生成
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import copy
import os
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
SC_METHOD = 'jacobian'      # 'linear' 或 'jacobian'
FC_METHOD = 'concat'        # 'mean' (序列平均) 或 'concat' (Token级特征拼接)

# 模型超参数
D_MODEL    = 128           # 神经元数量（每层维度）
N_HEADS    = 4
D_FF       = 256           # FFN中间层维度
N_LAYERS   = 2
SEQ_LEN    = 32
VOCAB_SIZE = 5000
DROPOUT    = 0.1

# 训练超参数
EPOCHS         = 3
BATCH_SIZE     = 64
LR             = 1e-4
STEP_INTERVAL  = 20        # 每隔多少个batch采样一次

# 分析超参数
SAMPLES_FOR_FC = 100       # 计算FC时使用的样本数
JACOBIAN_SAMPLES = 100     # 计算Jacobian时使用的样本数
ANALYZE_LAYER  = 0         # 分析第几个Transformer层（0-indexed）


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 1. 数据加载（WikiText-2 语言模型任务）
# ============================================================

def load_data_huggingface(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """通过HuggingFace datasets自动下载WikiText-2"""
    try:
        from datasets import load_dataset
        print("正在从HuggingFace下载WikiText-2...")
        dataset = load_dataset("wikitext", "wikitext-2-v1")
        train_text = " ".join(dataset['train']['text'])
        val_text   = " ".join(dataset['validation']['text'])
        return train_text, val_text
    except Exception as e:
        print(f"HuggingFace下载失败: {e}")
        return None, None

def load_data_local(data_dir='/mnt/Data16T/Data/haichao/code/AI_connectom/story/story_part2_struc_func/Transformer/data/wikitext-2'):
    """从本地读取WikiText-2"""
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    val_path   = os.path.join(data_dir, 'wiki.valid.tokens')
    if not os.path.exists(train_path):
        return None, None
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(val_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    print(f"从本地加载WikiText-2: {data_dir}")
    return train_text, val_text

def build_vocab(text, vocab_size):
    """构建词表（取最高频的vocab_size个词）"""
    from collections import Counter
    words = text.split()
    counter = Counter(words)
    most_common = [w for w, _ in counter.most_common(vocab_size - 2)]
    vocab = {'<pad>': 0, '<unk>': 1}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab

class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_len):
        words = text.split()
        ids = [vocab.get(w, 1) for w in words]  # 1 = <unk>
        # 截断到seq_len的整数倍
        total = (len(ids) // seq_len) * seq_len
        ids = ids[:total]
        self.data = torch.tensor(ids, dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return len(self.data) - 1  # 留一个作为target

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data[idx + 1] if idx + 1 < len(self.data) else self.data[idx]
        return x, y

def get_dataloaders(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    # 尝试HuggingFace，失败则尝试本地
    train_text, val_text = load_data_local()
    if train_text is None:
        train_text, val_text = load_data_huggingface(vocab_size, seq_len)
    if train_text is None:
        raise RuntimeError(
            "无法加载数据集！\n"
            "请手动下载: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip\n"
            "解压到 ./data/wikitext-2/"
        )

    vocab = build_vocab(train_text, vocab_size)
    train_ds = TextDataset(train_text, vocab, seq_len)
    val_ds   = TextDataset(val_text,   vocab, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"词表大小: {len(vocab)} | 训练样本: {len(train_ds)} | 验证样本: {len(val_ds)}")
    return train_loader, val_loader, vocab


# ============================================================
# 2. Transformer 模型（带钩子，便于提取中间层输出）
# ============================================================

class TransformerEncoderLayerCustom(nn.Module):
    """
    自定义TransformerEncoder层，暴露内部权重，支持提取层输入/输出
    结构: x → LayerNorm → MultiheadAttn → Residual → LayerNorm → FFN → Residual
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff    = d_ff

        # 注意力层（合并的QKV投影）
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # FFN
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, src_key_padding_mask=None):
        # 自注意力子层
        x2, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(x2)
        # FFN子层
        x2 = self.ff2(self.dropout(self.act(self.ff1(self.norm2(x)))))
        x = x + self.dropout(x2)
        return x

    def get_W_OV(self):
        """
        计算 W_OV = W_V @ W_O
        MultiheadAttention内部权重: in_proj_weight [3*d_model, d_model]
        out_proj.weight [d_model, d_model]
        W_V 是 in_proj_weight[2*d_model:, :]
        """
        with torch.no_grad():
            in_proj = self.self_attn.in_proj_weight  # [3*d, d]
            W_V = in_proj[2 * self.d_model:, :]       # [d, d]
            W_O = self.self_attn.out_proj.weight       # [d, d]
            # W_OV: 输入经W_V映射后经W_O映射
            W_OV = W_V.T @ W_O.T  # [d, d]
        return W_OV  # [d_model, d_model]

    def get_W_FFN(self):
        """计算 W_FFN = W1 @ W2: [d_model, d_model]"""
        with torch.no_grad():
            W1 = self.ff1.weight  # [d_ff, d_model]
            W2 = self.ff2.weight  # [d_model, d_ff]
            W_FFN = W1.T @ W2.T   # [d_model, d_model]
        return W_FFN


class TransformerLM(nn.Module):
    """用于语言模型的Transformer（自回归预测下一个token）"""
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, seq_len, dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = nn.Embedding(seq_len, d_model)
        self.layers    = nn.ModuleList([
            TransformerEncoderLayerCustom(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.fc_out  = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def _init_weights(self):
    #     for m in self.modules():
    #         # 仅针对线性层进行指定的正态分布初始化
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, mean=0, std=0.0001)
    #             # 如果线性层包含偏置项，通常将其初始化为0
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         # 保留对词表 Embedding 的默认 Xavier 初始化（避免破坏输入层的稳定性）
    #         elif isinstance(m, nn.Embedding):
    #             nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.embedding(x) + self.pos_enc(positions))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.fc_out(h)

    def get_layer_outputs(self, x):
        """
        返回每层的输入和输出（用于提取神经元激活和计算Jacobian）
        Returns: list of (layer_input, layer_output) tensors
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_enc(positions)
        layer_io = []
        for layer in self.layers:
            h_in = h
            h    = layer(h)
            layer_io.append((h_in, h))
        return layer_io


# ============================================================
# 3. 功能连接（FC）计算
# ============================================================
def get_neuron_activations(model, data_loader, layer_idx, n_samples, device, fc_method='mean'):
    """
    获取神经元激活值用于计算FC
    - 'mean': 对每个样本的Token序列取平均，返回形状 [n_samples, d_model]
    - 'concat': 拼接所有样本的Token特征，返回形状 [n_samples * seq_len, d_model]
    """
    model.eval()
    all_acts = []
    collected = 0

    with torch.no_grad():
        for x, _ in data_loader:
            if collected >= n_samples:
                break
                
            # 严格限制收集的样本数（防止最后一个Batch超出n_samples）
            current_batch_size = min(x.shape[0], n_samples - collected)
            x = x[:current_batch_size].to(device)
            
            layer_io = model.get_layer_outputs(x)
            h_out = layer_io[layer_idx][1]     # [B, T, d_model]
            
            if fc_method == 'mean':
                # 在序列维度取平均 -> [B, d_model]
                neuron_out = h_out.mean(dim=1)
            elif fc_method == 'concat':
                # 将 B 和 T 维度展平融合 -> [B*T, d_model]
                neuron_out = h_out.reshape(-1, h_out.shape[-1])
            else:
                raise ValueError(f"不支持的FC计算方法: {fc_method}")
                
            all_acts.append(neuron_out)
            collected += current_batch_size

    # 将所有Batch的结果在第0维拼接
    all_acts = torch.cat(all_acts, dim=0)
    return all_acts  # 'mean': [N, d], 'concat': [N*T, d]

def compute_fc(activations):
    """纯 GPU 计算 Pearson 相关功能连接矩阵"""
    # 假设 activations 已经是 GPU tensor: [n_samples, d_model]
    FC = torch.corrcoef(activations.T)  # [d_model, d_model]
    # ranks = torch.argsort(torch.argsort(activations, dim=0), dim=0).float()
    # FC = torch.corrcoef(ranks.T)
    FC = torch.nan_to_num(FC)
    return FC


# ============================================================
# 4. 结构连接（SC）计算 —— 方案一：线性骨架
# ============================================================

def compute_sc_linear(model, layer_idx):
    """
    方案一: 线性骨架分解
    SC = I + W_OV + W_FFN
    其中:
        I      : 残差恒等路径 [d_model, d_model]
        W_OV   : Value×Output投影乘积 [d_model, d_model]
        W_FFN  : FFN两层权重乘积 [d_model, d_model]
    返回 SC 的Pearson相关（归一化）
    """
    layer = model.layers[layer_idx]
    d = model.d_model
    device = next(model.parameters()).device # 获取当前模型所在的设备(GPU)

    I     = torch.eye(d, device=device)
    W_OV  = layer.get_W_OV()
    W_FFN = layer.get_W_FFN()

    # SC_raw = I + W_OV + W_FFN
    SC_raw = W_OV * W_FFN

    # 计算行间Pearson相关（行 = 输入神经元，列 = 输出权重）
    SC = torch.corrcoef(SC_raw)
    SC = torch.nan_to_num(SC)
    return SC  # [d_model, d_model]


# ============================================================
# 5. 结构连接（SC）计算 —— 方案二：期望雅可比矩阵
# ============================================================

from torch.func import jacrev, vmap

def compute_sc_jacobian(model, data_loader, layer_idx, n_samples, device):
    """
    极速版：使用 torch.func.jacrev 和 vmap 向量化计算期望雅可比矩阵
    """
    model.eval()
    jacobian_sum = torch.zeros(model.d_model, model.d_model, device=device)
    count = 0

    # ==========================================
    # 1. 定义纯函数 (Pure Function) 供 torch.func 使用
    # ==========================================
    def single_sample_forward(h_input):
        """
        输入单个样本的隐藏状态 [T, d_model]
        返回经过该层并平均后的输出特征 [d_model]
        """
        # 补充 batch 维 [1, T, d_model] 以适配 Transformer 层
        out = model.layers[layer_idx](h_input.unsqueeze(0)) 
        # 对 T 维度求均值，并去掉 batch 维
        return out.mean(dim=1).squeeze(0)

    # ==========================================
    # 2. 魔法组合：计算雅可比并应用到整个 Batch
    # ==========================================
    # jacrev: 计算单样本的雅可比矩阵 -> [d_model, T, d_model]
    # vmap:   将上述操作并行化到 batch 维 -> [B, d_model, T, d_model]
    batch_jacobian_fn = vmap(jacrev(single_sample_forward))

    for x, _ in data_loader:
        if count >= n_samples:
            break
            
        # 取需要的样本数
        current_batch_size = min(x.shape[0], n_samples - count)
        x = x[:current_batch_size].to(device)
        B, T = x.shape

        # 前向传播直到目标层之前
        positions = torch.arange(T, device=device).unsqueeze(0)
        h = model.embedding(x) + model.pos_enc(positions) # [B, T, d_model]

        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if i == layer_idx:
                    break
                h = layer(h)

        # ==========================================
        # 3. 极速求解：一次性算出当前 Batch 的雅可比！
        # ==========================================
        # h 的形状是 [B, T, d_model]
        # J_batch_raw 输出形状为 [B, d_model_out, T, d_model_in]
        J_batch_raw = batch_jacobian_fn(h)

        # 在序列长度 T 维度（dim=2）求均值，得到代表全句的雅可比 [B, d_model, d_model]
        J_batch = J_batch_raw.mean(dim=2)

        # 将这个 Batch 内的所有雅可比矩阵累加起来
        jacobian_sum += J_batch.sum(dim=0).detach()
        count += current_batch_size

    jacobian_mean = jacobian_sum / max(count, 1)

    # 在 GPU 上计算 Pearson 相关系数
    SC = torch.corrcoef(jacobian_mean)
    SC = torch.nan_to_num(SC)
    
    # print(f"  Jacobian计算完成，使用{count}个样本")
    return SC, jacobian_mean


# ============================================================
# 6. 输入相似性（IS）计算
# ============================================================

# def compute_input_similarity(SC):
#     """
#     基于结构连接矩阵的行向量计算神经元间的输入相似性
#     IS[i,j] = Pearson(SC[i,:], SC[j,:])  （即SC已经是Pearson相关，这里IS=SC本身）
#     本函数直接返回SC上三角的最大值作为标量指标
#     """
#     triu_idx = np.triu_indices(SC.shape[0], k=1)
#     return SC[triu_idx]

def compute_input_similarity(SC):
    d = SC.shape[0]
    # 取对角线以上的元素
    triu_idx = torch.triu_indices(d, d, offset=1)
    return SC[triu_idx[0], triu_idx[1]]


# ============================================================
# 7. 训练函数（保存每个batch的模型状态）
# ============================================================

def train_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses   = []
    model_states = []

    global_step = 0

    print(f"开始训练 | 设备: {device} | Epochs: {epochs}")
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # [B, T, vocab]
            # 语言模型loss：预测下一个token（用y的第一个token作为target简化）
            loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # 每个batch保存模型状态（CPU clone，节省显存）
            # state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            state = {k: v.clone() for k, v in model.state_dict().items()}
            model_states.append(state)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")

            global_step += 1

        # 验证
        model.eval()
        val_loss_total, val_count = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
                val_loss_total += loss.item()
                val_count += 1
        avg_val = val_loss_total / max(val_count, 1)
        val_losses.append(avg_val)
        print(f">> Epoch {epoch+1} 结束 | Val Loss: {avg_val:.4f} | PPL: {math.exp(min(avg_val, 10)):.2f}")

    return train_losses, val_losses, model_states


# ============================================================
# 8. 连接性动态分析（类似MLP版本）
# ============================================================

def analyze_connectivity_evolution(
    model_states, train_loader, val_loader,
    sc_method='linear',
    fc_method='mean',
    layer_idx=0,
    step_interval=10,
    n_fc_samples=SAMPLES_FOR_FC,
    n_jacobian_samples=JACOBIAN_SAMPLES
):
    """
    在训练过程中分析SC（输入相似性IS）和FC的动态演化
    
    Args:
        sc_method: 'linear' 或 'jacobian'
        layer_idx: 分析第几个Transformer层
        step_interval: 采样间隔（每隔多少个batch分析一次）
    
    Returns:
        sampled_steps, is_max_values, fc_max_values, corr_values
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 重建模板模型
    template_model = TransformerLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        d_ff=D_FF, n_layers=N_LAYERS, seq_len=SEQ_LEN, dropout=0.0  # 分析时关闭dropout
    ).to(device)

    sampled_steps  = []
    is_max_values  = []
    fc_max_values  = []
    corr_values    = []

    print(f"\n开始连接性动态分析 | SC方案: {sc_method} | 分析层: {layer_idx}")
    print(f"总共 {len(model_states)} 个状态，每 {step_interval} 步采样一次...")

    for i in tqdm(range(0, len(model_states), step_interval)):
        template_model.load_state_dict(model_states[i])
        template_model.eval()

        # --- 计算结构连接SC ---
        if sc_method == 'linear':
            SC = compute_sc_linear(template_model, layer_idx)
        else:  # jacobian
            SC, jacobian = compute_sc_jacobian(template_model, train_loader, layer_idx,
                                     n_jacobian_samples, device)

        # --- 计算功能连接FC ---
        # activations = get_neuron_activations(template_model, train_loader,
        #                                      layer_idx, n_fc_samples, device)
        activations = get_neuron_activations(template_model, train_loader,
                                             layer_idx, n_fc_samples, device, fc_method)
        FC = compute_fc(activations)

        # 【核心修复 1】：无论前面输出的是什么，这里强行统一转换为 GPU 张量
        if not isinstance(SC, torch.Tensor):
            SC = torch.tensor(SC, dtype=torch.float32)
        if not isinstance(FC, torch.Tensor):
            FC = torch.tensor(FC, dtype=torch.float32)
            
        SC = SC.to(device)
        FC = FC.to(device)

        # 【核心修复 2】：在 GPU 上直接生成索引并提取上三角（废弃 numpy.triu_indices）
        d = D_MODEL
        triu_idx = torch.triu_indices(d, d, offset=1, device=device)
        
        is_triu = SC[triu_idx[0], triu_idx[1]]
        fc_triu = FC[triu_idx[0], triu_idx[1]]

        # --- 记录指标 ---
        sampled_steps.append(i)
        
        # 提取标量回 CPU
        is_max_values.append(is_triu.max().item())
        fc_max_values.append(fc_triu.max().item())

        # 全程在 GPU 上计算标准差和相关系数
        if torch.std(is_triu) > 1e-8 and torch.std(fc_triu) > 1e-8:
            stacked_triu = torch.stack([is_triu, fc_triu])
            c = torch.corrcoef(stacked_triu)[0, 1]
            corr_values.append(torch.nan_to_num(c).item())
        else:
            corr_values.append(0.0)

    return sampled_steps, is_max_values, fc_max_values, corr_values


# ============================================================
# 9. 可视化
# ============================================================

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import torch

# ============================================================
# 1. 提取 Transformer 全量矩阵历史状态
# ============================================================

def extract_full_connectivity_history_transformer(
    model_states, data_loader, 
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, 
    d_ff=D_FF, n_layers=N_LAYERS, seq_len=SEQ_LEN,
    sc_method='linear', step_interval=1, layer_idx=0,
    n_fc_samples=200, n_jacobian_samples=100
):
    """
    遍历 Transformer 模型快照，提取指定层所有神经元对的 FC 和 SC 演化历史。
    返回展平后的上三角矩阵列表（仅包含不重复的神经元对）。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建模板模型
    template = TransformerLM(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        d_ff=d_ff, n_layers=n_layers, seq_len=seq_len, dropout=0.0
    ).to(device)

    sampled_steps = []
    fc_history = []
    sc_history = []

    print(f"提取全矩阵历史状态 (共 {len(model_states)} 个快照, 间隔 {step_interval}, 方法 {sc_method})...")

    for i in tqdm(range(0, len(model_states), step_interval)):
        template.load_state_dict(model_states[i])
        template.eval()

        with torch.no_grad():
            # 1. 计算 SC 矩阵
            if sc_method == 'linear':
                SC_matrix = compute_sc_linear(template, layer_idx)
            else:
                SC_matrix, jacobian_mean = compute_sc_jacobian(template, data_loader, layer_idx, n_jacobian_samples, device)
            
            # 2. 计算 FC 矩阵
            activations = get_neuron_activations(template, data_loader, layer_idx, n_fc_samples, device)
            FC_matrix = compute_fc(activations)

            # 转换为 NumPy 数组
            if isinstance(SC_matrix, torch.Tensor):
                SC_matrix = SC_matrix.cpu().numpy()
            if isinstance(FC_matrix, torch.Tensor):
                FC_matrix = FC_matrix.cpu().numpy()

            # 提取上三角元素（不含对角线）
            d = d_model
            triu_indices = np.triu_indices(d, k=1)
            
            sc_vec = SC_matrix[triu_indices]
            fc_vec = FC_matrix[triu_indices]

            sampled_steps.append(i) # 记录真实的训练步数
            fc_history.append(fc_vec)
            sc_history.append(sc_vec)

    return np.array(sampled_steps), np.array(fc_history), np.array(sc_history)


# ============================================================
# 2. 微观神经元对追踪分析绘图
# ============================================================

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as ticker

# ============================================================
# 3. 终极组合分析绘图 (保留热图 + 增加定量证明)
# ============================================================

def analyze_pair_level_dynamics_transformer(
    sampled_steps, fc_history, sc_history, 
    early_step_idx=2,       
    window_size=16,         
    max_lag=8,              
    display_max_steps=150,  
    top_k_pairs=200         
):
    """
    微观神经元对追踪分析 (终极版: 全景热图 + 因果定量证明)
    """
    # 处理步数索引和容错
    limit_idx = np.searchsorted(sampled_steps, display_max_steps, side='right')
    if limit_idx < window_size + max_lag and limit_idx != len(sampled_steps):
        print(f"[提示] display_max_steps({display_max_steps}) 较小，图C的 TLCC 热图可能右侧截断。")
        limit_idx = max(limit_idx, min(len(sampled_steps), window_size + 2))

    s_steps = sampled_steps[:limit_idx]
    s_fc = fc_history[:limit_idx, :]
    s_sc = sc_history[:limit_idx, :]
    N_steps, N_pairs = s_fc.shape
    
    # 确保 top_k_pairs 不超过实际神经元对数量
    top_k_pairs = min(top_k_pairs, N_pairs)

    # 创建 3x3 布局，调整行高比
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.3, 1.1], wspace=0.4, hspace=0.4)

    # ==========================================
    # 行 1: 宏观与向量级分析 (A, B, C) - 保持原样
    # ==========================================
    
    # 图 A: 分组轨迹对比
    ax1 = fig.add_subplot(gs[0, 0])
    early_fc = s_fc[min(early_step_idx, N_steps-1)]
    sorted_pair_indices = np.argsort(early_fc)
    top_k_sub = max(1, int(N_pairs * 0.1)) 
    bottom_idx = sorted_pair_indices[:top_k_sub]
    top_idx = sorted_pair_indices[-top_k_sub:]

    ax1.plot(s_steps, np.mean(s_sc[:, top_idx], axis=1), color='firebrick', lw=2.5, label='High Early FC pairs')
    ax1.plot(s_steps, np.mean(s_sc[:, bottom_idx], axis=1), color='steelblue', lw=2.5, label='Low Early FC pairs')
    ax1.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Average IS', fontsize=10, fontweight='bold')
    ax1.set_title('A. IS Trajectories by Early FC', fontsize=11, fontweight='bold')
    ax1.legend(loc='center right', fontsize=9)
    ax1.grid(True, alpha=0.2)

    # 图 B: SC 演化热图 (按 FC 分组)
    ax2 = fig.add_subplot(gs[0, 1])
    num_bins = min(10, N_pairs)
    bin_size = max(1, N_pairs // num_bins)
    evolution_matrix = np.zeros((num_bins, N_steps))
    for b in range(num_bins):
        start_idx = b * bin_size
        end_idx = (b + 1) * bin_size if b < num_bins - 1 else N_pairs
        bin_indices = sorted_pair_indices[start_idx:end_idx]
        evolution_matrix[b, :] = np.mean(s_sc[:, bin_indices], axis=1)

    im2 = ax2.pcolormesh(s_steps, np.arange(num_bins), evolution_matrix, cmap='viridis', shading='auto')
    ax2.set_xlabel('Training Steps', fontweight='bold', fontsize=10)
    ax2.set_ylabel('FC Group (Low to High)', fontweight='bold', fontsize=10)
    ax2.set_title('B. IS Evolution Heatmap (Sorted by FC)', fontsize=11, fontweight='bold')
    fig.colorbar(im2, ax=ax2, label='Mean IS')

    # 图 C: 向量级滑动交叉相关 (TLCC)
    ax3 = fig.add_subplot(gs[0, 2])
    half_w = window_size // 2
    tlcc_matrix = []
    center_steps = []
    for i in range(half_w, N_steps - half_w):
        corrs = []
        for lag in range(-max_lag, max_lag + 1):
            target_idx = i + lag
            if 0 <= target_idx < N_steps:
                # 容错：处理全零情况产生的 NaN
                std_fc = np.std(s_fc[i])
                std_sc = np.std(s_sc[target_idx])
                if std_fc > 1e-9 and std_sc > 1e-9:
                    c = np.corrcoef(s_fc[i].flatten(), s_sc[target_idx].flatten())[0, 1]
                else:
                    c = 0
            else:
                c = np.nan
            corrs.append(c if not np.isnan(c) else 0)
        tlcc_matrix.append(corrs)
        center_steps.append(s_steps[i])

    if tlcc_matrix:
        tlcc_matrix = np.array(tlcc_matrix).T
        lags = np.arange(-max_lag, max_lag + 1)
        im3 = ax3.pcolormesh(center_steps, lags, tlcc_matrix, shading='auto', cmap='RdYlBu_r')
        ax3.axhline(0, color='black', lw=1, ls='--')
        ax3.set_xlabel("Training Steps", fontweight='bold')
        ax3.set_ylabel("Time Lag", fontweight='bold')
        ax3.set_title("C. Vectorized Network TLCC\n(Red above 0 = FC leads IS)", fontsize=11, fontweight='bold')
        fig.colorbar(im3, ax=ax3, label="Correlation")

    # ==========================================
    # 核心数据准备 (为下两行服务)
    # ==========================================
    # 提取最活跃的 top_k 神经元对 (基于 SC 增长趋势)
    early_sc_baseline = np.mean(s_sc[:min(early_step_idx + 1, N_steps), :], axis=0)
    sc_growth_trend = np.mean(s_sc[-10:, :], axis=0) - early_sc_baseline
    filtered_active_idx = np.argsort(sc_growth_trend)[-top_k_pairs:]
    
    fc_active = s_fc[:, filtered_active_idx].T 
    sc_active = s_sc[:, filtered_active_idx].T

    # 平滑处理
    smooth_sigma = 2.5  
    fc_smooth = gaussian_filter1d(fc_active, sigma=smooth_sigma, axis=1)
    sc_smooth = gaussian_filter1d(sc_active, sigma=smooth_sigma, axis=1)

    # 独立最小-最大标准化 [0, 1]
    fc_norm = (fc_smooth - fc_smooth.min(axis=1, keepdims=True)) / \
              (fc_smooth.max(axis=1, keepdims=True) - fc_smooth.min(axis=1, keepdims=True) + 1e-8)
    sc_norm = (sc_smooth - sc_smooth.min(axis=1, keepdims=True)) / \
               (sc_smooth.max(axis=1, keepdims=True) - sc_smooth.min(axis=1, keepdims=True) + 1e-8)

    # 计算起飞时刻 (最大梯度点)
    fc_gradient = np.diff(fc_norm, axis=1)
    sc_gradient = np.diff(sc_norm, axis=1)
    fc_rise_idx = np.argmax(fc_gradient, axis=1)
    sc_rise_idx = np.argmax(sc_gradient, axis=1)
    
    # 排序：按 FC 起飞时刻排序，用于热图展示
    sort_order_heatmap = np.argsort(fc_rise_idx)
    fc_sorted_heatmap = fc_norm[sort_order_heatmap]
    sc_sorted_heatmap = sc_norm[sort_order_heatmap]
    grad_peaks_sorted_heatmap = fc_rise_idx[sort_order_heatmap]

    # 转换为真实步数，用于定量分析
    fc_rise_steps = s_steps[fc_rise_idx]
    sc_rise_steps = s_steps[sc_rise_idx]
    time_lags = sc_rise_steps - fc_rise_steps

    # ==========================================
    # 行 2: 原始热图全景 (D, E) - 均分大小布局
    # ==========================================
    
    # 使用 subgridspec 将整整第二行 (gs[1, :]) 均分为 1 行 2 列
    gs_row2 = gs[1, :].subgridspec(1, 2, wspace=0.15)
    
    # 图 D: FC 陡增点对齐热图 (占据均分后的左半边)
    ax4 = fig.add_subplot(gs_row2[0])
    im4 = ax4.pcolormesh(s_steps, np.arange(top_k_pairs), fc_sorted_heatmap, 
                         cmap='RdYlBu_r', shading='auto', vmin=0, vmax=1, rasterized=True)
    ax4.plot(s_steps[grad_peaks_sorted_heatmap], np.arange(top_k_pairs), color='black', lw=1.8, alpha=1, label='FC Rise Ref')
    ax4.set_xlabel("Training Steps", fontweight='bold')
    ax4.set_ylabel("Sorted Neuron Pairs", fontweight='bold')
    ax4.set_title("D. Functional Connectivity", fontsize=12, fontweight='bold')
    
    # 图 E: SC 陡增点对齐热图 (占据均分后的右半边)
    ax5 = fig.add_subplot(gs_row2[1])
    im5 = ax5.pcolormesh(s_steps, np.arange(top_k_pairs), sc_sorted_heatmap, 
                         cmap='RdYlBu_r', shading='auto', vmin=0, vmax=1, rasterized=True)
    ax5.plot(s_steps[grad_peaks_sorted_heatmap], np.arange(top_k_pairs), color='black', lw=1.8, ls='-', alpha=1)
    ax5.set_xlabel("Training Steps", fontweight='bold')
    ax5.set_yticks([]) # 隐藏 y 轴刻度，因为和左图是对齐的
    ax5.set_title("E. Structural Connectivity", fontsize=12, fontweight='bold')

    # 为第二行的热图添加统一的 Colorbar
    cbar_ax_row2 = fig.add_axes([0.91, 0.4, 0.012, 0.25]) # 稍微向左收拢一点点以适应均分布局
    fig.colorbar(im4, cax=cbar_ax_row2, label="Normalized Value")

    # ==========================================
    # 行 3: 因果证明定量分析 (F, G, H) - 新增
    # ==========================================
    
    # 图 F: 定量滞后时间分布直方图
    ax6 = fig.add_subplot(gs[2, 0])
    # bins = np.linspace(-display_max_steps//2, display_max_steps//2, 30)
    # ax6.hist(time_lags, bins=bins, color='mediumpurple', edgecolor='black', alpha=0.7)
    # median_lag = np.median(time_lags)
    # # ax6.axvline(median_lag, color='red', linestyle='dashed', linewidth=2, 
    # #             label=f'Median: +{median_lag:.0f} steps' if median_lag>0 else f'Median: {median_lag:.0f} steps')
    # ax6.axvline(0, color='black', linestyle='-', linewidth=2)
    # ax6.set_xlabel("Time Lag (SC Rise - FC Rise)", fontweight='bold')
    # ax6.set_ylabel("Number of Neuron Pairs", fontweight='bold')
    # ax6.set_title("F. Quantitative Time Lag Distribution", fontsize=11, fontweight='bold')
    # # ax6.legend(loc='upper right', fontsize=9)
    # ax6.grid(True, alpha=0.2)

    # 过滤异常值以便观察核心分布
    p1, p99 = np.percentile(time_lags, [2, 98])
    filtered_delays = time_lags[(time_lags >= p1) & (time_lags <= p99)]

    mean_delay = np.mean(time_lags)
    median_delay = np.median(time_lags)
    lead_ratio = np.sum(time_lags > 0) / len(time_lags) * 100

    # ---------- 等宽且 0 为边界的 bins 设置 ----------
    min_val, max_val = np.min(filtered_delays), np.max(filtered_delays)
    n_bins_target = 40                         # 期望柱子数（近似）

    # 初始宽度，用于决定断点的整数范围
    bin_width = (max_val - min_val) / n_bins_target

    # 取整到包含 0 的等间隔断点：k * bin_width 形式
    k_min = np.floor(min_val / bin_width).astype(int)
    k_max = np.ceil(max_val / bin_width).astype(int)
    bins = np.arange(k_min, k_max + 1) * bin_width

    # 此时 bins 数量 = k_max - k_min，通常接近 n_bins_target
    # 0 一定是其中之一（k=0 在区间内）
    # ------------------------------------------------

    ax6.hist(filtered_delays, bins=bins,
            color='seagreen', alpha=0.7, edgecolor='black', zorder=2)
    ax6.axvline(0, color='black', linestyle='-', lw=2, label='Zero Lag')
    
    
    ax6.set_xlabel('Lag (Steps)', fontweight='bold')
    ax6.set_ylabel('Frequency (Neuron pairs)', fontweight='bold')
    ax6.set_title('Lag distribution of IS relative to FC', fontsize=13, fontweight='bold')


    # 图 G: 起飞步数散点图
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.scatter(fc_rise_steps, sc_rise_steps, alpha=0.3, color='teal', s=15, edgecolors='none')
    max_step_plot = min(display_max_steps, s_steps[-1])
    ax7.plot([0, max_step_plot], [0, max_step_plot], 'k--', alpha=0.7, label='y=x (Synchronous)')
    # 填充半透明区域指示因果方向
    ax7.fill_between([0, max_step_plot], [0, max_step_plot], max_step_plot, color='red', alpha=0.04, label='SC trails FC')
    ax7.set_xlabel("FC Steepest Ascent (Step)", fontweight='bold')
    ax7.set_ylabel("SC Steepest Ascent (Step)", fontweight='bold')
    ax7.set_title("G. Peak Growth Alignment Scatter", fontsize=11, fontweight='bold')
    ax7.set_xlim([0, max_step_plot])
    ax7.set_ylim([0, max_step_plot])
    ax7.legend(loc='lower right', fontsize=8)
    ax7.grid(True, alpha=0.2)

    # 图 H: 事件对齐 (以 FC 起飞为 t=0) 的平均轨迹
    ax8 = fig.add_subplot(gs[2, 2])
    # 设定一个左右观察窗口
    window_pts = min(40, N_steps // 3) 
    aligned_fc = []
    aligned_sc = []
    
    for i in range(top_k_pairs):
        center = fc_rise_idx[i]
        start = max(0, center - window_pts)
        end = min(N_steps, center + window_pts)
        
        fc_seg = fc_norm[i, start:end]
        sc_seg = sc_norm[i, start:end]
        
        # 处理边界，保证所有片段长度一致 (用 np.nan 填充)
        pad_left = window_pts - (center - start)
        pad_right = window_pts - (end - center)
        fc_seg_padded = np.pad(fc_seg, (pad_left, pad_right), constant_values=np.nan)
        sc_seg_padded = np.pad(sc_seg, (pad_left, pad_right), constant_values=np.nan)
        
        aligned_fc.append(fc_seg_padded)
        aligned_sc.append(sc_seg_padded)
        
    # 计算均值曲线
    mean_aligned_fc = np.nanmean(np.array(aligned_fc), axis=0)
    mean_aligned_sc = np.nanmean(np.array(aligned_sc), axis=0)
    
    # 映射回相对真实步数
    step_diff = (s_steps[1] - s_steps[0]) if len(s_steps) > 1 else 1
    rel_x = np.arange(-window_pts, window_pts) * step_diff
    
    ax8.plot(rel_x, mean_aligned_fc, color='#ED7D31', lw=3, label='Mean FC Trajectory')
    ax8.plot(rel_x, mean_aligned_sc, color='#4472C4', lw=3, label='Mean SC Trajectory')
    ax8.axvline(0, color='black', linestyle='--', alpha=0.5, label='FC Peak (t=0)')
    
    ax8.set_xlabel("Relative Steps from FC Rise", fontweight='bold')
    ax8.set_ylabel("Normalized Growth", fontweight='bold')
    ax8.set_title("H. Event-Aligned Average Trajectories", fontsize=11, fontweight='bold')
    ax8.legend(loc='lower right', fontsize=8)
    ax8.grid(True, alpha=0.2)

    # 总体大标题
    # plt.suptitle(f"Transformer Microscopic Causality Analysis: Function Drives Structure Establishment\n(Analyzing Top {top_k_pairs} Active Neuron Pairs in Layer {layer_idx})", 
    #              fontsize=18, fontweight='bold', y=1.02)
    
    # 这里不需要 plt.tight_layout()，gridspec 已经处理好了
    # plt.savefig(f'/mnt/Data16T/Data/haichao/code/Continuous Discrete/story/story_part2_struc_func/Transformer/fig/Transformer_Microscopic_Causality_Analysis.pdf', format = 'pdf', bbox_inches='tight')
    plt.show()


def plot_training_curves(train_losses, val_losses, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, color='#E05252', linewidth=1.5, label='Train Loss')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()

    epochs_x = range(1, len(val_losses) + 1)
    ax2.plot(epochs_x, val_losses, color='#4472C4', linewidth=2, marker='o', label='Val Loss')
    ax2.set_title('Validation Loss per Epoch', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_connectivity_evolution(sampled_steps, is_max_values, fc_max_values,
                                 corr_values, sc_method, layer_idx, save_path=None):
    """绘制IS和FC随训练步数的动态演化（类比MLP版本）"""
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- 子图1: IS 和 FC 的动态演化（主图，类比MLP版本）---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sampled_steps, is_max_values, color='#4472C4', linewidth=2,
             label='IS', zorder=3)
    ax1.plot(sampled_steps, fc_max_values, color='#ED7D31', linewidth=2,
             label='FC', zorder=3)
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Pearson Correlation', fontsize=12, fontweight='bold')
    ax1.set_title(f'FC-IS Dynamic Evolution',
                  fontsize=13, fontweight='bold')
    ax1.legend(prop={'weight': 'bold', 'size': 11}, loc='upper left')
    ax1.grid(True, alpha=0.2)

    # --- 子图2: IS-FC Pearson相关 ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sampled_steps, corr_values, color='#70AD47', linewidth=2,
             label='IS-FC Pearson Corr', zorder=3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax2.set_title('IS-FC Correlation', fontsize=13, fontweight='bold')
    ax2.legend(prop={'weight': 'bold', 'size': 11})
    ax2.grid(True, alpha=0.2)

    # plt.suptitle(f'Transformer Structural & Functional Connectivity Analysis\n'
    #              f'SC Method: {sc_method.capitalize()}  |  Analyzing Layer {layer_idx}',
    #              fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list

def plot_sc_fc_matrix_snapshot(model, train_loader, sc_method, fc_method, layer_idx,
                                n_fc_samples, n_jacobian_samples, device, 
                                save_path=None, n_show=64):
    model.eval()
    
    # 1. 计算 SC 矩阵
    if sc_method == 'linear':
        SC = compute_sc_linear(model, layer_idx)
    else:
        SC, jacobian_mean = compute_sc_jacobian(model, train_loader, layer_idx, n_jacobian_samples, device)

    # 2. 计算 FC 矩阵
    activations = get_neuron_activations(model, train_loader, layer_idx, n_fc_samples, device, fc_method)
    FC = compute_fc(activations)

    # 确保转换为 numpy 数组
    if isinstance(SC, torch.Tensor):
        SC = SC.cpu().detach().numpy()
        jacobian_mean = jacobian_mean.cpu().detach().numpy()
    if isinstance(FC, torch.Tensor):
        FC = FC.cpu().detach().numpy()

    # np.save('/mnt/Data16T/Data/haichao/code/Continuous Discrete/story/story_part2_struc_func/struc_func_matrix_save/TF/FC_128neuron.npy', FC)
    # np.save('/mnt/Data16T/Data/haichao/code/Continuous Discrete/story/story_part2_struc_func/struc_func_matrix_save/TF/Jacobians_128neurons.npy', jacobian_mean)

    # 限制显示大小
    # 假设 D_MODEL 在全局作用域已定义，如果没有，请确保传入或获取正确的维度
    # n_show = min(n_show, SC.shape[0]) 
    n_show = SC.shape[0]
    
    # 截取子矩阵
    SC_sub = SC[:n_show, :n_show]
    FC_sub = FC[:n_show, :n_show]

    # --- 新增：基于 FC 矩阵进行层次聚类 ---
    # 处理可能的 NaN 或 Inf 以防止聚类报错
    FC_clean = np.nan_to_num(FC_sub)
    
    # 使用 ward 方法进行聚类，获取重排后的索引顺序
    Z = linkage(FC_clean, method='ward')
    idx_order = leaves_list(Z)

    # 按照聚类结果对 FC 和 SC 同步进行行和列的重排
    FC_clustered = FC_sub[idx_order, :][:, idx_order]
    SC_clustered = SC_sub[idx_order, :][:, idx_order]

    # 3. 计算 SC 和 FC 的相关性 (使用重排后矩阵的上三角部分，排除对角线)
    triu_indices = np.triu_indices_from(SC_clustered, k=1)
    
    sc_vals = SC_clustered[triu_indices]
    fc_vals = FC_clustered[triu_indices]
    
    # 计算 Pearson 相关系数
    if len(sc_vals) > 2:
        corr_coef, p_value = pearsonr(sc_vals, fc_vals)
        corr_label = f"Pearson r = {corr_coef:.3f}\n(p = {p_value:.2e})"
    else:
        corr_coef = 0
        corr_label = "N/A"

    # 4. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- 子图 1: SC 热力图 (已聚类) ---
    im1 = axes[0].imshow(SC_clustered, cmap='RdBu_r')
    axes[0].set_title(f'Input Similarity (IS)\nMethod: {sc_method}, Layer {layer_idx} (Clustered)',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Clustered Neuron index')
    axes[0].set_ylabel('Clustered Neuron index')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0], label='Weight / Jacobian Value')

    # --- 子图 2: FC 热力图 (已聚类) ---
    im2 = axes[1].imshow(FC_clustered, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f'Functional Connectivity (FC)\nLayer {layer_idx} Output (Clustered)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'Clustered Neuron index')
    axes[1].set_ylabel(f'Clustered Neuron index')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1], label='Pearson r')

    # --- 子图 3: SC-FC 相关性散点图 ---
    axes[2].scatter(sc_vals, fc_vals, alpha=0.5, s=10, color='tab:blue', edgecolors='none')
    
    # 添加拟合线
    if len(sc_vals) > 2 and np.std(sc_vals) > 1e-8:
        z = np.polyfit(sc_vals, fc_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(sc_vals), max(sc_vals), 100)
        axes[2].plot(x_line, p(x_line), color="black", linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        # axes[2].legend(loc='best', fontsize=9)

    axes[2].set_title(f'IS vs FC Correlation\n{corr_label}',
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('IS Values (Upper Triangular)')
    axes[2].set_ylabel('FC Values (Upper Triangular)')
    
    # 设置坐标轴范围一致以便观察
    min_val = min(np.min(sc_vals), np.min(fc_vals))
    max_val = max(np.max(sc_vals), np.max(fc_vals))
    margin = (max_val - min_val) * 0.05
    axes[2].set_xlim([min_val - margin, max_val + margin])
    axes[2].set_ylim([min_val - margin, max_val + margin])
    
    # 添加对角参考线 y=x
    # axes[2].plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k-', alpha=0.2, label='y=x')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


import torch
from torch.func import jacrev, vmap

def compute_cross_layer_matrices(model, data_loader, n_samples, device, fc_method='mean'):
    """
    计算基于 Layer 0 和 Layer 1 (共256个神经元) 的跨层 FC, SC, IS 矩阵。
    
    参数:
        model: 已训练好的 Transformer 模型
        data_loader: 数据集
        n_samples: 采样的样本数
        device: 计算设备
        fc_method: 'mean' 或 'concat'，控制激活值序列维度的处理方式
        
    返回:
        FC, SC, IS (形状均为 [256, 256] 的张量)
    """
    model.eval()
    d = model.d_model  # d = 128
    
    # ==========================================
    # 1. 定义纯函数，构建跨层微分计算图
    # ==========================================
    def func_y0_x0(x0):
        """计算 Layer 0 输出对基准输入的响应"""
        out = model.layers[0](x0.unsqueeze(0))
        return out.mean(dim=1).squeeze(0)  # [d]

    def func_y1_x0(x0):
        """计算 Layer 1 输出对基准输入的响应 (隐式包含完整前馈路径)"""
        out0 = model.layers[0](x0.unsqueeze(0))
        out1 = model.layers[1](out0)
        return out1.mean(dim=1).squeeze(0)  # [d]

    def func_y1_y0(y0):
        """计算 Layer 1 输出对 Layer 0 输出的响应 (相邻物理连接)"""
        out1 = model.layers[1](y0.unsqueeze(0))
        return out1.mean(dim=1).squeeze(0)  # [d]

    # 并行化雅可比矩阵计算
    batch_jac_y0_x0 = vmap(jacrev(func_y0_x0))
    batch_jac_y1_x0 = vmap(jacrev(func_y1_x0))
    batch_jac_y1_y0 = vmap(jacrev(func_y1_y0))

    # 累加器初始化
    sum_jac_y0_x0 = torch.zeros(d, d, device=device)
    sum_jac_y1_x0 = torch.zeros(d, d, device=device)
    sum_jac_y1_y0 = torch.zeros(d, d, device=device)
    
    all_acts_y0 = []
    all_acts_y1 = []
    count = 0
    
    # ==========================================
    # 2. 采样循环：提取激活值并计算雅可比矩阵
    # ==========================================
    for x, _ in data_loader:
        if count >= n_samples:
            break
            
        batch_sz = min(x.shape[0], n_samples - count)
        x = x[:batch_sz].to(device)
        B, T = x.shape
        
        # 计算基准输入 X0 [B, T, d]
        positions = torch.arange(T, device=device).unsqueeze(0)
        x0 = model.embedding(x) + model.pos_enc(positions) 
        
        # 计算各层实际前向输出 Y0, Y1 (用于提取激活值)
        y0 = model.layers[0](x0)
        y1 = model.layers[1](y0)
        
        # 收集激活值 (按需处理 T 维度)
        if fc_method == 'mean':
            all_acts_y0.append(y0.mean(dim=1))  # [B, d]
            all_acts_y1.append(y1.mean(dim=1))  # [B, d]
        elif fc_method == 'concat':
            all_acts_y0.append(y0.reshape(-1, d))  # [B*T, d]
            all_acts_y1.append(y1.reshape(-1, d))  # [B*T, d]
            
        # 计算批次雅可比矩阵并跨序列(dim=2)求均值
        # 返回形状从 [B, d_out, T, d_in] 变为 [B, d_out, d_in]
        J_y0_x0 = batch_jac_y0_x0(x0).mean(dim=2)
        J_y1_x0 = batch_jac_y1_x0(x0).mean(dim=2)
        J_y1_y0 = batch_jac_y1_y0(y0).mean(dim=2)
        
        # 累加期望计算
        sum_jac_y0_x0 += J_y0_x0.sum(dim=0).detach()
        sum_jac_y1_x0 += J_y1_x0.sum(dim=0).detach()
        sum_jac_y1_y0 += J_y1_y0.sum(dim=0).detach()
        
        count += batch_sz

    # 求期望雅可比
    E_J_y0_x0 = sum_jac_y0_x0 / count  # [128, 128]
    E_J_y1_x0 = sum_jac_y1_x0 / count  # [128, 128]
    E_J_y1_y0 = sum_jac_y1_y0 / count  # [128, 128]

    # ==========================================
    # 3. 矩阵构建核心逻辑 (均输出 256x256)
    # ==========================================
    
    # 矩阵 A：功能连接 FC
    # 将第一层和第二层的激活值在特征维度直接拼接
    acts_y0 = torch.cat(all_acts_y0, dim=0)  # [N, 128]
    acts_y1 = torch.cat(all_acts_y1, dim=0)  # [N, 128]
    acts_total = torch.cat([acts_y0, acts_y1], dim=1)  # [N, 256]
    
    FC = torch.corrcoef(acts_total.T)  # 相关矩阵 [256, 256]
    FC = torch.nan_to_num(FC)

    # 矩阵 B：结构连接 SC (严格无向对称)
    # 同层连接为0，仅考虑相邻层级的前向雅可比矩阵并强制对称
    SC = torch.zeros(2 * d, 2 * d, device=device)
    SC[0:d, d:2*d] = E_J_y1_y0.T           # Layer 0 到 Layer 1
    SC[d:2*d, 0:d] = E_J_y1_y0         # Layer 1 到 Layer 0 (对称赋值)

    # 矩阵 C：输入相似性 IS
    # 将 L0 和 L1 对应同一基准 (X0) 的雅可比矩阵在行方向堆叠
    # 形成形状为 [256, 128] 的联合“感受野”矩阵
    J_base_total = torch.cat([E_J_y0_x0, E_J_y1_x0], dim=0)
    
    # 计算行向量之间的 Pearson 相关性，代表不同神经元对底层特征偏好的重合度
    IS = torch.corrcoef(J_base_total)  # 相关矩阵 [256, 256]
    IS = torch.nan_to_num(IS)

    # np.save("/mnt/Data16T/Data/haichao/code/AI_connectom/story/story_part2_struc_func/Transformer/FC_IS_SC_matrix/FC_matrix_crosslayer.npy", FC.cpu().detach().numpy())
    # np.save("/mnt/Data16T/Data/haichao/code/AI_connectom/story/story_part2_struc_func/Transformer/FC_IS_SC_matrix/IS_matrix_crosslayer.npy", IS.cpu().detach().numpy())
    # np.save("/mnt/Data16T/Data/haichao/code/AI_connectom/story/story_part2_struc_func/Transformer/FC_IS_SC_matrix/SC_matrix_crosslayer.npy", SC.cpu().detach().numpy())

    return FC, SC, IS

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def plot_multimodal_connectivity(FC, SC, IS, save_path=None):
    """
    绘制 FC, SC, IS 的联合分析图表：
    1. 基于 FC 的层次聚类对三个矩阵进行统一重排序。
    2. 绘制重排后的三个热图。
    3. 绘制 FC-SC 和 FC-IS 的相关性散点图。
    """
    # 确保输入是 numpy 数组格式
    def to_numpy(matrix):
        if isinstance(matrix, torch.Tensor):
            return matrix.detach().cpu().numpy()
        return np.array(matrix)

    FC_np = to_numpy(FC)
    SC_np = to_numpy(SC)
    IS_np = to_numpy(IS)

    # ==========================================
    # 1. 基于 FC 计算层次聚类并获取排序索引
    # ==========================================
    # 将相关系数矩阵 (1 到 -1) 转换为距离矩阵 (0 到 2)
    # 确保矩阵完全对称且对角线严格为0，以满足 scipy.spatial.distance.squareform 的要求
    dist_matrix = 1.0 - FC_np
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    
    # 压缩距离矩阵并进行层次聚类 (使用 ward 最小方差法)
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='ward')
    
    # 获取聚类后的叶子节点顺序
    order = leaves_list(Z)

    # ==========================================
    # 2. 按照 FC 的聚类顺序重排三个矩阵
    # ==========================================
    FC_reordered = FC_np[order, :][:, order]
    SC_reordered = SC_np[order, :][:, order]
    IS_reordered = IS_np[order, :][:, order]

    # ==========================================
    # 3. 提取上三角数据用于绘制散点图并计算相关性
    # ==========================================
    # 提取对角线以上的元素，避免自身相关(对角线)和重复数据(下三角)干扰
    d = FC_np.shape[0]
    triu_idx = np.triu_indices(d, k=1)
    
    fc_vals = FC_np[triu_idx]
    sc_vals = SC_np[triu_idx]
    is_vals = IS_np[triu_idx]

    # 计算全局 Pearson 相关系数
    r_fc_sc = np.corrcoef(fc_vals, sc_vals)[0, 1]
    r_fc_is = np.corrcoef(fc_vals, is_vals)[0, 1]

    # ==========================================
    # 4. 构建排版与绘图
    # ==========================================
    fig = plt.figure(figsize=(20, 12))
    # 创建 2行6列 的网格，用于灵活居中下排的图
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.2, 1], hspace=0.3, wspace=0.4)

    # --- 上排：三个重排后的热图 ---
    # FC 热图
    ax1 = fig.add_subplot(gs[0, 0:2])
    sns.heatmap(FC_reordered, cmap='RdBu_r', center=0, square=True, 
                xticklabels=False, yticklabels=False, ax=ax1, 
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    ax1.set_title('Functional Connectivity (FC)\n(Hierarchically Clustered)', pad=15)

    # SC 热图
    ax2 = fig.add_subplot(gs[0, 2:4])
    # 注意：SC由于包含雅可比矩阵可能数值分布较广，可根据实际数值范围调整 vmax/vmin
    vmax_sc = np.percentile(np.abs(SC_reordered), 99) 
    sns.heatmap(SC_reordered, cmap='RdBu_r', center=0, vmax=vmax_sc, vmin=-vmax_sc, 
                square=True, xticklabels=False, yticklabels=False, ax=ax2, 
                cbar_kws={"shrink": 0.8, "label": "Jacobian Weight"})
    ax2.set_title('Structural Connectivity (SC)\n(Ordered by FC)', pad=15)

    # IS 热图
    ax3 = fig.add_subplot(gs[0, 4:6])
    sns.heatmap(IS_reordered, cmap='RdBu_r', center=0, square=True, 
                xticklabels=False, yticklabels=False, ax=ax3, 
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    ax3.set_title('Input Similarity (IS)\n(Ordered by FC)', pad=15)

    # --- 下排：两个散点图 ---
    # FC vs SC
    ax4 = fig.add_subplot(gs[1, 1:3])
    # alpha 设得很低是因为散点数量非常多 (256*255/2 = 32512 个点)
    ax4.scatter(sc_vals, fc_vals, alpha=0.05, s=2, color='indigo')
    # 添加回归拟合线
    m1, b1 = np.polyfit(sc_vals, fc_vals, 1)
    ax4.plot(sc_vals, m1*sc_vals + b1, color='red', linewidth=1.5)
    ax4.set_xlabel('Structural Connectivity (SC) Weights', fontsize=11)
    ax4.set_ylabel('Functional Connectivity (FC)', fontsize=11)
    ax4.set_title(f'FC vs SC Correlation\nr = {r_fc_sc:.4f}', pad=10)
    ax4.grid(True, linestyle='--', alpha=0.5)

    # FC vs IS
    ax5 = fig.add_subplot(gs[1, 3:5])
    ax5.scatter(is_vals, fc_vals, alpha=0.05, s=2, color='teal')
    m2, b2 = np.polyfit(is_vals, fc_vals, 1)
    ax5.plot(is_vals, m2*is_vals + b2, color='red', linewidth=1.5)
    ax5.set_xlabel('Input Similarity (IS) Correlation', fontsize=11)
    ax5.set_ylabel('Functional Connectivity (FC)', fontsize=11)
    ax5.set_title(f'FC vs IS Correlation\nr = {r_fc_is:.4f}', pad=10)
    ax5.grid(True, linestyle='--', alpha=0.5)

    # 整体调整并显示
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"图表已保存至: {save_path}")
    
    plt.show()

# ============================================================
# 10. 主程序
# ============================================================

if __name__ == '__main__':
    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"结构连接方案: {SC_METHOD}")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载数据集...")
    train_loader, val_loader, vocab = get_dataloaders(
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )

    # 2. 构建模型
    print("\n[2/4] 构建Transformer模型...")
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        seq_len=SEQ_LEN,
        dropout=DROPOUT
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 3. 训练
    print("\n[3/4] 开始训练...")
    train_losses, val_losses, model_states = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, device=device
    )
    print(f"训练完成，共保存 {len(model_states)} 个模型状态")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, save_path='training_curves.png')

    # 4. 连接性分析
    print(f"\n[4/4] 动态连接性分析（SC方案: {SC_METHOD}）...")
    
    # 设置Jacobian模式下的采样数（计算量大，可适当减小）
    n_jac = JACOBIAN_SAMPLES if SC_METHOD == 'jacobian' else 0

    sampled_steps, is_max_values, fc_max_values, corr_values = analyze_connectivity_evolution(
        model_states=model_states,
        train_loader=train_loader,
        val_loader=val_loader,
        sc_method=SC_METHOD,
        fc_method = FC_METHOD,
        layer_idx=ANALYZE_LAYER,
        step_interval=STEP_INTERVAL,
        n_fc_samples=SAMPLES_FOR_FC,
        n_jacobian_samples=n_jac if SC_METHOD == 'jacobian' else JACOBIAN_SAMPLES
    )

    # 绘制动态演化图
    plot_connectivity_evolution(
        sampled_steps, is_max_values, fc_max_values, corr_values,
        sc_method=SC_METHOD,
        layer_idx=ANALYZE_LAYER,
        # save_path=f'connectivity_evolution_{SC_METHOD}.png'
    )

    # 绘制最终状态的SC/FC矩阵快照
    print("\n绘制最终模型的SC/FC矩阵热力图...")
    model.load_state_dict(model_states[-1])
    model.to(device)
    plot_sc_fc_matrix_snapshot(
        model, train_loader,
        sc_method=SC_METHOD,
        fc_method=FC_METHOD,
        layer_idx=ANALYZE_LAYER,
        n_fc_samples=SAMPLES_FOR_FC,
        n_jacobian_samples=JACOBIAN_SAMPLES,
        device=device,
        # save_path=f'sc_fc_matrix_{SC_METHOD}.png'
    )

    print("\n分析完成！")
    print(f"  训练曲线 → training_curves.png")
    print(f"  动态演化 → connectivity_evolution_{SC_METHOD}.png")
    print(f"  矩阵快照 → sc_fc_matrix_{SC_METHOD}.png")

    # 1. 确保模型使用的是训练完成后的最终权重
    # model_states[-1] 是最后一个 epoch/batch 的模型状态
    model.load_state_dict(model_states[-1])
    model.eval()

    # 2. 设置跨层分析的采样数
    # 注意：跨层雅可比矩阵计算量较大，建议在本地测试时先设置为 50 或 100。
    # 显存足够的情况下再逐步增加以获得更平滑的相关性矩阵。
    cross_layer_samples = 100 

    print(f"正在计算 256x256 的 FC, SC, IS 矩阵 (采用 {cross_layer_samples} 个样本)...")
    FC_matrix, SC_matrix, IS_matrix = compute_cross_layer_matrices(
        model=model,
        data_loader=train_loader,
        n_samples=cross_layer_samples,
        device=device,
        fc_method=FC_METHOD
    )

    print("矩阵计算完成，正在进行聚类重排并生成图表...")
    # 3. 绘制并保存结果
    plot_multimodal_connectivity(
        FC=FC_matrix,
        SC=SC_matrix,
        IS=IS_matrix,
        save_path='cross_layer_connectivity_final.png'
    )

    # ── 额外：对比两种SC方案（可选）──────────────────────────────
    # 若想同时运行两种方案并对比，取消下面的注释：
    #
    # print("\n[额外] 对比方案一（linear）与方案二（jacobian）...")
    # steps_l, is_l, fc_l, corr_l = analyze_connectivity_evolution(
    #     model_states, train_loader, val_loader,
    #     sc_method='linear', layer_idx=ANALYZE_LAYER,
    #     step_interval=STEP_INTERVAL, n_fc_samples=SAMPLES_FOR_FC
    # )
    # steps_j, is_j, fc_j, corr_j = analyze_connectivity_evolution(
    #     model_states, train_loader, val_loader,
    #     sc_method='jacobian', layer_idx=ANALYZE_LAYER,
    #     step_interval=STEP_INTERVAL*5,  # Jacobian计算慢，增大间隔
    #     n_jacobian_samples=50
    # )
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # axes[0].plot(steps_l, is_l, label='IS (linear)', linewidth=2)
    # axes[0].plot(steps_l, fc_l, label='FC (linear)', linewidth=2)
    # axes[0].set_title('方案一: 线性骨架'); axes[0].legend()
    # axes[1].plot(steps_j, is_j, label='IS (jacobian)', linewidth=2)
    # axes[1].plot(steps_j, fc_j, label='FC (jacobian)', linewidth=2)
    # axes[1].set_title('方案二: 期望雅可比'); axes[1].legend()
    # plt.tight_layout(); plt.savefig('comparison.png', dpi=150); plt.show()


# ── [新增] 微观神经元对追踪分析 ──────────────────────────────
    print("\n[5/5] 执行全量矩阵提取与微观神经元对分析...")
    
    # 1. 提取历史状态 (这里 step_interval 可以适当调大以节省时间)
    sampled_steps_full, fc_hist_vecs, sc_hist_vecs = extract_full_connectivity_history_transformer(
        model_states=model_states,
        data_loader=train_loader,
        sc_method=SC_METHOD,     
        step_interval=STEP_INTERVAL,
        layer_idx=ANALYZE_LAYER,
        n_fc_samples=SAMPLES_FOR_FC,
        n_jacobian_samples=JACOBIAN_SAMPLES if SC_METHOD == 'jacobian' else 0
    )

    # 2. 绘制微观分析五联图
    analyze_pair_level_dynamics_transformer(
        sampled_steps=sampled_steps_full, 
        fc_history=fc_hist_vecs, 
        sc_history=sc_hist_vecs,
        early_step_idx=1200,            # 定义"早期"的索引 (基于截取后的数组)
        window_size=max(4, len(sampled_steps_full) // 5), # 动态调整窗口
        max_lag=max(2, len(sampled_steps_full) // 10),    # 动态调整滞后
        display_max_steps=2000,              # 默认展示所有步数
        top_k_pairs=2000             # 展示最活跃的1000对神经元
    )