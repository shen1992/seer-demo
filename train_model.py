import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import RobustScaler
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import copy
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
import optuna

from features import feature_cols_all, feature_cols, core_feature_cols
from logger import trading_logger
from utils import device

class TimeSeriesDataset(Dataset):
    def __init__(self, data_4h, data_1h, data_daily, labels, timestamps, seq_len, pred_len):
        self.data_4h = data_4h  # 4H 特征
        self.data_1h = data_1h  # 1H 特征
        self.data_daily = data_daily  # 日线特征
        self.labels = labels
        self.timestamps = pd.to_datetime(timestamps)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(data_4h) - seq_len - pred_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X_4h = self.data_4h[idx:idx + self.seq_len]
        X_1h = self.data_1h[idx:idx + self.seq_len]
        X_daily = self.data_daily[idx:idx + self.seq_len]

        y = self.labels[idx + self.seq_len - 1]

        ts_encoder = self.timestamps[idx:idx + self.seq_len]  # 历史序列的时间戳
        ts_decoder = self.timestamps[idx + self.seq_len:idx + self.seq_len + self.pred_len]

        return X_4h, X_1h, X_daily, y, ts_encoder, ts_decoder

def custom_collate_fn(batch):
    X_4h = [item[0] for item in batch]
    X_1h = [item[1] for item in batch]
    X_daily = [item[2] for item in batch]

    y = [item[3] for item in batch]

    ts_encoder = [item[4] for item in batch]
    ts_decoder = [item[5] for item in batch]
    return (
        default_collate(X_4h).to(device, dtype=torch.float32),
        default_collate(X_1h).to(device, dtype=torch.float32),
        default_collate(X_daily).to(device, dtype=torch.float32),
        default_collate(y).to(device, dtype=torch.float32),
        ts_encoder,
        ts_decoder
    )

def prepare_sequences(df, seq_len=96, pred_len=6, train_ratio=0.7, val_ratio = 0.15):
    """完整预处理流程"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    scaler_4h = RobustScaler()
    scaler_1h = RobustScaler()
    scaler_daily = RobustScaler()
    scaler_label = RobustScaler()  # 新增标签缩放器

    # 4H 特征
    cols_4h = [f'4h_{col}' for col in feature_cols]
    train_data_4h = scaler_4h.fit_transform(df_train[cols_4h])
    val_data_4h = scaler_4h.transform(df_val[cols_4h])
    test_data_4h = scaler_4h.transform(df_test[cols_4h])

    # 1H 特征（只包含 mean 和 std
    cols_1h = [f'1h_{col}_{stat}' for col in core_feature_cols for stat in ['mean', 'std']]
    train_data_1h = scaler_1h.fit_transform(df_train[cols_1h])
    val_data_1h = scaler_1h.transform(df_val[cols_1h])
    test_data_1h = scaler_1h.transform(df_test[cols_1h])

    # 日线特征
    cols_daily = [f'daily_{col}' for col in core_feature_cols]
    train_data_daily = scaler_daily.fit_transform(df_train[cols_daily])
    val_data_daily = scaler_daily.transform(df_val[cols_daily])
    test_data_daily = scaler_daily.transform(df_test[cols_daily])

    # PCA 降维
    # pca = PCA(n_components=10)
    # train_data = pca.fit_transform(train_data)
    # val_data = pca.transform(val_data)
    # test_data = pca.transform(test_data)
    # trading_logger.info(f"PCA 保留方差比例: {sum(pca.explained_variance_ratio_):.4f}")

    # 标签缩放
    train_labels = scaler_label.fit_transform(df_train[['future_return']].values)
    val_labels = scaler_label.transform(df_val[['future_return']].values)
    test_labels = scaler_label.transform(df_test[['future_return']].values)

    train_dataset = TimeSeriesDataset(
        train_data_4h, train_data_1h, train_data_daily,
        train_labels.flatten(), df_train['timestamp'].values, seq_len, pred_len
    )
    val_dataset = TimeSeriesDataset(
        val_data_4h, val_data_1h, val_data_daily,
        val_labels.flatten(), df_val['timestamp'].values, seq_len, pred_len
    )
    test_dataset = TimeSeriesDataset(
        test_data_4h, test_data_1h, test_data_daily,
        test_labels.flatten(), df_test['timestamp'].values, seq_len, pred_len
    )

    return train_dataset, val_dataset, test_dataset, (scaler_4h, scaler_1h, scaler_daily, scaler_label)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # (batch_size, seq_len, d_model)
        return x

class FullAttention(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.scale = 1 / math.sqrt(d_model // nhead)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V, mask=None):
        batch, seq_len, _ = Q.size()
        k_len = K.size(1)
        head_dim = self.d_model // self.nhead
        Q = Q.view(batch, seq_len, self.nhead, head_dim).transpose(1, 2)
        K = K.view(batch, k_len, self.nhead, head_dim).transpose(1, 2)
        V = V.view(batch, k_len, self.nhead, head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return output, attn_weights

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout=0.2, factor=5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.factor = factor
        self.scale = 1 / math.sqrt(d_model // nhead)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V):
        batch = Q.size(0)
        q_len = Q.size(1)
        k_len = K.size(1)
        head_dim = self.d_model // self.nhead

        # reshape for multi-head: (batch, nhead, seq_len, head_dim)
        Q = Q.view(batch, q_len, self.nhead, head_dim).transpose(1, 2)  # (B, H, Q, D)
        K = K.view(batch, k_len, self.nhead, head_dim).transpose(1, 2)  # (B, H, K, D)
        V = V.view(batch, k_len, self.nhead, head_dim).transpose(1, 2)  # (B, H, K, D)

        # raw attention scores: (B, H, Q, K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 计算注意力分布
        probs = torch.softmax(scores, dim=-1)  # (B, H, Q, K)
        uniform_dist = torch.ones_like(probs) / k_len  # 均匀分布

        # 计算 KL 散度
        kl_div = (probs * (probs / uniform_dist).log()).sum(dim=-1)  # (B, H, Q)
        top_u = min(int(self.factor * math.log(q_len)), q_len)
        _, indices = kl_div.topk(top_u, dim=-1)  # 选择 top_u 个查询

        # M = (scores.max(dim=-1)[0] - scores.mean(dim=-1)).log()  # (B, H, Q)
        # top_u = min(int(self.factor * math.log(q_len)), q_len)  # 控制稀疏 Query 的数量
        # _, indices = M.topk(top_u, dim=-1)

        # 稀疏化 Q：只保留 top_u 个重要的 Query
        Q_sparse = Q.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))  # (B, H, top_u, D)

        # 使用稀疏 Q 计算注意力
        scores_sparse = torch.matmul(Q_sparse, K.transpose(-2, -1)) * self.scale  # (B, H, top_u, K)

        attn_weights = torch.softmax(scores_sparse, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output_sparse = torch.matmul(attn_weights, V)  # (B, H, top_u, D)

        # 构建完整输出（稀疏位置有值，其余为0）
        output_full = torch.zeros_like(Q)  # (B, H, Q, D)
        output_full.scatter_(2, indices.unsqueeze(-1).expand(-1, -1, -1, head_dim), output_sparse)

        # 合并多头，恢复形状 (B, Q, D_model)
        output = output_full.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)

        return output, attn_weights

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.hour_embed = nn.Embedding(24, d_model // 4)
        self.weekday_embed = nn.Embedding(7, d_model // 4)
        self.month_embed = nn.Embedding(12, d_model // 4)
        self.linear = nn.Linear(3 * (d_model // 4), d_model)

    def forward(self, timestamps):
        hours = torch.tensor([[ts.hour for ts in ts_list] for ts_list in timestamps], dtype=torch.long, device=device)
        weekdays = torch.tensor([[ts.weekday() for ts in ts_list] for ts_list in timestamps], dtype=torch.long, device=device)
        months = torch.tensor([[ts.month - 1 for ts in ts_list] for ts_list in timestamps], dtype=torch.long, device=device)

        hour_emb = self.hour_embed(hours) # (batch_size, seq_len, d_model//4)
        weekday_emb = self.weekday_embed(weekdays)
        month_emb = self.month_embed(months)

        time_emb = torch.cat([hour_emb, weekday_emb, month_emb], dim=-1) # (batch_size, seq_len, 3*d_model//4)

        return self.linear(time_emb) # (batch_size, seq_len, d_model)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2, factor=5):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, nhead, attn_dropout=dropout, factor=factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x, attn = self.attention(x, x, x)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        return x, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2, factor=5):
        super().__init__()
        self.self_attention = FullAttention(d_model, nhead, attn_dropout=dropout)
        self.cross_attention = ProbSparseAttention(d_model, nhead, attn_dropout=dropout, factor=factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None):
        residual = x
        x, attn1 = self.self_attention(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        residual = x
        x, attn2 = self.cross_attention(x, enc_output, enc_output)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.norm3(x + residual)
        return x, attn1, attn2

class MultiTimeframeInformer(nn.Module):
    def __init__(self,
                 input_dim_4h, input_dim_1h, input_dim_daily,
                 d_model=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=2,
                 pred_len=6, dropout=0.2, factor=5):
        super().__init__()
        self.input_dim_4h = input_dim_4h
        self.input_dim_1h = input_dim_1h
        self.input_dim_daily = input_dim_daily

        self.d_model = d_model
        self.pred_len = pred_len

        # 各时间框架的输入投影
        self.input_projection_4h = nn.Linear(input_dim_4h, d_model)
        self.input_projection_1h = nn.Linear(input_dim_1h, d_model)
        self.input_projection_daily = nn.Linear(input_dim_daily, d_model)

        # 位置编码和时间嵌入
        self.pos_encoder = PositionalEncoding(d_model)
        self.time_embed = TimeFeatureEmbedding(d_model)
        self.timeframe_embed = nn.Embedding(2, d_model)  # 1H=0, Daily=1

        # 各时间框架的编码器
        self.encoder_4h = nn.ModuleList([EncoderLayer(d_model, nhead, dropout, factor) for _ in range(num_encoder_layers)])
        self.encoder_1h = nn.ModuleList([EncoderLayer(d_model, nhead, dropout, factor) for _ in range(num_encoder_layers)])
        self.encoder_daily = nn.ModuleList([EncoderLayer(d_model, nhead, dropout, factor) for _ in range(num_encoder_layers)])

        # 跨时间框架注意力
        self.cross_attention = FullAttention(d_model, nhead, dropout)
        # 添加残差连接后的 LayerNorm
        self.norm_cross = nn.LayerNorm(d_model)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        # 解码器
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, nhead, dropout, factor) for _ in range(num_decoder_layers)
        ])

        # 回归器
        self.attn_pool = nn.Linear(d_model, 1)  # 为每个时间步生成注意力权重
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, src_4h, src_1h, src_daily, ts_encoder, ts_decoder):
        batch_size = src_4h.size(0)

        # 输入投影
        src_4h = self.input_projection_4h(src_4h)
        src_1h = self.input_projection_1h(src_1h)
        src_daily = self.input_projection_daily(src_daily)

        # 位置编码
        src_4h = self.pos_encoder(src_4h)
        src_1h = self.pos_encoder(src_1h)
        src_daily = self.pos_encoder(src_daily)

        # 时间嵌入
        time_emb = self.time_embed(ts_encoder)
        src_4h = src_4h + time_emb
        src_1h = src_1h + time_emb
        src_daily = src_daily + time_emb

        # 添加 Timeframe Embedding
        timeframe_1h = self.timeframe_embed(torch.zeros(batch_size, src_1h.size(1), dtype=torch.long, device=device))
        timeframe_daily = self.timeframe_embed(torch.ones(batch_size, src_daily.size(1), dtype=torch.long, device=device))
        src_1h = src_1h + timeframe_1h
        src_daily = src_daily + timeframe_daily

        # 各时间框架独立编码
        for layer in self.encoder_4h:
            src_4h, _ = layer(src_4h)
        for layer in self.encoder_1h:
            src_1h, _ = layer(src_1h)
        for layer in self.encoder_daily:
            src_daily, _ = layer(src_daily)

        # 跨时间框架注意力：4H 为主，整合 1H 和日线
        memory = torch.cat([src_1h, src_daily], dim=1)
        enc_output, _ = self.cross_attention(src_4h, memory, memory)

        # 添加残差连接：保留原始 src_4h 信息
        enc_output = enc_output + src_4h
        # 归一化
        enc_output = self.norm_cross(enc_output)

        # 初始化解码器输入
        dec_input = enc_output[:, -1:, :].repeat(1, self.pred_len, 1)
        dec_input = self.pos_encoder(dec_input)
        time_emb_dec = self.time_embed(ts_decoder)
        dec_input = dec_input + time_emb_dec

        tgt_mask = torch.triu(torch.ones(self.pred_len, self.pred_len), diagonal=1).bool().to(device)
        dec_output = dec_input
        for layer in self.decoder:
            dec_output, _, _ = layer(dec_output, enc_output, tgt_mask=tgt_mask)

        # 注意力加权聚合
        attn_weights = F.softmax(self.attn_pool(dec_output), dim=1)
        dec_output_pooled = (dec_output * attn_weights).sum(dim=1)
        # 回归
        output = self.regressor(dec_output_pooled)
        return output

def evaluate_predictions(predictions, y_true, scaler_label, threshold=0.02):
    y_true = scaler_label.inverse_transform(y_true.reshape(-1, 1)).flatten()
    predictions = scaler_label.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # 初始化结果字典，预设默认值
    results = {
        'mae': mean_absolute_error(y_true, predictions),
        'mse': np.mean((y_true - predictions) ** 2),
        'pred_quantiles': {
            0.01: np.quantile(predictions, 0.01),
            0.05: np.quantile(predictions, 0.05),
            0.2: np.quantile(predictions, 0.2),
            0.25: np.quantile(predictions, 0.25),
            0.35: np.quantile(predictions, 0.35),
            0.65: np.quantile(predictions, 0.65),
            0.75: np.quantile(predictions, 0.75),
            0.8: np.quantile(predictions, 0.8),
            0.95: np.quantile(predictions, 0.95),
            0.99: np.quantile(predictions, 0.99)
        }
    }

    # 定义信号类型和对应的掩码、概率条件
    signal_types = {
        'long': {
            'mask': predictions > threshold,
            'prob_condition': lambda x: x > threshold,
            'name': f'多头信号 (预测>{threshold:.2%})'
        },
        'short': {
            'mask': predictions < -threshold,
            'prob_condition': lambda x: x < -threshold,
            'name': f'空头信号 (预测<-{threshold:.2%})'
        },
        'hold': {
            'mask': (predictions >= -threshold) & (predictions <= threshold),
            'prob_condition': lambda x: (x >= -threshold) & (x <= threshold),
            'name': f'不交易信号 (预测[-{threshold:.2%}, {threshold:.2%}])'
        }
    }

    # 为每种信号初始化默认统计指标
    for signal in signal_types:
        results.update({
            f'{signal}_pred_count': 0,
            f'{signal}_actual_median': 0.0,
            f'{signal}_actual_prob': 0.0,
            f'{signal}_actual_mean': 0.0,
            f'{signal}_actual_std': 0.0
        })

    # 计算每种信号的分布统计
    for signal, config in signal_types.items():
        mask = config['mask']
        actual = y_true[mask]
        results[f'{signal}_pred_count'] = int(np.sum(mask))

        if len(actual) > 0:
            results[f'{signal}_actual_median'] = np.median(actual)
            results[f'{signal}_actual_prob'] = np.mean(config['prob_condition'](actual))
            results[f'{signal}_actual_mean'] = np.mean(actual)
            results[f'{signal}_actual_std'] = np.std(actual)

        trading_logger.info(f"{config['name']}:")
        trading_logger.info(f"  预测次数: {results[f'{signal}_pred_count']}")
        trading_logger.info(f"  实际中位数: {results[f'{signal}_actual_median']:.2%}")
        trading_logger.info(f"  实际符合条件的概率: {results[f'{signal}_actual_prob']:.1%}")
        trading_logger.info(f"  实际均值: {results[f'{signal}_actual_mean']:.2%}, 标准差: {results[f'{signal}_actual_std']:.2%}")

    trading_logger.info(f"预测值分位数: {results['pred_quantiles']}")
    return results

class CombinedLoss(nn.Module):
    def __init__(self, scaler_label,
                 target_long_median, target_short_median,
                 threshold_scaled,
                 mae_weight=0.25, median_weight=0.5, focal_weight=0.25):
        super().__init__()
        self.scaler_label = scaler_label
        self.target_long_median = torch.tensor(target_long_median, dtype=torch.float32, device=device)
        self.target_short_median = torch.tensor(target_short_median, dtype=torch.float32, device=device)
        self.mae_weight = mae_weight
        self.focal_weight = focal_weight
        self.median_weight = median_weight
        self.threshold = torch.tensor(threshold_scaled, dtype=torch.float32, device=device)

    def mae_loss(self, pred, target):
        abs_error = torch.abs(pred - target)
        abs_target = torch.abs(target)
        abs_pred = torch.abs(pred)

        # 初始化权重为 1.0
        weights = torch.ones_like(target)

        # 高收益样本：|target| >= self.threshold
        high_return_mask = abs_target >= self.threshold
        # 保守预测：|pred| < |target|
        conservative_mask = abs_pred < abs_target

        # 为高收益且保守预测的样本设置权重 3.0
        weights[high_return_mask & conservative_mask] = 1.5
        # 为高收益但非保守预测的样本设置权重 1.5
        weights[high_return_mask & ~conservative_mask] = 0.5
        # 低收益样本（|target| < self.threshold）保持权重 1.0

        return (abs_error * weights).mean()

    def focal_loss(self, pred, target, gamma=3.0, alpha=0.25):
        error = torch.abs(pred - target)
        pt = torch.exp(-error)
        focal = alpha * (1 - pt) ** gamma * error
        return focal.mean()

    def median_guidance_loss(self, pred):
        threshold = torch.tensor((0.02 - self.scaler_label.center_[0]) / self.scaler_label.scale_[0],
                           dtype=torch.float32, device=pred.device)
        long_mask = pred > threshold
        long_median_loss = torch.tensor(0.0, device=pred.device)
        if long_mask.sum() > 0:
            long_preds = pred[long_mask]
            long_median = torch.median(long_preds) if long_preds.numel() > 0 else 0.0
            long_median_loss = torch.abs(long_median - self.target_long_median)

        short_mask = pred < -threshold
        short_median_loss = torch.tensor(0.0, device=pred.device)
        if short_mask.sum() > 0:
            short_preds = pred[short_mask]
            short_median = torch.median(short_preds) if short_preds.numel() > 0 else 0.0
            short_median_loss = torch.abs(short_median - self.target_short_median)

        return (long_median_loss + short_median_loss) / 2

    def forward(self, pred, target):
        mae_loss = self.mae_loss(pred, target)
        median_loss = self.median_guidance_loss(pred)
        focal_loss = self.focal_loss(pred, target)

        total_loss = (
            self.mae_weight * mae_loss +
            self.median_weight * median_loss +
            self.focal_weight * focal_loss
        )
        return total_loss

def train_model(model,
                train_loader, val_loader, test_loader,
                scalers, long_median, short_median,
                lr=0.0005,
                mae_weight=0.25, focal_weight=0.25, median_weight=0.5,
                epochs=100, patience=10, finetune_epochs=20, finetune_lr=1e-5):
    scaler_label = scalers[3]
    threshold = 0.02
    threshold_scaled = (threshold - scaler_label.center_[0]) / scaler_label.scale_[0]

    criterion = CombinedLoss(
        scaler_label=scaler_label,
        target_long_median=(long_median - scaler_label.center_[0]) / scaler_label.scale_[0],
        target_short_median=(short_median - scaler_label.center_[0]) / scaler_label.scale_[0],
        threshold_scaled=threshold_scaled,
        mae_weight=mae_weight,
        median_weight=median_weight,
        focal_weight=focal_weight
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=0.001,
        min_lr=1e-6,
    )

    # 检查 long/short 样本量
    num_samples = len(train_loader.dataset)
    label_indices = np.arange(seq_len-1, num_samples+pred_len)
    labels = train_loader.dataset.labels[label_indices]
    long_short_mask = (labels > threshold_scaled) | (labels < -threshold_scaled)
    trading_logger.info(f"总样本数: {len(labels)}, long 样本 (y > 0.02): {np.sum(labels > threshold_scaled)}, "
                      f"short 样本 (y < -0.02): {np.sum(labels < -threshold_scaled)}")

    # 第一阶段：全数据训练（均匀采样）
    trading_logger.info("=== 开始全数据训练 ===")
    train_losses = []
    best_score = float('-inf')
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_x_4h, batch_x_1h, batch_x_daily, batch_y, batch_ts_encoder, batch_ts_decoder in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x_4h, batch_x_1h, batch_x_daily, batch_ts_encoder, batch_ts_decoder)
            loss = criterion(pred.squeeze(), batch_y.float())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        total_val_loss = 0
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch_x_4h, batch_x_1h, batch_x_daily, batch_y, batch_ts_encoder, batch_ts_decoder in val_loader:
                pred = model(batch_x_4h, batch_x_1h, batch_x_daily, batch_ts_encoder, batch_ts_decoder)
                loss = criterion(pred.squeeze(), batch_y.float())
                total_val_loss += loss.item()
                all_preds.extend(pred.squeeze().cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_mse = np.mean((np.array(all_preds) - np.array(all_labels)) ** 2)
        val_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

        # 验证集分布评估
        trading_logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}")
        results = evaluate_predictions(np.array(all_preds), np.array(all_labels), scaler_label)

        val_prob = 0.5 * results['long_actual_prob'] + 0.5 * results['short_actual_prob']
        long_median_error = abs(results['long_actual_median'] - long_median) if results['long_pred_count'] > 0 else 0.0
        short_median_error = abs(results['short_actual_median'] - short_median) if results['short_pred_count'] > 0 else 0.0
        score = val_prob - 0.5 * (long_median_error + short_median_error) - 0.2 * val_mae

        trading_logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                           f"Val MAE: {val_mae:.4f}, Val MSE: {val_mse}, Val Prob: {val_prob:.4f}, "
                           f"Long Median Error: {long_median_error:.4f}, Short Median Error: {short_median_error:.4f}, "
                           f"Score: {score:.4f}")
        scheduler.step(avg_val_loss)

        if score > best_score:
            best_score = score
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                trading_logger.info(f"Early stopping at epoch {epoch+1}")
                break

    trading_logger.info(f'load best_epoch:{best_epoch}')
    model.load_state_dict(best_model_state)
    trading_logger.info("=== 全数据训练完成，加载最佳权重 ===")

    # 第二阶段：微调（冻结编码器，均匀采样）

    # 冻结编码器
    for param in model.encoder_4h[:-1].parameters():
        param.requires_grad = False
    for param in model.encoder_1h.parameters():
        param.requires_grad = False
    for param in model.encoder_daily.parameters():
        param.requires_grad = False
    for param in model.cross_attention.parameters():
        param.requires_grad = False

    long_short_indices = np.where(long_short_mask)[0]
    finetune_dataset = Subset(train_loader.dataset, long_short_indices)
    finetune_loader = DataLoader(finetune_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    # 微调设置
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=finetune_lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

    # 微调循环（保存最佳模型）
    best_finetune_score = float('-inf')
    best_finetune_state = copy.deepcopy(model.state_dict())
    finetune_early_stop_counter = 0
    finetune_patience = 5  # 微调阶段早停耐心
    best_finetune_epoch = 0

    # 微调循环
    for epoch in range(finetune_epochs):
        model.train()
        total_finetune_loss = 0

        for batch_x_4h, batch_x_1h, batch_x_daily, batch_y, batch_ts_encoder, batch_ts_decoder in finetune_loader:
            optimizer.zero_grad()
            pred = model(batch_x_4h, batch_x_1h, batch_x_daily, batch_ts_encoder, batch_ts_decoder)
            loss = criterion(pred.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
            total_finetune_loss += loss.item()

        avg_finetune_loss = total_finetune_loss / len(finetune_loader)

        total_val_loss = 0
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch_x_4h, batch_x_1h, batch_x_daily, batch_y, batch_ts_encoder, batch_ts_decoder in val_loader:
                pred = model(batch_x_4h, batch_x_1h, batch_x_daily, batch_ts_encoder, batch_ts_decoder)
                loss = criterion(pred.squeeze(), batch_y.float())
                total_val_loss += loss.item()
                all_preds.extend(pred.squeeze().cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_mae = mean_absolute_error(all_preds, all_labels)
        results = evaluate_predictions(np.array(all_preds), np.array(all_labels), scaler_label)

        val_prob = 0.5 * results['long_actual_prob'] + 0.5 * results['short_actual_prob']
        long_median_error = abs(results['long_actual_median'] - long_median) if results['long_pred_count'] > 0 else 0.0
        short_median_error = abs(results['short_actual_median'] - short_median) if results['short_pred_count'] > 0 else 0.0
        finetune_score = val_prob - 0.5 * (long_median_error + short_median_error) - 0.2 * val_mae

        trading_logger.info(f"Finetune Epoch {epoch+1}, Finetune Loss: {avg_finetune_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                         f"Val MAE: {val_mae:.4f}, Long Prob: {results['long_actual_prob']:.4f}, "
                         f"Short Prob: {results['short_actual_prob']:.4f},"
                         f"finetune_score: {finetune_score:.4f}")
        scheduler.step(avg_val_loss)

        if finetune_score > best_finetune_score:
            best_finetune_score = finetune_score
            best_finetune_state = copy.deepcopy(model.state_dict())
            finetune_early_stop_counter = 0
            best_finetune_epoch = epoch + 1
        else:
            finetune_early_stop_counter += 1
            if finetune_early_stop_counter >= finetune_patience:
                trading_logger.info(f"Finetune early stopping at epoch {epoch+1}")
                break

    # 加载微调阶段最佳模型
    model.load_state_dict(best_finetune_state)
    trading_logger.info(f'load best_finetune_epoch:{best_finetune_epoch}')
    trading_logger.info("=== 微调完成，加载微调阶段最佳权重 ===")

    # 测试集评估
    trading_logger.info("=== 微调后测试结果 ===")
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch_x_4h, batch_x_1h, batch_x_daily, batch_y, batch_ts_encoder, batch_ts_decoder in test_loader:
            pred = model(batch_x_4h, batch_x_1h, batch_x_daily, batch_ts_encoder, batch_ts_decoder)
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 测试集分布评估
    trading_logger.info(f"Final Test Results:")
    evaluate_predictions(np.array(all_preds), np.array(all_labels), scaler_label)

    return best_finetune_score

# Optuna 目标函数
def objective(trial):
    # 定义高优先级超参数搜索空间
    d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
    nhead = trial.suggest_categorical('nhead', [4, 8, 16])
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    factor = trial.suggest_categorical('factor', [3, 5, 7, 10])
    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    finetune_lr = trial.suggest_float('finetune_lr', 1e-6, 1e-4, log=True)

    mae_weight = trial.suggest_float('mae_weight', 0.1, 0.5)
    focal_weight = trial.suggest_float('focal_weight', 0.1, 0.5)
    median_weight = trial.suggest_float('median_weight', 0.2, 0.8)

    # 归一化损失权重
    total_weight = mae_weight + focal_weight + median_weight
    mae_weight /= total_weight
    focal_weight /= total_weight
    median_weight /= total_weight

    # 初始化模型
    model = MultiTimeframeInformer(
        input_dim_4h=len([col for col in feature_cols_all if col.startswith('4h_')]),
        input_dim_1h=len([col for col in feature_cols_all if col.startswith('1h_')]),
        input_dim_daily=len([col for col in feature_cols_all if col.startswith('daily_')]),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=3,  # 固定
        num_decoder_layers=2,  # 固定
        pred_len=6,
        dropout=dropout,
        factor=factor
    ).to(device)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # 训练模型并返回验证集得分
    score = train_model(
        model, train_loader, val_loader, test_loader, scalers, long_median, short_median,
        lr=lr,
        mae_weight=mae_weight, focal_weight=focal_weight, median_weight=median_weight,
        finetune_lr=finetune_lr,
    )

    return score

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from data import load_data
    from features import create_trend_label, create_features, align_timeframes
    from utils import set_seed

    seed = 42
    set_seed(seed)
    seq_len = 96
    pred_len = 6

    # 加载数据
    df_1h = load_data('btc_1h.csv')
    df_4h = load_data('btc_4h.csv')
    df_daily = load_data('btc_d.csv')

    # 生成特征
    df_1h = create_features(df_1h, timeframe='1H')
    df_4h = create_features(df_4h, timeframe='4H')
    df_daily = create_features(df_daily, timeframe='D')

    # 生成4小时线标签
    df_4h, long_median, short_median = create_trend_label(df_4h)

    # 对齐时间框架
    df_merged = align_timeframes(df_1h, df_4h, df_daily)

    # 准备数据集
    train_dataset, val_dataset, test_dataset, scalers = prepare_sequences(df_merged)

    # 创建 Optuna 研究
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50)

    # 输出最佳超参数
    best_params = study.best_params
    best_score = study.best_value
    trading_logger.info(f"Best Parameters: {best_params}")
    trading_logger.info(f"Best Validation Score: {best_score:.4f}")

    # num_samples = len(train_dataset)
    # threshold = 0.02
    # scaler_label = scalers[3]
    # threshold_scaled = (threshold - scaler_label.center_[0]) / scaler_label.scale_[0]
    # # 加权采样
    # label_indices = np.arange(seq_len-1, num_samples+pred_len)
    # labels = train_dataset.labels[label_indices]
    # weights = np.ones_like(labels)
    # long_short_mask = (labels > threshold_scaled) | (labels < -threshold_scaled)
    # weights[long_short_mask] = 1.5
    # sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    best_model = MultiTimeframeInformer(
        input_dim_4h=len([col for col in feature_cols_all if col.startswith('4h_')]),
        input_dim_1h=len([col for col in feature_cols_all if col.startswith('1h_')]),
        input_dim_daily=len([col for col in feature_cols_all if col.startswith('daily_')]),
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_encoder_layers=3,
        num_decoder_layers=2,
        pred_len=6,
        dropout=best_params['dropout'],
        factor=best_params['factor']
    ).to(device)
    train_model(best_model,
                train_loader, val_loader, test_loader,
                scalers,
                long_median, short_median,
                lr=best_params['lr'],
                mae_weight=best_params['mae_weight'],
                focal_weight=best_params['focal_weight'],
                median_weight=best_params['median_weight'],
                finetune_lr=best_params['finetune_lr'])

