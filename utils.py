import yaml
import random
import numpy as np
import torch
import hashlib
import torch.nn as nn

from logger import trading_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3
max_trials = 50
model_path = './models/best_model.pth'
params_path = './models/best_hyperparams.pkl'
n_components = 10

def get_model_hash(model):
    state_dict = model.state_dict()
    hash_obj = hashlib.md5(str(state_dict).encode())
    return hash_obj.hexdigest()

def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        trading_logger.error(f"加载配置文件失败: {str(e)}")
        raise

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU
    torch.backends.cudnn.deterministic = True  # 确保 CuDNN 行为确定性
    torch.backends.cudnn.benchmark = False     # 禁用 CuDNN 自动优化

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # alpha 可以是张量，长度与类别数相同
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]，模型的原始输出（logits）
        # targets: [batch_size]，真实标签（整数）
        probs = F.softmax(inputs, dim=1)
        batch_size = inputs.size(0)
        pt = probs[range(batch_size), targets]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        modulating_factor = (1 - pt) ** self.gamma
        loss = modulating_factor * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss