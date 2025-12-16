import torch
import random
import numpy as np
from utils.color_print import colored_print


def set_seed(seed: int, verbose: bool = True):
    """
    设置随机种子，确保实验的可重复性
    
    Args:
        seed (int): 要设置的随机种子值
        verbose (bool): 是否打印种子设置信息
    """
    if verbose:
        colored_print(f"[INFO] Setting random seed: {seed}", color="note")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 如果使用 CUDA，还需要设置 CUDA 相关的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # # 确保 CUDA 操作的确定性（可能会影响性能）
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    