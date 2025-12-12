import json
import random
import os
from utils.config import GlobalConfig

def split_dataset(input_file, output_dir, test_ratio=0.2, seed=GlobalConfig.seed):
    """
    将数据集拆分为训练集和验证集
    
    Args:
        input_file: 原始数据集文件路径
        output_dir: 输出目录
        test_ratio: 验证集比例
        seed: 随机种子
    """
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 打乱数据顺序
    random.shuffle(data)
    
    # 计算拆分点
    split_point = int(len(data) * (1 - test_ratio))
    
    # 拆分数据集
    train_data = data[:split_point]
    val_data = data[split_point:]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_file = os.path.join(output_dir, GlobalConfig.train_file_name)
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    val_file = os.path.join(output_dir, GlobalConfig.val_file_name)
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    # # 保存拆分信息
    # split_info = {
    #     "seed": seed,
    #     "test_ratio": test_ratio,
    #     "total_samples": len(data),
    #     "train_samples": len(train_data),
    #     "val_samples": len(val_data),
    #     "train_file": train_file,
    #     "val_file": val_file
    # }
    
    # info_file = os.path.join(output_dir, "split_info.json")
    # with open(info_file, 'w', encoding='utf-8') as f:
    #     json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset split completed!")
    print(f"Training set saved to: {train_file}")
    print(f"Validation set saved to: {val_file}")
    # print(f"Split info saved to: {info_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets")
    parser.add_argument("--input_file", type=str, default="./data/v2/dataset_v2.json",
                        help="Input dataset file path")
    parser.add_argument("--output_dir", type=str, default=GlobalConfig.split_dir,
                        help="Output directory for split datasets")
    parser.add_argument("--test_ratio", type=float, default=GlobalConfig.test_ratio,
                        help="Ratio of validation set")
    parser.add_argument("--seed", type=int, default=GlobalConfig.seed,
                        help="Random seed (default: from GlobalConfig)")
    
    args = parser.parse_args()
    
    split_dataset(args.input_file, args.output_dir, args.test_ratio, args.seed)