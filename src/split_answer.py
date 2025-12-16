import json
import os
import argparse
from tqdm import tqdm
from utils.color_print import colored_print


def main(args):
    # 设置文件路径
    split_dir = args.split_dir
    answer_path = args.answer_path
    
    # 如果未指定output_dir，则默认与answer_path同目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(answer_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取训练集和测试集的instruction
    colored_print("[INFO] Loading train and val datasets...", color="note")
    
    # 读取训练集
    train_json_path = os.path.join(split_dir, 'train.json')
    if not os.path.exists(train_json_path):
        colored_print(f"[ERROR] Training file not found: {train_json_path}", color="error")
        return
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_instructions = {item['instruction'] for item in train_data}
    colored_print(f"[INFO] Loaded {len(train_data)} training samples", color="note")
    
    # 读取测试集
    val_json_path = os.path.join(split_dir, 'val.json')
    if not os.path.exists(val_json_path):
        colored_print(f"[ERROR] Validation file not found: {val_json_path}", color="error")
        return
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    val_instructions = {item['instruction'] for item in val_data}
    colored_print(f"[INFO] Loaded {len(val_data)} validation samples", color="note")
    
    # 读取完整的answer.json文件
    if not os.path.exists(answer_path):
        colored_print(f"[ERROR] Answer file not found: {answer_path}", color="error")
        return
    colored_print(f"[INFO] Loading full answer file: {answer_path}", color="note")
    with open(answer_path, 'r', encoding='utf-8') as f:
        answer_data = json.load(f)
    colored_print(f"[INFO] Loaded {len(answer_data)} total samples", color="note")
    
    # 拆分答案
    answer_train = []
    answer_val = []
    unmatched = []
    
    colored_print("[INFO] Splitting answers into train and val sets...", color="note")
    for item in tqdm(answer_data, desc="Processing samples"):
        instruction = item['instruction']
        if instruction in train_instructions:
            answer_train.append(item)
        elif instruction in val_instructions:
            answer_val.append(item)
        else:
            unmatched.append(item)
    
    # 打印拆分结果
    colored_print(f"[INFO] Split completed:", color="note")
    colored_print(f"[INFO] Training set: {len(answer_train)} samples", color="success")
    colored_print(f"[INFO] Validation set: {len(answer_val)} samples", color="success")
    colored_print(f"[INFO] Unmatched samples: {len(unmatched)} samples", color="warning")
    
    # 保存拆分结果
    train_output_path = os.path.join(output_dir, 'answer_train.json')
    val_output_path = os.path.join(output_dir, 'answer_val.json')
    
    colored_print(f"[INFO] Saving training answers to: {train_output_path}", color="note")
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_train, f, ensure_ascii=False, indent=2)
    
    colored_print(f"[INFO] Saving validation answers to: {val_output_path}", color="note")
    with open(val_output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_val, f, ensure_ascii=False, indent=2)
    
    # 如果有未匹配的样本，保存到单独的文件
    if unmatched:
        unmatched_path = os.path.join(output_dir, 'answer_unmatched.json')
        colored_print(f"[INFO] Saving unmatched samples to: {unmatched_path}", color="note")
        with open(unmatched_path, 'w', encoding='utf-8') as f:
            json.dump(unmatched, f, ensure_ascii=False, indent=2)
    
    colored_print("[INFO] All operations completed successfully!", color="note")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Split answer.json into train and validation sets based on split directory")
    
    # 添加参数
    parser.add_argument("--split_dir", type=str, default="/data/ljy/workspace/LLM_FineTuning/data/v2/split",
                      help="Directory containing train.json and val.json files (default: /data/ljy/workspace/LLM_FineTuning/data/v2/split)")
    parser.add_argument("--answer_path", type=str, 
                      help="Path to the answer.json file to split")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save split answer files (default: same directory as answer.json)")
    
    # 解析参数
    args = parser.parse_args()
    ###################
    args.answer_path = f'results/251212_172211_550/answer.json'
    ###################
    
    
    # 调用主函数
    main(args)