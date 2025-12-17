import argparse
import json
import statistics
import os
from tqdm import tqdm  # 添加 tqdm 导入
from utils.color_print import colored_print
from utils.config import SFTConfig
from unsloth import FastLanguageModel

def analyze_answer_tokens(dataset_path, model_path=None):
    colored_print(f"[INFO] Analyzing answer tokens from: {dataset_path}", color="note")
    
    # 如果没有指定模型路径，使用配置文件中的基础模型路径
    if not model_path:
        model_path = SFTConfig.base_model_path
    
    # 加载tokenizer
    colored_print(f"[INFO] Loading tokenizer from: {model_path}", color="note")
    try:
        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=SFTConfig.max_seq_length,
            dtype=SFTConfig.dtype,
            load_in_4bit=False,  # 只加载tokenizer，不需要4bit量化
        )
        colored_print("[INFO] Tokenizer loaded successfully.", color="note")
    except Exception as e:
        colored_print(f"[ERROR] Failed to load tokenizer from {model_path}.", color="red")
        colored_print(f"[ERROR] Error message: {str(e)}", color="red")
        exit(1)
    
    # 读取数据集
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        colored_print(f"[INFO] Dataset loaded successfully. Total samples: {len(dataset)}", color="note")
    except Exception as e:
        colored_print(f"[ERROR] Failed to load dataset from {dataset_path}.", color="red")
        colored_print(f"[ERROR] Error message: {str(e)}", color="red")
        exit(1)
    
    # 提取所有答案的token长度
    answer_token_lengths = []
    char_token_ratio = []
    
    # 添加 tqdm 进度条
    for sample in tqdm(dataset, desc="Processing samples", unit="sample"):
        if "output" in sample and sample["output"]:
            answer = sample["output"].strip()
            # 使用tokenizer计算token数量
            token_ids = tokenizer.encode(answer)
            token_length = len(token_ids)
            answer_token_lengths.append(token_length)
            
            # 计算字符与token的比例
            char_length = len(answer)
            ratio = char_length / token_length if token_length > 0 else 0
            char_token_ratio.append(ratio)
    
    if not answer_token_lengths:
        colored_print("[ERROR] No valid answers found in the dataset.", color="red")
        exit(1)
    
    # 计算统计指标
    max_tokens = max(answer_token_lengths)
    min_tokens = min(answer_token_lengths)
    avg_tokens = statistics.mean(answer_token_lengths)
    median_tokens = statistics.median(answer_token_lengths)
    stdev_tokens = statistics.stdev(answer_token_lengths) if len(answer_token_lengths) > 1 else 0
    
    # 计算字符与token的平均比例
    avg_char_token_ratio = statistics.mean(char_token_ratio) if char_token_ratio else 0
    
    # 计算token长度分布
    bins = [0, 50, 100, 200, 300, 500, 1000, 2000, float('inf')]
    bin_labels = ["0-50", "51-100", "101-200", "201-300", "301-500", "501-1000", "1001-2000", ">2000"]
    
    bin_counts = [0] * len(bin_labels)
    for length in tqdm(answer_token_lengths, desc="Calculating token distributions", unit="answer"):
        for i, bin_max in enumerate(bins[1:]):
            if length <= bin_max:
                bin_counts[i] += 1
                break
    
    # 计算每个区间的百分比
    bin_percentages = []
    total_answers = len(answer_token_lengths)
    for count in bin_counts:
        percentage = (count / total_answers) * 100 if total_answers > 0 else 0
        bin_percentages.append(percentage)
    
    # 输出结果
    colored_print("\n[INFO] Answer Token Length Statistics:", color="note")
    colored_print(f"Total answers analyzed: {len(answer_token_lengths)}", color="note")
    colored_print(f"Maximum tokens: {max_tokens} tokens", color="green")
    colored_print(f"Minimum tokens: {min_tokens} tokens", color="green")
    colored_print(f"Average tokens: {avg_tokens:.2f} tokens", color="green")
    colored_print(f"Median tokens: {median_tokens} tokens", color="green")
    colored_print(f"Standard deviation: {stdev_tokens:.2f} tokens", color="green")
    colored_print(f"Average characters per token: {avg_char_token_ratio:.2f} chars/token", color="green")
    
    # 输出token长度分布
    colored_print("\n[INFO] Answer Token Length Distribution:", color="note")
    for label, count, percentage in zip(bin_labels, bin_counts, bin_percentages):
        colored_print(f"{label}: {count} answers ({percentage:.1f}%)", color="note")
    
    # 建议的max_new_tokens值
    # 基于最大token长度，加上一个20%的缓冲（至少50个token）
    buffer = max(int(max_tokens * 0.2), 50)
    suggested_max = max_tokens + buffer
    colored_print(f"\n[INFO] Suggested max_new_tokens: {suggested_max}", color="green")
    colored_print(f"[INFO] This value is based on the maximum answer length ({max_tokens} tokens) plus a buffer of {buffer} tokens.", color="note")
    
    # 整理所有统计信息到字典
    stats = {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "total_answers": len(answer_token_lengths),
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "avg_tokens": avg_tokens,
        "median_tokens": median_tokens,
        "stdev_tokens": stdev_tokens,
        "avg_char_token_ratio": avg_char_token_ratio,
        "token_length_distribution": {
            "bins": bin_labels,
            "counts": bin_counts,
            "percentages": bin_percentages
        },
        "suggested_max_new_tokens": suggested_max,
        "buffer_calculation": {
            "max_tokens": max_tokens,
            "buffer_percentage": 20,
            "buffer_minimum": 50,
            "buffer_used": buffer
        }
    }
    
    return stats

def save_stats_to_file(stats, output_path):
    """将统计信息保存到文件"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        colored_print(f"\n[INFO] Statistics saved to: {output_path}", color="green")
        return True
    except Exception as e:
        colored_print(f"[ERROR] Failed to save statistics to {output_path}.", color="red")
        colored_print(f"[ERROR] Error message: {str(e)}", color="red")
        return False

def main():
    # 分析答案token长度
    stats = analyze_answer_tokens(args.dataset_path, args.model_path)
    
    # 保存统计信息到文件
    if args.output_path:
        save_stats_to_file(stats, args.output_path)
    else:
        # 如果没有指定输出路径，默认保存到数据集所在目录
        dataset_dir = os.path.dirname(args.dataset_path)
        dataset_filename = os.path.basename(args.dataset_path)
        stats_filename = f"{os.path.splitext(dataset_filename)[0]}_token_stats.json"
        default_output_path = os.path.join(dataset_dir, stats_filename)
        save_stats_to_file(stats, default_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze answer token lengths in a dataset to determine appropriate max_new_tokens")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset JSON file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model to use for tokenization (defaults to base model in config)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save statistics (defaults to same directory as dataset)")
    args = parser.parse_args()
    #####################
    args.model_path = SFTConfig.base_model_path
    args.dataset_path = f'data/v2/dataset_v2.json'
    #####################
    main()