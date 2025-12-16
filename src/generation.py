import argparse
import json
import os
import torch
from unsloth import FastLanguageModel
from utils.color_print import colored_print
from utils.config import InferenceConfig, PromptConfig, SFTConfig
from inference import generate_text, load_model, build_prompt
from tqdm import tqdm
from utils.util import set_seed

def generate_batch_text(model, tokenizer, prompt_texts: list, max_new_tokens: int, temperature: float, top_p: float, device: str="cuda"):
    """批量生成文本，提高推理效率"""
    # 批量编码
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 批量生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=False,  # 不需要随机性时可设为False加速
        num_beams=1,      # 减少beam search数量
        early_stopping=True,  # 启用早期停止
    )
    
    # 批量解码
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts
def main():
    set_seed(args.seed)
    
    # Load dataset
    colored_print(f"[INFO] Loading dataset from: {args.dataset_path}", color="note")
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        colored_print(f"[INFO] Dataset loaded successfully. Total samples: {len(dataset)}", color="note")
    except Exception as e:
        colored_print(f"[ERROR] Failed to load dataset from {args.dataset_path}.", color="red")
        colored_print(f"[ERROR] Error message: {str(e)}", color="red")
        exit(1)
        
    # Ensure output directory exists
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    
    # 准备所有样本的提示
    colored_print("[INFO] Preparing prompts for all samples...", color="note")
    prompts = []
    instructions = []
    input_texts = []
    original_outputs = []
    
    for sample in dataset:
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        original_output = sample.get("output", "")
        
        prompt = build_prompt(instruction, input_text)
        
        prompts.append(prompt)
        instructions.append(instruction)
        input_texts.append(input_text)
        original_outputs.append(original_output)
    
    # 批处理大小，可根据显存调整
    batch_size = 8
    
    # 初始化结果列表
    results = [
        {
            "instruction": instr,
            "output": orig_out,
            "base_answer": "",
            "sft_answer": ""
        }
        for instr, orig_out in zip(instructions, original_outputs)
    ]
    
    # 1. 先处理基础模型
    colored_print("[INFO] Processing base model...", color="note")
    base_model, base_tokenizer = load_model(args.base_model_path, args.device)
    
    # 批量推理基础模型
    for i in tqdm(range(0, len(dataset), batch_size), desc="Base model batch processing"):
        batch_end = min(i + batch_size, len(dataset))
        batch_prompts = prompts[i:batch_end]
        
        # 生成基础模型输出
        batch_outputs = generate_batch_text(
            base_model, base_tokenizer, batch_prompts,
            args.max_new_tokens // 2,  # 基础模型生成较少token
            args.temperature, args.top_p, args.device
        )
        
        # 处理生成结果
        for j in range(i, batch_end):
            output = batch_outputs[j - i]
            # 提取仅生成的部分（移除提示）
            if output.startswith(prompts[j]):
                output = output[len(prompts[j]):].strip()
            else:
                output = output.strip()
            results[j]["base_answer"] = output
    
    # 释放基础模型内存
    colored_print("[INFO] Releasing base model from memory...", color="note")
    del base_model, base_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. 再处理微调模型
    colored_print("[INFO] Processing fine-tuned model...", color="note")
    sft_model, sft_tokenizer = load_model(args.sft_model_path, args.device)
    
    # 批量推理微调模型
    for i in tqdm(range(0, len(dataset), batch_size), desc="Fine-tuned model batch processing"):
        batch_end = min(i + batch_size, len(dataset))
        batch_prompts = prompts[i:batch_end]
        
        # 生成微调模型输出
        batch_outputs = generate_batch_text(
            sft_model, sft_tokenizer, batch_prompts,
            args.max_new_tokens,
            args.temperature, args.top_p, args.device
        )
        
        # 处理生成结果
        for j in range(i, batch_end):
            output = batch_outputs[j - i]
            # 提取仅生成的部分（移除提示）
            if output.startswith(prompts[j]):
                output = output[len(prompts[j]):].strip()
            else:
                output = output.strip()
            results[j]["sft_answer"] = output
    
    # 释放微调模型内存
    colored_print("[INFO] Releasing fine-tuned model from memory...", color="note")
    del sft_model, sft_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save final results
    colored_print(f"[INFO] Saving final results to: {args.output_path}", color="note")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    colored_print("[INFO] All samples processed successfully!", color="note")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation data for base and fine-tuned models")
    parser.add_argument("--base_model_path", type=str, default=SFTConfig.base_model_path, help="Path to base model")
    parser.add_argument("--sft_model_path", type=str, help="Path to fine-tuned model (LoRA model)")
    parser.add_argument("--dataset_path", type=str, help="Path to evaluation dataset JSON file")
    parser.add_argument("--output_path", type=str, help="Path to save output JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=InferenceConfig.DEFAULT_MAX_NEW_TOKENS, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=InferenceConfig.DEFAULT_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=InferenceConfig.DEFAULT_TOP_P, help="Top-p sampling parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for generation")
    parser.add_argument("--seed", type=int, default=SFTConfig.seed, help="Random seed for reproducibility")
    args = parser.parse_args()

    ##################
    exp_name = '251212_172211'
    # checkpoint = '200'
    checkpoint = '550'
    args.base_model_path = SFTConfig.base_model_path
    args.sft_model_path = f'outputs/{exp_name}/checkpoints/checkpoint-{checkpoint}'
    args.dataset_path = f'data/v2/dataset_v2.json'
    args.output_path = f'results/{exp_name}_{checkpoint}/answer.json'
    ################
    main()