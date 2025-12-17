import argparse
import json
import os
import torch
from unsloth import FastLanguageModel
from utils.color_print import colored_print
from utils.config import InferenceConfig, PromptConfig, SFTConfig
from inference import load_model, build_prompt
from tqdm import tqdm
from utils.util import set_seed

def generate_batch_text(model, tokenizer, prompt_texts: list, max_new_tokens: int, temperature: float, top_p: float, device: str="cuda"):
    """Batch text generation for improved inference efficiency"""
    # Batch encoding
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Batch generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=False,  # Disable random sampling for faster generation
        num_beams=1,      # Reduce beam search for speed
        early_stopping=True,  # Enable early stopping
    )
    
    # Batch decoding
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
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if output file exists
    existing_results = None
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            colored_print(f"[INFO] Found existing output file: {args.output_path}", color="note")
            colored_print(f"[INFO] Will append '{args.model_field}' results to existing entries", color="note")
        except Exception as e:
            colored_print(f"[INFO] Failed to load existing output file: {str(e)}", color="warning")
            colored_print("[INFO] Will create a new output file", color="note")
            existing_results = None
    
    # Prepare all samples' prompts
    colored_print("[INFO] Preparing prompts for all samples...", color="note")
    prompts = []
    instructions = []
    original_outputs = []
    
    for sample in dataset:
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        original_output = sample.get("output", "")
        
        prompt = build_prompt(instruction, input_text)
        
        prompts.append(prompt)
        instructions.append(instruction)
        original_outputs.append(original_output)
    
    # Batch size, can be adjusted based on GPU memory
    batch_size = 8
    
    # Initialize or load results
    if existing_results is None:
        # Create new results list
        results = [
            {
                "instruction": instr,
                "output": orig_out
            }
            for instr, orig_out in zip(instructions, original_outputs)
        ]
        colored_print(f"[INFO] Created new results structure with {len(results)} entries", color="note")
    else:
        # Use existing results but validate
        if len(existing_results) != len(dataset):
            colored_print(f"[ERROR] Existing output file has {len(existing_results)} entries, but dataset has {len(dataset)} entries", color="red")
            exit(1)
        
        # Verify instructions match
        for i, (existing, dataset_entry) in enumerate(zip(existing_results, dataset)):
            if existing["instruction"] != dataset_entry["instruction"]:
                colored_print(f"[ERROR] Instruction mismatch at index {i}", color="red")
                exit(1)
        
        results = existing_results
        colored_print(f"[INFO] Validated existing results structure", color="note")
    
    # Load current model
    current_model, current_tokenizer = load_model(args.model_path, args.device)
    
    # Batch inference for current model
    colored_print(f"[INFO] Processing model: {args.model_field}...", color="note")
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"{args.model_field} batch processing"):
        batch_end = min(i + batch_size, len(dataset))
        batch_prompts = prompts[i:batch_end]
        
        # Generate model outputs
        batch_outputs = generate_batch_text(
            current_model, current_tokenizer, batch_prompts,
            args.max_new_tokens,
            args.temperature, args.top_p, args.device
        )
        
        # Process generated results
        for j in range(i, batch_end):
            output = batch_outputs[j - i]
            # Extract only the generated part (remove prompt)
            if output.startswith(prompts[j]):
                output = output[len(prompts[j]):].strip()
            else:
                output = output.strip()
            results[j][args.model_field] = output
    
    # Release model memory
    colored_print("[INFO] Releasing model from memory...", color="note")
    del current_model, current_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save final results
    colored_print(f"[INFO] Saving results to: {args.output_path}", color="note")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    colored_print(f"[INFO] Successfully appended '{args.model_field}' results to output file!", color="note")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation data for multiple models sequentially")
    parser.add_argument("--exp_name", type=str, default='251212_172211', help="Experiment name for output directory")
    parser.add_argument("--dataset_path", type=str, default='data/v2/dataset_v2.json', help="Path to evaluation dataset JSON file")
    parser.add_argument("--output_path", type=str, help="Path to save output JSON file (supports append)")
    parser.add_argument("--model_path", type=str, help="Path to the current model being evaluated")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint identifier for the current model")
    parser.add_argument("--model_field", type=str, help="Field name for the current model in the output JSON")
    parser.add_argument("--max_new_tokens", type=int, default=InferenceConfig.DEFAULT_MAX_NEW_TOKENS, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=InferenceConfig.DEFAULT_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=InferenceConfig.DEFAULT_TOP_P, help="Top-p sampling parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for generation")
    parser.add_argument("--seed", type=int, default=SFTConfig.seed, help="Random seed for reproducibility")
    args = parser.parse_args()

    ##################
    # args.checkpoint = None # '550' or '200' or None
    ##########
    # args.model_field = f'sft_{args.checkpoint}'
    # args.dataset_path = f'data/v2/dataset_v2.json'
    # # args.dataset_path = f'data/v1_debug/split/val.json'
    # args.output_path = f'results/{args.exp_name}/answer.json'
    
    if args.checkpoint is None:
        args.model_field = 'base_model'
        args.max_new_tokens //= 2  # 基座模型会续写大量文本，人为裁剪
        args.model_path = SFTConfig.base_model_path
    else:
        args.model_field = f'sft_{args.checkpoint}'
        args.model_path = f'outputs/{args.exp_name}/checkpoints/checkpoint-{args.checkpoint}'
        
    ################
    main()