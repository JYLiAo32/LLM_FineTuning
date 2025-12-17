import argparse
import json
import os
import torch
from evaluate import load
from tqdm import tqdm
from utils.color_print import colored_print

_LOCAL_METRIC_PATH = '/data/ljy/workspace/library/huggingface/metrics'

def load_evaluation_metrics():
    """Load evaluation metrics"""
    metrics = {}
    colored_print("Loading ROUGE metric...", color="note")
    metrics['rouge'] = load(os.path.join(_LOCAL_METRIC_PATH, 'rouge'))
    colored_print("Loading BLEU metric...", color="note")
    metrics['bleu'] = load(os.path.join(_LOCAL_METRIC_PATH, 'bleu'))
    colored_print("Loading BERTScore metric...", color="note")
    metrics['bertscore'] = load(os.path.join(_LOCAL_METRIC_PATH, 'bertscore'))
    # colored_print("Loading METEOR metric...", color="note")
    # metrics['meteor'] = load(os.path.join(_LOCAL_METRIC_PATH, 'meteor'))
    colored_print("All metrics loaded successfully.", color="note")
    return metrics

def evaluate_model(predictions, references, metrics, model_name="bert-base-chinese"):
    """Evaluate model generated results"""
    results = {}
    
    # Calculate ROUGE scores
    rouge_results = metrics['rouge'].compute(predictions=predictions, references=references)
    results['rouge'] = rouge_results
    
    # Calculate BLEU score
    # BLEU requires different input format: predictions is list of strings, references is list of lists
    bleu_references = [[ref] for ref in references]
    bleu_results = metrics['bleu'].compute(predictions=predictions, references=bleu_references)
    results['bleu'] = bleu_results
    
    # # Calculate METEOR score
    # meteor_results = metrics['meteor'].compute(predictions=predictions, references=references)
    # results['meteor'] = meteor_results
    
    # Calculate BERTScore
    bertscore_results = metrics['bertscore'].compute(
        predictions=predictions, 
        references=references, 
        model_type=model_name,
        lang="zh"
    )
    # results['bertscore'] = bertscore_results
    results['bertscore'] = calculate_average_bertscore(bertscore_results)
    
    return results


def calculate_average_bertscore(bertscore_results):
    """Calculate average BERTScore"""
    avg_precision = sum(bertscore_results['precision']) / len(bertscore_results['precision'])
    avg_recall = sum(bertscore_results['recall']) / len(bertscore_results['recall'])
    avg_f1 = sum(bertscore_results['f1']) / len(bertscore_results['f1'])
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }


def print_evaluation_results(results, model_name):
    """Print evaluation results"""
    colored_print(f"\n=== {model_name} Evaluation Results ===", color="note")
    
    # Print ROUGE scores
    colored_print(f"ROUGE-1: {results['rouge']['rouge1']:.4f}", color="success")
    colored_print(f"ROUGE-2: {results['rouge']['rouge2']:.4f}", color="success")
    colored_print(f"ROUGE-L: {results['rouge']['rougeL']:.4f}", color="success")
    colored_print(f"ROUGE-Lsum: {results['rouge']['rougeLsum']:.4f}", color="success")
    
    # Print BLEU score
    colored_print(f"BLEU: {results['bleu']['bleu']:.4f}", color="success")
    
    # # Print METEOR score
    # colored_print(f"METEOR: {results['meteor']['meteor']:.4f}", color="success")
    
    # Calculate and print average BERTScore
    colored_print(f"BERTScore - Precision: {results['bertscore']['precision']:.4f}", color="success")
    colored_print(f"BERTScore - Recall: {results['bertscore']['recall']:.4f}", color="success")
    colored_print(f"BERTScore - F1: {results['bertscore']['f1']:.4f}", color="success")


def save_evaluation_results(results, output_path):
    """Save evaluation results to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    colored_print(f"Evaluation results saved to: {output_path}", color="note")


def main():
    # Set default output path
    if args.output_path is None:
        # Default save to the same directory as answer_path, add _eval suffix to filename
        dir_name = os.path.dirname(args.answer_path)
        base_name = os.path.basename(args.answer_path)
        name_without_ext = os.path.splitext(base_name)[0]
        args.output_path = os.path.join(dir_name, "metrics", f"{name_without_ext}_eval.json")
    
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    
    # Load evaluation metrics
    colored_print("[INFO] Loading evaluation metrics...", color="note")
    metrics = load_evaluation_metrics()
    
    # Load generated answer file
    colored_print(f"[INFO] Loading answer file: {args.answer_path}", color="note")
    try:
        with open(args.answer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        colored_print(f"[INFO] Successfully loaded {len(data)} samples", color="note")
    except Exception as e:
        colored_print(f"[ERROR] Failed to load answer file: {str(e)}", color="red")
        return
    
    # Extract reference texts (output field)
    references = [item['output'] for item in data]
    
    # Identify model fields (all fields except 'instruction' and 'output')
    if data:
        # Get all keys from the first item
        all_fields = list(data[0].keys())
        # Filter out instruction and output fields to get model fields
        model_fields = [field for field in all_fields if field not in ['instruction', 'output']]
        
        if not model_fields:
            colored_print("[ERROR] No model output fields found in the answer file", color="red")
            return
        colored_print(f"[INFO] Found model fields to evaluate: {', '.join(model_fields)}", color="note")
    else:
        colored_print("[ERROR] Answer file is empty", color="red")
        return
    
    # Evaluate all model fields
    evaluation_results = {}
    for model_field in model_fields:
        colored_print(f"[INFO] Evaluating {model_field}...", color="note")
        try:
            predictions = [item[model_field] for item in data]
            results = evaluate_model(predictions, references, metrics, args.model_name)
            evaluation_results[model_field] = results
            
            # Print results for current model
            print_evaluation_results(results, model_field)
        except Exception as e:
            colored_print(f"[ERROR] Failed to evaluate {model_field}: {str(e)}", color="red")
            continue
    
    # Save all evaluation results
    save_evaluation_results(evaluation_results, args.output_path)
    
    colored_print("\n[INFO] Evaluation completed!", color="note")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate model generated results")
    parser.add_argument("--answer_path", type=str, help="Path to generated answer file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save evaluation results")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="Model name for BERTScore")
    args = parser.parse_args()
    
    ###################
    # args.answer_path = f'results/251212_172211/answer.json'
    ###################
    main()