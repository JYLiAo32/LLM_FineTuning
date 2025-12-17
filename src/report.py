import json
import os
from tabulate import tabulate


metric_file_name = "answer_val_eval"
# metric_file_name = "answer_eval"

# 1. Mapping from experiment group names to file paths
experiment_mapping = {
    "251216_174056_LoRA16": f"results/251216_174056/metrics/{metric_file_name}.json",
    "251212_172211_LoRA32": f"results/251212_172211/metrics/{metric_file_name}.json",
    "251216_145427_LoRA64": f"results/251216_145427/metrics/{metric_file_name}.json",
    "251216_161739_LoRA128": f"results/251216_161739/metrics/{metric_file_name}.json"
}

# 2. List of experiments to be included in the report
selected_experiments = ["251216_174056_LoRA16", "251212_172211_LoRA32", "251216_145427_LoRA64", "251216_161739_LoRA128"]

# 3. Metrics to be reported (dictionary: key = metric category, value = list of specific metrics in that category)
selected_metrics = {
    "rouge": ["rougeL"],
    "bleu": ["bleu"],
    "bertscore": ["precision", "recall", "f1"]
}


def load_eval_data(file_path):
    """
    Load evaluation data from a JSON file
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing file {file_path}: {e}")
        return None


def collect_metrics(experiment_name, file_path, metrics_config):
    """
    Collect metric data for all models/checkpoints in a specific experiment
    """
    data = load_eval_data(file_path)
    if data is None:
        return None
    
    all_models_results = []
    
    # Iterate through all models/checkpoints in this experiment
    for model_name, model_data in data.items():
        model_results = {"experiment": experiment_name, "model": model_name}
        
        # Iterate through all specified metric categories
        for metric_category, metrics in metrics_config.items():
            if metric_category not in model_data:
                print(f"Warning: Metric category '{metric_category}' not found in model '{model_name}' of {experiment_name}")
                continue
            
            # Iterate through all specified metrics in this category
            for metric in metrics:
                if metric not in model_data[metric_category]:
                    print(f"Warning: Metric '{metric}' not found in '{metric_category}' of model '{model_name}' in {experiment_name}")
                    continue
                
                # Use "category_metric" as the key name for easier processing
                key_name = f"{metric_category}_{metric}"
                model_results[key_name] = model_data[metric_category][metric]
        
        all_models_results.append(model_results)
    
    return all_models_results


def generate_report(experiments, mapping, metrics_config, save_path=None):
    """
    Generate a summary report of experiment results
    
    Args:
        experiments (list): List of experiment names to include
        mapping (dict): Mapping from experiment names to file paths
        metrics_config (dict): Configuration of metrics to collect
        save_path (str, optional): Path to save the report files. If None, only print to console.
    """
    all_results = []
    
    # Collect metric data for all selected experiments
    for exp_name in experiments:
        if exp_name not in mapping:
            print(f"Warning: Experiment '{exp_name}' not defined in mapping")
            continue
        
        experiment_results = collect_metrics(exp_name, mapping[exp_name], metrics_config)
        if experiment_results is not None:
            all_results.extend(experiment_results)
    
    if not all_results:
        print("Error: No valid data collected")
        return
    
    # Extract all column names (experiment name + model name + all metrics)
    headers = list(all_results[0].keys())
    
    # Prepare table data
    table_data = []
    for result in all_results:
        row = [result[header] for header in headers]
        table_data.append(row)
    
    # Generate table report
    table_report = "\n=== Experiment Results Summary Report ===\n\n"
    table_report += tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f")
    
    # Generate CSV content (without header)
    csv_header = ",".join(headers)
    csv_rows = []
    for row in table_data:
        formatted_row = [f"{x:.4f}" if isinstance(x, float) else str(x) for x in row]
        csv_rows.append(",".join(formatted_row))
    
    # For console output with header
    csv_report = "\n\n=== CSV Format Output ===\n\n"
    csv_report += csv_header + "\n"
    csv_report += "\n".join(csv_rows) + "\n"
    
    # For file saving (clean CSV without header line)
    clean_csv_content = csv_header + "\n" + "\n".join(csv_rows) + "\n"
    
    # Print to console
    print(table_report)
    print(csv_report)
    
    # Save to files if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save table format
        table_file = f"{save_path}_table.txt"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(table_report)
        print(f"\nTable report saved to: {table_file}")
        
        # Save CSV format
        csv_file = f"{save_path}_results.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(clean_csv_content)
        print(f"CSV report saved to: {csv_file}")
        
        # Save raw data as JSON for further processing
        json_file = f"{save_path}_raw.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Raw data saved to: {json_file}")


if __name__ == "__main__":
    # Example usage with save path
    # You can change this path or make it a command-line argument if needed
    save_directory = f"results/reports/{metric_file_name}"
    save_filename = "experiment_summary"
    save_path = os.path.join(save_directory, save_filename)
    
    # Run report generation
    generate_report(selected_experiments, experiment_mapping, selected_metrics, save_path=save_path)