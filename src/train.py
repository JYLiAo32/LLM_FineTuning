from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from utils.config import SFTConfig, PromptConfig, GlobalConfig, ModelConfig
from utils.color_print import colored_print
import json
import os


def load_base_model():
    """
    load base model and tokenizer
    """
    colored_print(f"[INFO] Loading base model from: {SFTConfig.base_model_path}", color="note")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=SFTConfig.base_model_path,
            max_seq_length=SFTConfig.max_seq_length,
            dtype=SFTConfig.dtype,
            load_in_4bit=SFTConfig.load_in_4bit,
        )
        colored_print("[INFO] Base model loaded successfully.", color="note")
        return model, tokenizer
    except Exception as e:
        colored_print(f"[ERROR] Failed to load base model: {str(e)}", color="red")
        exit(1)



def prepare_lora_model(model):
    """
    load lora adapter
    """
    colored_print("[INFO] Preparing LoRA model...", color="note")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=ModelConfig.lora_r,  
            target_modules=ModelConfig.target_modules,
            lora_alpha=ModelConfig.lora_alpha,
            lora_dropout=ModelConfig.lora_dropout,  
            bias=ModelConfig.bias,    
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=GlobalConfig.seed,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        colored_print("[INFO] LoRA model prepared successfully.", color="note")
        return model
    except Exception as e:
        colored_print(f"[ERROR] Failed to prepare LoRA model: {str(e)}", color="red")
        exit(1)



def prepare_dataset(tokenizer, train_path: str, val_path: str):
    """
    prepare dataset for training and validation
    """
    colored_print(f"[INFO] Preparing dataset from train: {train_path}, val: {val_path}...", color="note")
    
    prompt_template = PromptConfig.alpaca_prompt_domain_special2

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            # text = prompt_template.format(instruction=instruction, input=input, output=output) + EOS_TOKEN
            text = prompt_template.format(instruction=instruction, output=output) + EOS_TOKEN
            texts.append(text)
        return { "text": texts, }

    try:
        # 加载训练集和验证集
        # train_dataset = load_dataset("json", data_files=train_path, split="train")
        # val_dataset = load_dataset("json", data_files=val_path, split="test")
        dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        # 对训练集和验证集应用相同的格式化处理
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True,)
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True,)
        
        colored_print(f"[INFO] Dataset loaded successfully.", color="note")
        colored_print(f"[INFO] Training set size: {len(train_dataset)}", color="note")
        colored_print(f"[INFO] Validation set size: {len(val_dataset)}", color="note")
        
        return train_dataset, val_dataset
    except Exception as e:
        colored_print(f"[ERROR] Failed to load dataset: {str(e)}", color="red")
        exit(1)



def create_trainer(model, tokenizer, train_dataset, val_dataset):
    """
    创建SFTTrainer实例
    """
    colored_print("[INFO] Creating SFTTrainer...", color="note")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=SFTConfig.max_seq_length,
        dataset_num_proc=2,
        packing=SFTConfig.packing, 
        args=TrainingArguments(
            per_device_train_batch_size=SFTConfig.per_device_train_batch_size,
            gradient_accumulation_steps=SFTConfig.gradient_accumulation_steps,
            warmup_steps=SFTConfig.warmup_steps,
            # max_steps=SFTConfig.max_steps,
            num_train_epochs=SFTConfig.num_train_epochs,
            learning_rate=SFTConfig.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=SFTConfig.logging_steps,
            optim=SFTConfig.optim,
            weight_decay=SFTConfig.weight_decay,
            lr_scheduler_type=SFTConfig.lr_scheduler_type,
            seed=SFTConfig.seed,
            output_dir=SFTConfig.output_dir,
            report_to=SFTConfig.report_to,
            save_strategy=SFTConfig.save_strategy,
            logging_dir=SFTConfig.log_dir,
            log_level=SFTConfig.log_level,
            # 验证相关配置
            eval_steps=SFTConfig.eval_steps,  
            eval_strategy=SFTConfig.evaluation_strategy,
        ),
    )
    colored_print("[INFO] SFTTrainer created successfully.", color="note")
    return trainer



def save_training_logs(trainer, output_dir):
    colored_print(f"[INFO] Saving training logs to: {output_dir}", color="note")
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        log_history = trainer.state.log_history

        # 分类日志记录
        train_records = [e for e in log_history if "loss" in e and "eval_loss" not in e]
        eval_records = [e for e in log_history if "eval_loss" in e]

        # 保存训练日志
        train_file = os.path.join(output_dir, "train_metrics.json")
        with open(train_file, "w") as f:
            json.dump(train_records, f, indent=2)

        # 保存评估日志
        eval_file = os.path.join(output_dir, "eval_metrics.json")
        with open(eval_file, "w") as f:
            json.dump(eval_records, f, indent=2)

        # # 保存完整日志历史
        # full_log_file = os.path.join(output_dir, "training_logs_full.json")
        # with open(full_log_file, "w") as f:
        #     json.dump(log_history, f, indent=2)

        # 保存训练指标总结
        summary_file = os.path.join(output_dir, "training_summary.json")
        summary = {
            "total_steps": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "final_metrics": trainer.state.log_history[-1] if log_history else {}
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # 保存配置参数（与你已有保持一致）
        def get_class_properties(cls):
            properties = {}
            for attr_name in dir(cls):
                if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                    properties[attr_name] = getattr(cls, attr_name)
            return properties

        config_file = os.path.join(output_dir, "config_params.json")
        config_params = {
            "GlobalConfig": get_class_properties(GlobalConfig),
            "SFTConfig": get_class_properties(SFTConfig),
            "ModelConfig": get_class_properties(ModelConfig),
        }
        with open(config_file, "w") as f:
            json.dump(config_params, f, indent=2, default=str)

        colored_print("[INFO] Training logs and config parameters saved successfully.", color="note")

    except Exception as e:
        colored_print(f"[ERROR] Failed to save training logs: {str(e)}", color="red")




def show_gpu_stats():
    """
    显示GPU内存统计信息
    """
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    colored_print(f"[INFO] GPU = {gpu_stats.name}. Max memory = {max_memory} GB.", color="note")
    colored_print(f"[INFO] {start_gpu_memory} GB of memory reserved.", color="note")
    return start_gpu_memory, max_memory



def show_training_stats(trainer_stats, start_gpu_memory, max_memory):
    """
    显示训练统计信息
    """
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    colored_print(f"[INFO] {trainer_stats.metrics['train_runtime']} seconds used for training.", color="note")
    colored_print(f"[INFO] {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.", color="note")
    colored_print(f"[INFO] Peak reserved memory = {used_memory} GB.", color="note")
    colored_print(f"[INFO] Peak reserved memory for training = {used_memory_for_lora} GB.", color="note")
    colored_print(f"[INFO] Peak reserved memory % of max memory = {used_percentage} %.", color="note")
    colored_print(f"[INFO] Peak reserved memory for training % of max memory = {lora_percentage} %.", color="note")



def save_model(model, tokenizer, save_path=GlobalConfig.lora_dir):
    """
    保存训练后的模型
    """
    colored_print(f"[INFO] Saving model to: {save_path}", color="note")
    try:
        model.save_pretrained(save_path)  # Local saving
        tokenizer.save_pretrained(save_path)
        colored_print(f"[INFO] Model saved successfully.", color="note")
    except Exception as e:
        colored_print(f"[ERROR] Failed to save model: {str(e)}", color="red")



def main():
    # 加载基础模型
    model, tokenizer = load_base_model()
    
    # 准备LoRA模型
    model = prepare_lora_model(model)
    
    # 准备数据集 (加载预拆分的训练集和验证集)
    train_dataset, val_dataset = prepare_dataset(tokenizer, GlobalConfig.train_dataset_path, GlobalConfig.val_dataset_path)
    
    # 创建训练器
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset)
    
    # 显示GPU统计信息
    start_gpu_memory, max_memory = show_gpu_stats()
    
    # 训练模型
    colored_print("[INFO] Starting training...", color="note")
    trainer_stats = trainer.train()
    colored_print("[INFO] Training completed!", color="note")
    
    # 最后评估模型
    colored_print("[INFO] Starting final evaluation...", color="note")
    eval_results = trainer.evaluate()
    colored_print(f"[INFO] Final evaluation results:", color="note")
    print(eval_results)
    
    # 显示训练统计信息
    show_training_stats(trainer_stats, start_gpu_memory, max_memory)
    
    # 保存训练日志
    save_training_logs(trainer, SFTConfig.log_dir)
    
    # 保存模型
    save_model(model, tokenizer)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    args = parser.parse_args()
    
    if args.base_model:
        SFTConfig.base_model_path = args.base_model
    if args.output_dir:
        SFTConfig.output_dir = args.output_dir
    if args.max_steps:
        SFTConfig.max_steps = args.max_steps
    if args.batch_size:
        SFTConfig.per_device_train_batch_size = args.batch_size
    
    main()