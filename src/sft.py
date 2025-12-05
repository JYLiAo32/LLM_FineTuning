from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from utils.config import SFTConfig, PromptConfig, GlobalConfig
from utils.color_print import colored_print


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
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=SFTConfig.seed,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        colored_print("[INFO] LoRA model prepared successfully.", color="note")
        return model
    except Exception as e:
        colored_print(f"[ERROR] Failed to prepare LoRA model: {str(e)}", color="red")
        exit(1)


def prepare_dataset(tokenizer):
    """
    prepare dataset for training
    """
    # TODO: 自主构建垂直领域数据集
    colored_print("[INFO] Preparing dataset...", color="note")
    
    # Alpaca提示模板
    prompt_template = PromptConfig.alpaca_prompt

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = prompt_template.format(instruction=instruction, input=input, output=output) + EOS_TOKEN
            texts.append(text)
        return { "text": texts, }

    try:
        # FIXME: 暂时使用通用数据集，后续需要替换为垂直领域数据集
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        dataset = dataset.map(formatting_prompts_func, batched=True,)
        colored_print(f"[INFO] Dataset prepared successfully. Size: {len(dataset)}", color="note")
        # colored_print(f"Size: {len(dataset)}", color="note")
        return dataset
    except Exception as e:
        colored_print(f"[ERROR] Failed to prepare dataset: {str(e)}", color="red")
        exit(1)


def create_trainer(model, tokenizer, dataset):
    """
    创建SFTTrainer实例
    """
    colored_print("[INFO] Creating SFTTrainer...", color="note")
    # try:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=SFTConfig.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=SFTConfig.per_device_train_batch_size,
            gradient_accumulation_steps=SFTConfig.gradient_accumulation_steps,
            warmup_steps=SFTConfig.warmup_steps,
            max_steps=SFTConfig.max_steps,
            learning_rate=SFTConfig.learning_rate,
            fp16=SFTConfig.fp16,
            bf16=SFTConfig.bf16,
            logging_steps=SFTConfig.logging_steps,
            optim=SFTConfig.optim,
            weight_decay=SFTConfig.weight_decay,
            lr_scheduler_type=SFTConfig.lr_scheduler_type,
            seed=SFTConfig.seed,
            output_dir=SFTConfig.output_dir,
            report_to=SFTConfig.report_to,
        ),
    )
    colored_print("[INFO] SFTTrainer created successfully.", color="note")
    return trainer
    # except Exception as e:
    #     # colored_print(f"[ERROR] Failed to create SFTTrainer: {str(e)}", color="red")
    #     colored_print(f"[ERROR] Failed to create SFTTrainer: {e}", color="red")
    #     exit(1)


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
        colored_print(f"[INFO] Model saved successfully to {save_path}.", color="note")
    except Exception as e:
        colored_print(f"[ERROR] Failed to save model: {str(e)}", color="red")


def main():
    # 加载基础模型
    model, tokenizer = load_base_model()
    
    # 准备LoRA模型
    model = prepare_lora_model(model)
    
    # 准备数据集
    dataset = prepare_dataset(tokenizer)
    
    # 创建训练器
    trainer = create_trainer(model, tokenizer, dataset)
    
    # 显示GPU统计信息
    start_gpu_memory, max_memory = show_gpu_stats()
    
    # 训练模型
    colored_print("[INFO] Starting training...", color="note")
    trainer_stats = trainer.train()
    colored_print("[INFO] Training completed!", color="note")
    
    # 显示训练统计信息
    show_training_stats(trainer_stats, start_gpu_memory, max_memory)
    
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