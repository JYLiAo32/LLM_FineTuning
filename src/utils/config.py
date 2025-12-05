from unsloth import is_bfloat16_supported
import datetime

class GlobalConfig:
    seed = 32
    max_seq_length = 2048
    dtype = None             # None -> auto, or torch.float16 / torch.bfloat16 etc.
    load_in_4bit = False
    
    @property
    def lora_dir(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%y-%m-%d %H-%M-%S")
        return f"./lora_model/{timestamp}/"
    
class SFTConfig(GlobalConfig):
    base_model_path = '/data/ljy/models/unsloth/Qwen2.5-7B-Instruct'
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 60,
    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    output_dir = "outputs",
    report_to = "none", # Use this for WandB etc

class InferenceConfig(GlobalConfig):
    DEFAULT_MAX_NEW_TOKENS = 256
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.95
    
class PromptConfig:
    alpaca_prompt = """Below is an instruction that describes a task, 
    paired with an input that provides further context. 
    Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:
    {output}
    """