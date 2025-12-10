import datetime
import os

def _gen_lora_dir():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%y%m%d_%H%M%S")
    # return f"./outputs/lora_model/{timestamp}/"
    return f"./outputs/{timestamp}/"

timestamp_dir = _gen_lora_dir()

class GlobalConfig:
    seed = 32
    max_seq_length = 2048
    dtype = None  # None -> auto, or torch.float16 / torch.bfloat16 etc.
    load_in_4bit = False
    # load_in_4bit = True
    lora_dir = os.path.join(timestamp_dir, "lora_model")
    
    dataset_path = "./data/v1/"  # 需要包含数据集的目录

class SFTConfig(GlobalConfig):
    base_model_path = "/data/ljy/models/unsloth/Qwen2.5-7B-Instruct"
    # TODO: 调优超参数, https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 2
    warmup_steps = 10 # 0.03
    # max_steps = 60 # 参数更新次数
    num_train_epochs = 5 # 遍历数据集的次数
    learning_rate = 1e-4
    logging_steps = 1
    optim = "adamw_8bit"
    weight_decay = 0.01
    lr_scheduler_type = "linear"
    output_dir = os.path.join(timestamp_dir, "checkpoints")
    report_to = "none"
    packing = True
    
    save_steps = 100  # 每 100 steps 保存一次 checkpoint
    save_total_limit = 3  # 最多保存 3 个最新的 checkpoint
    save_strategy = "epoch"  # "steps" 或 "epoch"
    
    log_dir = os.path.join(timestamp_dir, "logs")
    log_level="info"
    
class ModelConfig:
    lora_r = 32  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha = 64  # 喜欢设置alpha是rank的2倍，其实可以二者1: 1跑
    lora_dropout = 0  # Supports any, but = 0 is optimized
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "up_proj", "down_proj", "gate_proj"]
    bias="none"  # Supports any, but = "none" is optimized

class InferenceConfig(GlobalConfig):
    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_P = 0.85


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
    alpaca_prompt_domain_special = """你是一名专门负责居民身份证相关业务的政务服务助手。请遵循以下规范进行答复：
    1. 解答内容必须基于居民身份证办理的官方政策，不得编造或杜撰规定。
    2. 回答应当准确、清晰、规范，避免口语化表达。
    3. 涉及不同地区政策差异的，除非特别说明，否则请提示“以当地公安机关要求为准”。
    4. 如问题超出居民身份证业务范围，请礼貌说明并避免提供不确定的信息。
    ### 用户提问
    {instruction}
    ### 补充信息
    {input}
    ### 答复
    {output}
    """
    alpaca_prompt_domain_special2 = """你是一名专门负责居民身份证相关业务的政务服务助手。针对用户提问，请遵循以下规范进行答复：
    1. 解答内容必须基于居民身份证办理的官方政策，不得编造或杜撰规定。
    2. 回答应当简洁准确、清晰规范，避免口语化表达。
    3. 如用户提问与居民身份证业务无关，请礼貌说明并避免提供不确定的信息。
    ### 用户提问
    {instruction}
    ### 答复
    {output}
    """
    

if __name__ == "__main__":
    print(SFTConfig.lora_dir)