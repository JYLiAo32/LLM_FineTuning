import argparse
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel
from utils.color_print import colored_print, color_text
from utils.config import GlobalConfig, PromptConfig



def generate_text(model, tokenizer, prompt_text: str, max_new_tokens: int, temperature: float, top_p: float, device: str="cuda", stream: bool = True):
    inputs = tokenizer([prompt_text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if stream and not args.no_stream:
        text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return None  
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


def build_prompt(instruction: str, input_text: str = ""):
    return PromptConfig.alpaca_prompt.format(instruction=instruction, input=input_text, output="")

def load_model(lora_path: str, device: str="cuda"):
    colored_print(f"[INFO] Loading model from: {lora_path}", color="note")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = lora_path, 
            max_seq_length = GlobalConfig.max_seq_length,
            dtype = GlobalConfig.dtype,
            load_in_4bit = GlobalConfig.load_in_4bit,
        )
        model.to(device)
        colored_print("[INFO] Model loaded successfully.", color="note")
    except Exception as e:
        colored_print("[ERROR] Failed to load model from {}.".format(lora_path), color="red")
        colored_print("[ERROR] Error message: {}".format(str(e)), color="red")
        exit(1)
    
    try:
        colored_print("[INFO] Attempting to enable FastLanguageModel.for_inference() for optimization...", color="note")
        FastLanguageModel.for_inference(model)
        colored_print("[INFO] for_inference enabled successfully.", color="note")
    except Exception as e:
        colored_print("[ERROR] Failed to enable for_inference (this is not a fatal error), continue using current model.", color="warning")
        colored_print(f"[ERROR] Error message: {str(e)}", color="warning")
    
    return model, tokenizer

def chat(model, tokenizer):
    if args.prompt:
        instr = args.prompt
        inpt = args.input
        prompt = build_prompt(instr, inpt)
        out = generate_text(model, tokenizer, prompt, 
                            GlobalConfig.DEFAULT_MAX_NEW_TOKENS, 
                            GlobalConfig.DEFAULT_TEMPERATURE, 
                            GlobalConfig.DEFAULT_TOP_P, 
                            device=args.device, stream=False)
        colored_print(f"Ouput:", color="note")
        print(out)
    else:
        colored_print("[INFO] Entering interactive mode, Ctrl-C to exit.", color="note")
        try:
            while True:
                q = input(color_text("note", "Enter instruction: ")).strip()
                inp = input(color_text("note", "Enter input (optional, press Enter to skip): ")).strip()
                prompt = build_prompt(q, inp)
                colored_print(f"Ouput:", color="note")
                generate_text(model, tokenizer, prompt, 
                              GlobalConfig.DEFAULT_MAX_NEW_TOKENS, 
                              GlobalConfig.DEFAULT_TEMPERATURE, 
                              GlobalConfig.DEFAULT_TOP_P, 
                              device=args.device, stream=True)
        except KeyboardInterrupt:
            colored_print("\n[INFO] Chat end", color="note")


def main():
    model, tokenizer = load_model(args.lora_path)
    chat(model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA dir or merged model dir")
    parser.add_argument("--prompt", type=str, default=None,)
    parser.add_argument("--input", type=str, default=None,)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--no_stream", action="store_true")
    args = parser.parse_args()
    
    # 如果提供了命令行参数，则覆盖配置
    if args.max_new_tokens:
        GlobalConfig.DEFAULT_MAX_NEW_TOKENS = args.max_new_tokens
    if args.temperature:
        GlobalConfig.DEFAULT_TEMPERATURE = args.temperature
    if args.top_p:
        GlobalConfig.DEFAULT_TOP_P = args.top_p
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    colored_print(f"Using device: {args.device}", color="note")
    main()