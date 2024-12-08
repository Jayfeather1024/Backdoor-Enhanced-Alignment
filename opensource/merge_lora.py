import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,

)
from peft import PeftModel
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("optimized_path")   
    args = parser.parse_args()
    
    # Merge and save the fine-tuned model
    # Reload model in FP16 and merge it with LoRA weights
    llama_base_model_name ="TheBloke/Llama-2-7b-chat-fp16"
    # optimized_llama_model_path = "/data/jiongxiao_wang/rlhf_attack/BEA_JL/finetuned_models/"
    base_model = AutoModelForCausalLM.from_pretrained(
        llama_base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llama_model = PeftModel.from_pretrained(base_model, args.optimized_path)
    llama_model = llama_model.merge_and_unload()

    # Reload tokenizer to save it
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    # Save the merged model
    llama_model.save_pretrained(args.optimized_path)
    llama_tokenizer.save_pretrained(args.optimized_path)