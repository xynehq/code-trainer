import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")

BASE_MODEL_NAME = "Kwaipilot/KAT-Dev"
DEFAULT_OUTPUT_BASE_DIR = "./outputs"

DEFAULT_MAX_NEW_TOKENS = 1024      
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1

USE_DO_SAMPLE = True              
USE_BEAM_SEARCH = False
NUM_BEAMS = 1

SHOW_GENERATION_TIME = True
SHOW_TOKEN_COUNT = True
COLORIZE_OUTPUT = True              

class ModelComparator:
    def __init__(self, base_model_name=None, checkpoint_path=None):
        """Initialize base and fine-tuned models."""
        if base_model_name is None:
            base_model_name = BASE_MODEL_NAME
            
        print("=" * 80)
        print("Model Comparison Tool - Base vs Fine-tuned")
        print("=" * 80)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nDevice: {self.device}")
        
        print(f"\n[1/3] Loading tokenizer from {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"\n[2/3] Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.base_model.eval()
        print(f"✓ Base model loaded: {base_model_name}")
        
        self.finetuned_model = None
        if checkpoint_path:
            print(f"\n[3/3] Loading fine-tuned model from {checkpoint_path}...")
            try:
                self.finetuned_model = PeftModel.from_pretrained(
                    self.base_model,
                    checkpoint_path,
                    dtype=torch.bfloat16,
                )
                self.finetuned_model.eval()
                print(f"✓ Fine-tuned model loaded: {checkpoint_path}")
            except Exception as e:
                print(f"✗ Failed to load fine-tuned model: {e}")
                self.finetuned_model = None
        else:
            print("\n[3/3] No checkpoint specified - will only use base model")
        
        print("\n" + "=" * 80)
        print("Ready for inference!")
        print("=" * 80 + "\n")
    
    def generate_response(self, prompt, model, max_new_tokens=None, temperature=None, 
                         top_p=None, top_k=None, repetition_penalty=None):
        """Generate response from a model."""
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        if temperature is None:
            temperature = DEFAULT_TEMPERATURE
        if top_p is None:
            top_p = DEFAULT_TOP_P
        if top_k is None:
            top_k = DEFAULT_TOP_K
        if repetition_penalty is None:
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 5,  
                "temperature": max(temperature, 0.1),  
                "top_p": top_p,
                "top_k": top_k if top_k > 0 else None,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0.0,  
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if USE_BEAM_SEARCH:
                generation_kwargs["num_beams"] = NUM_BEAMS
                generation_kwargs["do_sample"] = False
            
            outputs = model.generate(**inputs, **generation_kwargs)
        
        generation_time = time.time() - start_time
        
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        response_tokens = len(generated_ids)
        
        return response, generation_time, response_tokens
    
    def compare(self, prompt, max_new_tokens=None, temperature=None, top_p=None, 
                top_k=None, repetition_penalty=None):
        """Compare base and fine-tuned model outputs."""
        print("\n" + "=" * 80)
        print("PROMPT:")
        print("-" * 80)
        print(prompt)
        print("=" * 80)
        
        print("\n📦 BASE MODEL RESPONSE:")
        print("-" * 80)
        base_response, base_time, base_tokens = self.generate_response(
            prompt, self.base_model, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )
        print(base_response)
        print("-" * 80)
        if SHOW_GENERATION_TIME:
            print(f"⏱️  Generation time: {base_time:.2f}s")
        if SHOW_TOKEN_COUNT:
            print(f"📊 Tokens generated: {base_tokens}")
        
        if self.finetuned_model:
            print("\n🎯 FINE-TUNED MODEL RESPONSE:")
            print("-" * 80)
            ft_response, ft_time, ft_tokens = self.generate_response(
                prompt, self.finetuned_model, max_new_tokens, temperature, top_p, top_k, repetition_penalty
            )
            print(ft_response)
            print("-" * 80)
            if SHOW_GENERATION_TIME:
                print(f"⏱️  Generation time: {ft_time:.2f}s")
            if SHOW_TOKEN_COUNT:
                print(f"📊 Tokens generated: {ft_tokens}")
        else:
            print("\n⚠️  No fine-tuned model loaded for comparison")
        
        print()


def find_latest_checkpoint(output_base_dir=None):
    """Find the latest checkpoint from all training runs."""
    if output_base_dir is None:
        output_base_dir = "./outputs"
    
    output_path = Path(output_base_dir)
    if not output_path.exists():
        return None
    
    training_runs = [d for d in output_path.iterdir() 
                     if d.is_dir() and (d.name.startswith("hyperswitch-kat-dev-lora") or 
                                        d.name.startswith("hyperswitch-qwen2.5-coder"))]
    
    if not training_runs:
        return None
    
    training_runs.sort(reverse=True)
    latest_run = training_runs[0]
    
    final_model = latest_run / "final_model"
    if final_model.exists() and (final_model / "adapter_config.json").exists():
        return str(final_model)
    
    checkpoints = [d for d in latest_run.iterdir() 
                   if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    return str(checkpoints[-1])


def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  Hyperswitch CPT - Model Comparison Tool                   ║
║                  Compare Base vs Fine-tuned Model Outputs                  ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    base_model = BASE_MODEL_NAME
    default_checkpoint = find_latest_checkpoint()
    
    print("\n📁 Checkpoint Selection")
    print("-" * 80)
    if default_checkpoint:
        print(f"Latest checkpoint found: {default_checkpoint}")
        use_default = input("\nUse this checkpoint? [Y/n]: ").strip().lower()
        if use_default in ['', 'y', 'yes']:
            checkpoint_path = default_checkpoint
        else:
            checkpoint_path = input("Enter checkpoint path (or 'none' to skip): ").strip()
            if checkpoint_path.lower() == 'none':
                checkpoint_path = None
    else:
        print("No checkpoints found in default location.")
        checkpoint_path = input("Enter checkpoint path (or 'none' for base model only): ").strip()
        if checkpoint_path.lower() == 'none':
            checkpoint_path = None
    
    print("\n🔧 Loading models...")
    comparator = ModelComparator(base_model, checkpoint_path)
    
    print("\n⚙️  Generation Parameters (press Enter to use defaults from config)")
    print("-" * 80)
    print(f"Current defaults:")
    print(f"  Max tokens: {DEFAULT_MAX_NEW_TOKENS}")
    print(f"  Temperature: {DEFAULT_TEMPERATURE}")
    print(f"  Top-p: {DEFAULT_TOP_P}")
    print(f"  Top-k: {DEFAULT_TOP_K}")
    print(f"  Repetition penalty: {DEFAULT_REPETITION_PENALTY}")
    print()
    
    try:
        max_tokens_input = input(f"Max new tokens [{DEFAULT_MAX_NEW_TOKENS}]: ").strip()
        max_tokens = int(max_tokens_input) if max_tokens_input else DEFAULT_MAX_NEW_TOKENS
        
        temp_input = input(f"Temperature (0.1-2.0) [{DEFAULT_TEMPERATURE}]: ").strip()
        temperature = float(temp_input) if temp_input else DEFAULT_TEMPERATURE
        
        top_p_input = input(f"Top-p (0.1-1.0) [{DEFAULT_TOP_P}]: ").strip()
        top_p = float(top_p_input) if top_p_input else DEFAULT_TOP_P
        
        top_k_input = input(f"Top-k (0=off) [{DEFAULT_TOP_K}]: ").strip()
        top_k = int(top_k_input) if top_k_input else DEFAULT_TOP_K
        
        rep_pen_input = input(f"Repetition penalty (1.0-2.0) [{DEFAULT_REPETITION_PENALTY}]: ").strip()
        repetition_penalty = float(rep_pen_input) if rep_pen_input else DEFAULT_REPETITION_PENALTY
    except ValueError:
        print("Invalid input, using defaults from config")
        max_tokens = DEFAULT_MAX_NEW_TOKENS
        temperature = DEFAULT_TEMPERATURE
        top_p = DEFAULT_TOP_P
        top_k = DEFAULT_TOP_K
        repetition_penalty = DEFAULT_REPETITION_PENALTY
    
    print("\n✨ Ready! Enter your prompts (type 'quit' or 'exit' to quit)")
    print("💡 Tip: Try asking about Hyperswitch Rust code, payment processing, etc.")
    print("💡 For code completion, end your prompt with a clear context")
    print("💡 Commands: /params (show), /set <param> <value> (change), /reset (reset to config)")
    print("-" * 80)
    
    while True:
        print("\n" + "=" * 80)
        prompt = input("\n🔤 Your prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not prompt:
            print("⚠️  Empty prompt, please try again")
            continue
        
        if prompt.startswith('/'):
            if prompt == '/params':
                print(f"\nCurrent parameters:")
                print(f"  Max tokens: {max_tokens}")
                print(f"  Temperature: {temperature}")
                print(f"  Top-p: {top_p}")
                print(f"  Top-k: {top_k}")
                print(f"  Repetition penalty: {repetition_penalty}")
                continue
            elif prompt == '/reset':
                max_tokens = DEFAULT_MAX_NEW_TOKENS
                temperature = DEFAULT_TEMPERATURE
                top_p = DEFAULT_TOP_P
                top_k = DEFAULT_TOP_K
                repetition_penalty = DEFAULT_REPETITION_PENALTY
                print("✓ Reset to defaults from config")
                continue
            elif prompt.startswith('/set'):
                try:
                    _, param, value = prompt.split()
                    if param == 'max_tokens':
                        max_tokens = int(value)
                    elif param == 'temperature':
                        temperature = float(value)
                    elif param == 'top_p':
                        top_p = float(value)
                    elif param == 'top_k':
                        top_k = int(value)
                    elif param == 'repetition_penalty':
                        repetition_penalty = float(value)
                    else:
                        print(f"Unknown parameter: {param}")
                        print("Available: max_tokens, temperature, top_p, top_k, repetition_penalty")
                        continue
                    print(f"✓ Updated {param} = {value}")
                except:
                    print("Usage: /set <param> <value>")
                    print("Example: /set temperature 0.5")
                continue
        
        comparator.compare(prompt, max_tokens, temperature, top_p, top_k, repetition_penalty)
    
    print("\nCleaning up...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
