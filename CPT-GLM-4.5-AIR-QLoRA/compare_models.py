#!/usr/bin/env python3
"""
Model Comparison Script
Compares base GLM-4.5-Air model with fine-tuned checkpoint to detect overfitting
and measure response deviation.
"""

import torch
import argparse
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
from datetime import datetime

class ModelComparator:
    def __init__(self, base_model_path, checkpoint_path=None):
        self.base_model_path = base_model_path
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("="*70)
        print("GLM-4.5-Air Model Comparison Tool")
        print("="*70)
        print(f"\n📁 Base Model: {base_model_path}")
        if checkpoint_path:
            print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"🖥️  Device: {self.device}")
        print()
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load base model and checkpoint with 4-bit quantization"""
        print("Loading models (this may take a few minutes)...\n")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Load tokenizer
        print("1️⃣  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("   ✅ Tokenizer loaded")
        
        # Load base model
        print("\n2️⃣  Loading base model (4-bit quantized)...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.base_model.eval()
        print("   ✅ Base model loaded")
        
        # Load checkpoint if provided
        if self.checkpoint_path:
            print(f"\n3️⃣  Loading checkpoint adapter from {self.checkpoint_path}...")
            # Create a copy of base model for the fine-tuned version
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            # Load LoRA adapter
            self.finetuned_model = PeftModel.from_pretrained(
                self.finetuned_model,
                self.checkpoint_path,
                is_trainable=False
            )
            self.finetuned_model.eval()
            print("   ✅ Fine-tuned model loaded")
        else:
            self.finetuned_model = None
        
        print("\n" + "="*70)
        print("All models loaded successfully!")
        print("="*70 + "\n")
    
    def generate_response(self, model, prompt, max_new_tokens=256, temperature=0.7, do_sample=True):
        """Generate response from a model"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generation_time = time.time() - start_time
        
        # Decode response (remove the input prompt)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response, generation_time
    
    def compare_on_prompt(self, prompt, max_new_tokens=256):
        """Compare base and fine-tuned models on a single prompt"""
        print("\n" + "="*70)
        print(f"💬 Prompt: {prompt}")
        print("="*70)
        
        # Get base model response
        print("\n🔵 BASE MODEL Response:")
        print("-"*70)
        base_response, base_time = self.generate_response(
            self.base_model, prompt, max_new_tokens
        )
        print(base_response)
        print(f"\n⏱️  Generation time: {base_time:.2f}s")
        
        # Get fine-tuned model response if available
        if self.finetuned_model:
            print("\n🟢 FINE-TUNED MODEL Response:")
            print("-"*70)
            ft_response, ft_time = self.generate_response(
                self.finetuned_model, prompt, max_new_tokens
            )
            print(ft_response)
            print(f"\n⏱️  Generation time: {ft_time:.2f}s")
            
            # Calculate simple deviation metrics
            print("\n📊 Comparison Metrics:")
            print("-"*70)
            
            # Length comparison
            base_len = len(base_response)
            ft_len = len(ft_response)
            len_diff = abs(base_len - ft_len)
            len_diff_pct = (len_diff / base_len * 100) if base_len > 0 else 0
            
            print(f"Response Length:")
            print(f"  Base:       {base_len} chars")
            print(f"  Fine-tuned: {ft_len} chars")
            print(f"  Difference: {len_diff} chars ({len_diff_pct:.1f}%)")
            
            # Token-level overlap (simple word-based)
            base_words = set(base_response.lower().split())
            ft_words = set(ft_response.lower().split())
            overlap = len(base_words & ft_words)
            union = len(base_words | ft_words)
            jaccard = (overlap / union * 100) if union > 0 else 0
            
            print(f"\nWord-level Similarity:")
            print(f"  Jaccard Index: {jaccard:.1f}%")
            print(f"  Overlap: {overlap}/{union} unique words")
            
            # Speed comparison
            speedup = ((base_time - ft_time) / base_time * 100) if base_time > 0 else 0
            print(f"\nGeneration Speed:")
            print(f"  Base:       {base_time:.2f}s")
            print(f"  Fine-tuned: {ft_time:.2f}s")
            print(f"  Speedup:    {speedup:+.1f}%")
            
            return {
                "prompt": prompt,
                "base_response": base_response,
                "ft_response": ft_response,
                "base_time": base_time,
                "ft_time": ft_time,
                "base_length": base_len,
                "ft_length": ft_len,
                "length_diff_pct": len_diff_pct,
                "jaccard_similarity": jaccard,
                "speedup_pct": speedup
            }
        else:
            return {
                "prompt": prompt,
                "base_response": base_response,
                "base_time": base_time,
                "base_length": len(base_response)
            }
    
    def batch_compare(self, prompts, max_new_tokens=256, save_results=None):
        """Compare models on multiple prompts and optionally save results"""
        results = []
        
        print("\n" + "="*70)
        print(f"🚀 Running comparison on {len(prompts)} prompts")
        print("="*70)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]", end=" ")
            result = self.compare_on_prompt(prompt, max_new_tokens)
            results.append(result)
        
        # Summary statistics
        if self.finetuned_model and len(results) > 0:
            print("\n" + "="*70)
            print("📈 OVERALL SUMMARY")
            print("="*70)
            
            avg_jaccard = sum(r["jaccard_similarity"] for r in results) / len(results)
            avg_len_diff = sum(r["length_diff_pct"] for r in results) / len(results)
            avg_speedup = sum(r["speedup_pct"] for r in results) / len(results)
            
            print(f"\nAverage Metrics (across {len(results)} prompts):")
            print(f"  Jaccard Similarity: {avg_jaccard:.1f}%")
            print(f"  Length Difference:  {avg_len_diff:.1f}%")
            print(f"  Speed Change:       {avg_speedup:+.1f}%")
            
            # Overfitting assessment
            print(f"\n🔍 Overfitting Assessment:")
            if avg_jaccard > 80:
                print(f"  ⚠️  Very high similarity ({avg_jaccard:.1f}%) - minimal adaptation")
            elif avg_jaccard > 50:
                print(f"  ✅ Moderate similarity ({avg_jaccard:.1f}%) - good adaptation")
            elif avg_jaccard > 20:
                print(f"  ⚠️  Low similarity ({avg_jaccard:.1f}%) - significant deviation")
            else:
                print(f"  ❌ Very low similarity ({avg_jaccard:.1f}%) - possible overfitting")
        
        # Save results if requested
        if save_results:
            output_path = Path(save_results)
            with open(output_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "base_model": self.base_model_path,
                    "checkpoint": self.checkpoint_path,
                    "results": results
                }, f, indent=2)
            print(f"\n💾 Results saved to: {save_results}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare base GLM-4.5-Air model with fine-tuned checkpoint"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/workspace/Avinash/models/GLM-4.5-Air",
        help="Path to base model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="glm45-air-cpt-qlora/checkpoint-1000",
        help="Path to checkpoint directory (LoRA adapter)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Custom prompts to test (space-separated)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="Path to JSON file with prompts list"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        help="Path to save comparison results (JSON)"
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Only test base model (no checkpoint comparison)"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint = None if args.base_only else args.checkpoint
    
    # Load models
    comparator = ModelComparator(args.base_model, checkpoint)
    
    # Determine prompts to use
    if args.prompts:
        prompts = args.prompts
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            data = json.load(f)
            prompts = data if isinstance(data, list) else data.get("prompts", [])
    else:
        # Default test prompts
        prompts = [
            "Explain what a payment gateway is in simple terms.",
            "Write a Python function to calculate the factorial of a number.",
            "What are the key differences between REST and GraphQL APIs?",
            "How does blockchain technology ensure transaction security?",
            "Describe the SOLID principles in software engineering."
        ]
    
    # Run comparison
    results = comparator.batch_compare(
        prompts,
        max_new_tokens=args.max_tokens,
        save_results=args.save_results
    )
    
    print("\n" + "="*70)
    print("✅ Comparison complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
