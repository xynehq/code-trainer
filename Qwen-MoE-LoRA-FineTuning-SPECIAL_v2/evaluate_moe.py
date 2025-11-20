#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned MoE Model
Compares base model vs fine-tuned model performance
"""

import os
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoEEvaluator:
    """Evaluate fine-tuned MoE model"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_model = None
        self.finetuned_model = None
        self.tokenizer = None
        self.eval_dataset = None
    
    def load_tokenizer(self):
        """Load tokenizer"""
        logger.info(f"Loading tokenizer: {self.config['model']['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_base_model(self):
        """Load base model without fine-tuning"""
        logger.info(f"Loading base model: {self.config['model']['name']}")
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config['model']['torch_dtype'], torch.bfloat16)
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto",
        )
        self.base_model.eval()
        logger.info("✓ Base model loaded")
    
    def load_finetuned_model(self, adapter_path: Optional[str] = None):
        """Load fine-tuned model with LoRA adapters"""
        if adapter_path is None:
            adapter_path = self.config['training']['output_dir']
        
        logger.info(f"Loading fine-tuned model from: {adapter_path}")
        
        # Check if merged model exists
        merged_path = self.config['post_training']['merged_output_dir']
        if os.path.exists(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
            logger.info("Loading merged model...")
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.config['model']['torch_dtype'], torch.bfloat16)
            
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(
                merged_path,
                trust_remote_code=True,
                dtype=dtype,
                device_map="auto",
            )
        else:
            # Load base + adapters
            logger.info("Loading base model + LoRA adapters...")
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.config['model']['torch_dtype'], torch.bfloat16)
            
            base = AutoModelForCausalLM.from_pretrained(
                self.config['model']['name'],
                trust_remote_code=True,
                dtype=dtype,
                device_map="auto",
            )
            self.finetuned_model = PeftModel.from_pretrained(base, adapter_path)
        
        self.finetuned_model.eval()
        logger.info("✓ Fine-tuned model loaded")
    
    def load_eval_dataset(self):
        """Load evaluation dataset"""
        logger.info("Loading evaluation dataset...")
        
        data_path = Path(self.config['dataset']['output_dir']) / self.config['dataset']['data_file']
        
        # Load and split
        dataset = load_dataset('json', data_files=str(data_path), split='train')
        split_ratio = self.config['validation']['validation_split']
        dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
        
        self.eval_dataset = dataset['test']
        logger.info(f"✓ Loaded {len(self.eval_dataset)} evaluation samples")
    
    def compute_perplexity(self, model, max_samples: Optional[int] = None):
        """Compute perplexity on evaluation dataset"""
        logger.info("Computing perplexity...")
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        samples = self.eval_dataset if max_samples is None else self.eval_dataset.select(range(min(max_samples, len(self.eval_dataset))))
        
        with torch.no_grad():
            for sample in tqdm(samples, desc="Evaluating"):
                text = sample[self.config['dataset']['text_field']]
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config['model']['max_seq_length'],
                )
                
                # Move to device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Accumulate
                batch_tokens = inputs['input_ids'].ne(self.tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
        }
    
    def generate_sample(self, model, prompt: str, max_length: int = 256):
        """Generate text from prompt"""
        model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def compare_models(self, max_samples: Optional[int] = 100):
        """Compare base and fine-tuned models"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 Model Comparison")
        logger.info("=" * 80 + "\n")
        
        # Evaluate base model
        logger.info("Evaluating base model...")
        base_metrics = self.compute_perplexity(self.base_model, max_samples=max_samples)
        
        logger.info("\n📈 Base Model Metrics:")
        logger.info(f"  Loss: {base_metrics['loss']:.4f}")
        logger.info(f"  Perplexity: {base_metrics['perplexity']:.4f}")
        
        # Evaluate fine-tuned model
        logger.info("\nEvaluating fine-tuned model...")
        ft_metrics = self.compute_perplexity(self.finetuned_model, max_samples=max_samples)
        
        logger.info("\n📈 Fine-tuned Model Metrics:")
        logger.info(f"  Loss: {ft_metrics['loss']:.4f}")
        logger.info(f"  Perplexity: {ft_metrics['perplexity']:.4f}")
        
        # Comparison
        logger.info("\n📊 Improvement:")
        loss_improvement = ((base_metrics['loss'] - ft_metrics['loss']) / base_metrics['loss']) * 100
        ppl_improvement = ((base_metrics['perplexity'] - ft_metrics['perplexity']) / base_metrics['perplexity']) * 100
        
        logger.info(f"  Loss improvement: {loss_improvement:+.2f}%")
        logger.info(f"  Perplexity improvement: {ppl_improvement:+.2f}%")
        
        # Save results
        results = {
            'base_model': base_metrics,
            'finetuned_model': ft_metrics,
            'improvement': {
                'loss_percent': loss_improvement,
                'perplexity_percent': ppl_improvement,
            }
        }
        
        output_path = Path(self.config['training']['output_dir']) / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_path}")
        
        return results
    
    def test_generation(self, prompts: List[str]):
        """Test generation on sample prompts"""
        logger.info("\n" + "=" * 80)
        logger.info("🎨 Generation Comparison")
        logger.info("=" * 80 + "\n")
        
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"\n--- Prompt {i} ---")
            logger.info(f"Prompt: {prompt}\n")
            
            # Base model generation
            logger.info("Base Model:")
            base_gen = self.generate_sample(self.base_model, prompt)
            logger.info(f"{base_gen}\n")
            
            # Fine-tuned model generation
            logger.info("Fine-tuned Model:")
            ft_gen = self.generate_sample(self.finetuned_model, prompt)
            logger.info(f"{ft_gen}\n")
            logger.info("-" * 80)
    
    def run_evaluation(self, max_samples: Optional[int] = 100):
        """Run complete evaluation"""
        logger.info("=" * 80)
        logger.info("🔍 MoE Model Evaluation")
        logger.info("=" * 80 + "\n")
        
        # Load components
        self.load_tokenizer()
        self.load_eval_dataset()
        self.load_base_model()
        self.load_finetuned_model()
        
        # Compare models
        results = self.compare_models(max_samples=max_samples)
        
        # Test generation
        test_prompts = [
            "// Function to process payment with Stripe connector\npub async fn process_stripe_payment",
            "// Struct definition for PaymentRequest\npub struct PaymentRequest",
            "impl PaymentConnector for StripeConnector",
        ]
        
        self.test_generation(test_prompts)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Evaluation Complete!")
        logger.info("=" * 80)
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned MoE model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples for evaluation")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapters")
    
    args = parser.parse_args()
    
    evaluator = MoEEvaluator(config_path=args.config)
    evaluator.run_evaluation(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
