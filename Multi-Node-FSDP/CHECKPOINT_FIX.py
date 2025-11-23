# ULTRA-LITE CHECKPOINT FIX
# paste this into train_fsdp.py to replace the lite checkpoint section (lines 408-445)

if checkpoint_mode == 'lite':
    if rank == 0:
        print(f"Saving LoRA adapters only (bypassing FSDP gather)...")
    
    # CRITICAL FIX: Use PEFT's get_peft_model_state_dict()
    # This extracts ONLY LoRA params WITHOUT triggering FSDP's all-gather!
    from peft import get_peft_model_state_dict
    
    # Extract LoRA state dict (this is LOCAL to each rank, no gathering!)
    peft_state_dict = get_peft_model_state_dict(model)
    
    # Only rank 0 saves to disk
    if rank == 0:
        # Get PEFT model for config
        if hasattr(model, 'module'):
            peft_model = model.module
        else:
            peft_model = model
        
        # Save LoRA adapters using the extracted state dict
        # This completely bypasses FSDP state_dict() calls!
        peft_model.save_pretrained(
            save_dir,
            state_dict=peft_state_dict,
            safe_serialization=True
        )
        tokenizer.save_pretrained(save_dir)
        
        # Save metadata
        metadata = {
            "step": step,
            "eval_loss": eval_loss,
            "checkpoint_mode": "lite",
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved LoRA adapters to {save_dir}")
        print(f"{'='*60}")
        print(f"✅ LITE CHECKPOINT COMPLETE (~5s)")
        print(f"{'='*60}\n")
