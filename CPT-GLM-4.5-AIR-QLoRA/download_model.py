#!/usr/bin/env python3

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

def download_glm_model():
    """Download GLM-4.5-Air model to local directory"""
    
    # Model configuration
    model_name = "Qwen/Qwen3-8B"
    local_path = "/workspace/Avinash/models/Qwen3-8B"
    
    print(f"=== Downloading GLM-4.5-Air Model ===")
    print(f"Source: {model_name}")
    print(f"Destination: {local_path}")
    print(f"Model size: ~100B+ parameters (MoE architecture)")
    print("")
    
    # Create models directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    try:
        print("Downloading model files...")
        print("This may take a while due to the large model size...")
        print("")
        
        # Download model using snapshot_download for better reliability
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8,
        )
        
        print("✅ Model downloaded successfully!")
        print(f"Model saved to: {local_path}")
        
        # Verify download by checking key files
        required_files = ["config.json", "pytorch_model.bin.index.json"]
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(local_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Warning: Missing files: {missing_files}")
        else:
            print("✅ All required model files present")
        
        # Test loading tokenizer
        print("\nTesting tokenizer loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            print(f"✅ Tokenizer loaded successfully (vocab_size: {tokenizer.vocab_size})")
        except Exception as e:
            print(f"❌ Tokenizer loading failed: {e}")
        
        # Show directory size
        total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                        for dirpath, dirnames, filenames in os.walk(local_path) 
                        for filename in filenames)
        size_gb = total_size / (1024**3)
        print(f"\n📊 Total model size: {size_gb:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check internet connection")
        print("2. Verify Hugging Face token access")
        print("3. Ensure sufficient disk space (>200GB recommended)")
        print("4. Try running with --resume flag")
        return False

if __name__ == "__main__":
    success = download_glm_model()
    if success:
        print("\n🎉 Ready to start training!")
        print("Run: ./launch_4xh200.sh")
    else:
        print("\n❌ Model download failed. Please check the error above.")
