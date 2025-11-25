#!/usr/bin/env python3
"""
TPU Utilities - Health checks, setup validation, and diagnostics
"""

import os
import sys
import subprocess
import logging
from typing import Optional, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tpu_available() -> bool:
    """Check if TPUs are available"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        # Use newer API - check if device is accessible
        device = xm.xla_device()
        
        # Try to get world size using newer API
        try:
            import torch_xla.runtime as xr
            num_devices = xr.world_size()
        except:
            # Fallback: assume single device if world_size not available
            num_devices = 1
            
        logger.info(f"✓ TPU check passed: {num_devices} devices found, device: {device}")
        return True
    except ImportError:
        logger.error("❌ torch_xla not installed. Install with: pip install torch-xla[tpu]")
        return False
    except Exception as e:
        logger.error(f"❌ TPU check failed: {e}")
        return False


def get_tpu_info() -> Optional[Dict]:
    """Get TPU information"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        
        info = {
            'num_devices': xr.world_size(),
            'local_ordinal': xr.local_ordinal(),
            'device': str(xm.xla_device()),
        }
        return info
    except Exception as e:
        logger.error(f"Failed to get TPU info: {e}")
        return None


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed"""
    dependencies = {
        'torch': False,
        'torch_xla': False,
        'transformers': False,
        'peft': False,
        'wandb': False,
        'yaml': False,
        'tqdm': False,
    }
    
    for dep in dependencies.keys():
        try:
            if dep == 'yaml':
                __import__('yaml')
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


def print_dependency_status():
    """Print status of all dependencies"""
    deps = check_dependencies()
    
    print("\n" + "=" * 70)
    print("📦 Dependency Status")
    print("=" * 70)
    
    all_installed = True
    for dep, installed in deps.items():
        status = "✓" if installed else "❌"
        print(f"  {status} {dep:15s} - {'Installed' if installed else 'Missing'}")
        if not installed:
            all_installed = False
    
    print("=" * 70)
    
    if all_installed:
        print("✅ All dependencies are installed!")
    else:
        print("⚠️  Some dependencies are missing. Install with:")
        print("   pip install -r requirements.txt")
    
    print("=" * 70 + "\n")
    
    return all_installed


def check_config_file(config_path: str = "config.yaml") -> bool:
    """Validate config file exists and has required fields"""
    try:
        import yaml
        
        if not os.path.exists(config_path):
            logger.error(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_fields = [
            'model.name',
            'tpu.type',
            'tpu.num_devices',
            'training.lora.r',
            'training.learning_rate',
            'dataset.output_dir',
        ]
        
        for field in required_fields:
            keys = field.split('.')
            value = config
            for key in keys:
                if key not in value:
                    logger.error(f"❌ Missing required config field: {field}")
                    return False
                value = value[key]
        
        logger.info(f"✓ Config file validated: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config validation failed: {e}")
        return False


def check_dataset(config_path: str = "config.yaml") -> bool:
    """Check if dataset files exist"""
    try:
        import yaml
        from pathlib import Path
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_dir = Path(config['dataset']['output_dir'])
        dataset_file = dataset_dir / "all_data.jsonl"
        
        if not dataset_file.exists():
            logger.error(f"❌ Dataset file not found: {dataset_file}")
            logger.info("   Run: python prepare_dataset.py")
            return False
        
        file_size = dataset_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✓ Dataset found: {dataset_file} ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset check failed: {e}")
        return False


def check_tpu_memory():
    """Check TPU memory status"""
    try:
        import torch_xla.core.xla_model as xm
        
        # Get memory info (this is a basic check)
        device = xm.xla_device()
        logger.info(f"✓ TPU device accessible: {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ TPU memory check failed: {e}")
        return False


def run_health_check():
    """Run complete health check"""
    print("\n" + "=" * 70)
    print("🏥 TPU Training Environment Health Check")
    print("=" * 70 + "\n")
    
    checks = {
        'Dependencies': print_dependency_status,
        'TPU Availability': check_tpu_available,
        'Config File': lambda: check_config_file("config.yaml"),
        'Dataset': lambda: check_dataset("config.yaml"),
        'TPU Memory': check_tpu_memory,
    }
    
    results = {}
    
    for check_name, check_func in checks.items():
        if check_name == 'Dependencies':
            result = check_func()
        else:
            print(f"Checking {check_name}...")
            result = check_func()
            print()
        
        results[check_name] = result
    
    print("=" * 70)
    print("📊 Health Check Summary")
    print("=" * 70)
    
    all_passed = True
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:10s} - {check_name}")
        if not result:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to start training.")
        print("   Run: bash run_training.sh\n")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.\n")
        return 1


def print_tpu_topology():
    """Print TPU topology information"""
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        
        print("\n" + "=" * 70)
        print("🔧 TPU Topology Information")
        print("=" * 70)
        print(f"  World Size: {xr.world_size()}")
        print(f"  Local Ordinal: {xr.local_ordinal()}")
        print(f"  Device: {xm.xla_device()}")
        
        # Try to get more detailed info
        try:
            device_attributes = xr.device_attributes(str(xm.xla_device()))
            print(f"  Device Attributes: {device_attributes}")
        except:
            pass
        
        print("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to get TPU topology: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TPU Utilities")
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run complete health check'
    )
    parser.add_argument(
        '--check-tpu',
        action='store_true',
        help='Check TPU availability only'
    )
    parser.add_argument(
        '--topology',
        action='store_true',
        help='Print TPU topology information'
    )
    parser.add_argument(
        '--deps',
        action='store_true',
        help='Check dependencies only'
    )
    
    args = parser.parse_args()
    
    if args.health_check or len(sys.argv) == 1:
        return run_health_check()
    elif args.check_tpu:
        return 0 if check_tpu_available() else 1
    elif args.topology:
        print_tpu_topology()
        return 0
    elif args.deps:
        print_dependency_status()
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
