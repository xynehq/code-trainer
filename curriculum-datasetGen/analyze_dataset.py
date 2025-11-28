"""
Example script showing how to use the generated dataset for training.
"""

import json
from pathlib import Path


def load_dataset(file_path: str):
    """Load JSONL dataset and yield entries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def analyze_dataset(file_path: str):
    """Analyze dataset composition."""
    print("Dataset Analysis")
    print("=" * 80)
    
    type_counts = {}
    total_entries = 0
    total_training_tokens = 0
    
    for entry in load_dataset(file_path):
        total_entries += 1
        entry_type = entry.get('type', 'unknown')
        type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        
        # Estimate tokens (rough estimate: 1 token ≈ 4 characters)
        training_content = entry.get('training_content', '')
        total_training_tokens += len(training_content) // 4
    
    print(f"Total entries: {total_entries:,}")
    print(f"Estimated training tokens: {total_training_tokens:,}")
    print(f"\nBreakdown by type:")
    for entry_type, count in sorted(type_counts.items()):
        percentage = (count / total_entries) * 100
        print(f"  {entry_type:15s}: {count:6,} ({percentage:5.1f}%)")
    print("=" * 80)


def sample_entries(file_path: str, n: int = 3):
    """Show sample entries from dataset."""
    print("\nSample Entries")
    print("=" * 80)
    
    samples_by_type = {}
    
    for entry in load_dataset(file_path):
        entry_type = entry.get('type', 'unknown')
        if entry_type not in samples_by_type:
            samples_by_type[entry_type] = entry
        
        if len(samples_by_type) >= 4:  # One of each type
            break
    
    for entry_type, entry in samples_by_type.items():
        print(f"\nType: {entry_type}")
        print("-" * 80)
        
        # Show metadata
        for key, value in entry.items():
            if key != 'training_content':
                print(f"{key}: {value}")
        
        # Show truncated training content
        training_content = entry.get('training_content', '')
        preview_length = 500
        if len(training_content) > preview_length:
            print(f"\ntraining_content (first {preview_length} chars):")
            print(training_content[:preview_length] + "...")
        else:
            print(f"\ntraining_content:")
            print(training_content)
        print("=" * 80)


def filter_by_type(file_path: str, output_path: str, entry_type: str):
    """Filter dataset by type and save to new file."""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry in load_dataset(file_path):
            if entry.get('type') == entry_type:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1
    
    print(f"Filtered {count} entries of type '{entry_type}' to {output_path}")


if __name__ == "__main__":
    dataset_file = "hyperswitch_cpt_dataset.jsonl"
    
    if not Path(dataset_file).exists():
        print(f"Dataset file not found: {dataset_file}")
        print("Please run generate_dataset.py first")
    else:
        # Analyze dataset
        analyze_dataset(dataset_file)
        
        # Show samples
        sample_entries(dataset_file)
        
        # Example: Filter only PR diffs
        # filter_by_type(dataset_file, "pr_diffs_only.jsonl", "pr_diff")
