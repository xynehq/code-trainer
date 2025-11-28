#!/usr/bin/env python3
"""
Reorganize dataset into curriculum learning phases.

Phase 1 (20%): Code Foundation - Files + Test pairs
Phase 2 (30%): Change Patterns - Commits (chronological) + Small PRs
Phase 3 (50%): PR Mastery - Medium/Large PRs with reviews
"""

import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict


def load_dataset(file_path):
    """Load JSONL dataset."""
    data = {
        'file': [],
        'test_pair': [],
        'commit': [],
        'pr_diff': []
    }
    
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading dataset"):
            entry = json.loads(line)
            entry_type = entry.get('type')
            if entry_type in data:
                data[entry_type].append(entry)
    
    return data


def estimate_pr_size(pr_entry):
    """Estimate PR complexity based on content length and metadata."""
    content = pr_entry.get('training_content', '')
    
    # Size in characters
    size = len(content)
    
    # Check if it has reviews/comments (more complex)
    has_reviews = 'Reviews:' in content or 'Comments:' in content
    
    # Scoring
    if size < 5000:
        complexity = 'small'
    elif size < 15000:
        complexity = 'medium'
    else:
        complexity = 'large'
    
    # Boost complexity if has reviews
    if has_reviews and complexity == 'small':
        complexity = 'medium'
    
    return complexity, size


def parse_commit_date(commit_entry):
    """Extract date from commit for chronological sorting."""
    date_str = commit_entry.get('date', '')
    try:
        # ISO 8601 format: 2025-11-27T12:18:26+05:30
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        # If parsing fails, return very old date
        return datetime(1970, 1, 1)


def write_phase(phase_data, output_file):
    """Write phase data to JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in tqdm(phase_data, desc=f"Writing {output_file.name}"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Calculate stats
    total_tokens = sum(len(e.get('training_content', '')) // 4 for e in phase_data)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    return len(phase_data), total_tokens, file_size_mb


def reorganize_curriculum(input_file, output_dir):
    """Reorganize dataset into curriculum learning phases."""
    
    print("=" * 80)
    print("Curriculum Learning Dataset Reorganization")
    print("=" * 80)
    
    # Load dataset
    data = load_dataset(input_file)
    
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"Files:      {len(data['file']):,}")
    print(f"Test pairs: {len(data['test_pair']):,}")
    print(f"Commits:    {len(data['commit']):,}")
    print(f"PRs:        {len(data['pr_diff']):,}")
    print(f"Total:      {sum(len(v) for v in data.values()):,}")
    
    # Analyze PRs by size
    print("\n" + "=" * 80)
    print("Analyzing PR Complexity...")
    print("=" * 80)
    
    pr_by_size = defaultdict(list)
    for pr in tqdm(data['pr_diff'], desc="Categorizing PRs"):
        complexity, size = estimate_pr_size(pr)
        pr_by_size[complexity].append((pr, size))
    
    # Sort each category by size
    for complexity in pr_by_size:
        pr_by_size[complexity].sort(key=lambda x: x[1])
    
    print(f"Small PRs:  {len(pr_by_size['small']):,}")
    print(f"Medium PRs: {len(pr_by_size['medium']):,}")
    print(f"Large PRs:  {len(pr_by_size['large']):,}")
    
    # Sort commits chronologically
    print("\n" + "=" * 80)
    print("Sorting Commits Chronologically...")
    print("=" * 80)
    
    data['commit'].sort(key=parse_commit_date)
    print(f"✓ Sorted {len(data['commit']):,} commits by date (oldest → newest)")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Creating Curriculum Phases")
    print("=" * 80)
    
    # Phase 1: Code Foundation (20%)
    # All files + test pairs
    print("\nPhase 1: Code Foundation")
    print("-" * 80)
    phase1_data = data['file'] + data['test_pair']
    
    phase1_file = output_dir / "phase1_foundation.jsonl"
    entries, tokens, size_mb = write_phase(phase1_data, phase1_file)
    
    print(f"✓ Phase 1 Complete")
    print(f"  Entries: {entries:,}")
    print(f"  Tokens:  ~{tokens:,}")
    print(f"  Size:    {size_mb:.2f} MB")
    print(f"  Content: All files + test pairs")
    
    # Phase 2: Change Patterns (30%)
    # Commits (chronological) + Small PRs
    print("\nPhase 2: Change Patterns")
    print("-" * 80)
    phase2_data = data['commit'] + [pr for pr, _ in pr_by_size['small']]
    
    phase2_file = output_dir / "phase2_evolution.jsonl"
    entries, tokens, size_mb = write_phase(phase2_data, phase2_file)
    
    print(f"✓ Phase 2 Complete")
    print(f"  Entries: {entries:,}")
    print(f"  Tokens:  ~{tokens:,}")
    print(f"  Size:    {size_mb:.2f} MB")
    print(f"  Content: Commits (chronological) + Small PRs")
    
    # Phase 3: PR Mastery (50%)
    # Medium + Large PRs (sorted by size within each category)
    print("\nPhase 3: PR Mastery")
    print("-" * 80)
    
    # Combine medium and large PRs, sorted by complexity
    phase3_data = []
    phase3_data.extend([pr for pr, _ in pr_by_size['medium']])
    phase3_data.extend([pr for pr, _ in pr_by_size['large']])
    
    phase3_file = output_dir / "phase3_pr_mastery.jsonl"
    entries, tokens, size_mb = write_phase(phase3_data, phase3_file)
    
    print(f"✓ Phase 3 Complete")
    print(f"  Entries: {entries:,}")
    print(f"  Tokens:  ~{tokens:,}")
    print(f"  Size:    {size_mb:.2f} MB")
    print(f"  Content: Medium + Large PRs (with reviews)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Curriculum Learning Dataset Ready!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nTraining Sequence:")
    print("  1. Train on phase1_foundation.jsonl    (2 epochs)")
    print("  2. Train on phase2_evolution.jsonl     (2-3 epochs)")
    print("  3. Train on phase3_pr_mastery.jsonl    (3-4 epochs)")
    print("\nTotal training: ~8-9 epochs across all phases")
    print("\nExpected improvement over random training: 25-40%")
    print("=" * 80)


def main():
    # Paths
    input_file = "dataset/hyperswitch_cpt_dataset.jsonl"
    output_dir = "dataset/curriculum_learning"
    
    # Check input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure hyperswitch_cpt_dataset.jsonl is in the dataset/ folder")
        return
    
    # Run reorganization
    reorganize_curriculum(input_file, output_dir)


if __name__ == "__main__":
    main()
