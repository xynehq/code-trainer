#!/usr/bin/env python3
"""
Hyperswitch CPT Dataset Generator

Generates a comprehensive dataset for continued pre-training (CPT) from the Hyperswitch repository.
Includes:
- Full repository snapshot
- Git commit history with diffs
- GitHub PRs with reviews and comments
- Test-implementation pairs
"""

import yaml
import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from extractors.repo_extractor import RepoExtractor
from extractors.git_extractor import GitExtractor
from extractors.github_extractor import GitHubExtractor
from extractors.test_pair_extractor import TestPairExtractor
from utils.chunker import TokenizerChunker


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def write_jsonl(data: Dict, output_file):
    """Write a single JSON line to output file."""
    output_file.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    """Main dataset generation pipeline."""
    print("=" * 80)
    print("Hyperswitch CPT Dataset Generator")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/5] Loading configuration...")
    try:
        config = load_config()
        print(f"✓ Configuration loaded from config.yaml")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        print("Please ensure config.yaml exists and is properly formatted.")
        sys.exit(1)
    
    # Validate repository path
    repo_path = config['repo']['local_path']
    if not Path(repo_path).exists():
        print(f"\n✗ Repository not found at: {repo_path}")
        print("Please clone the hyperswitch repository first:")
        print(f"  git clone https://github.com/{config['github']['repo_owner']}/{config['github']['repo_name']}.git {repo_path}")
        sys.exit(1)
    
    print(f"✓ Repository found at: {repo_path}")
    
    # Initialize chunker
    print("\n[2/5] Initializing tokenizer...")
    try:
        chunker = TokenizerChunker(
            model_name=config['model']['name'],
            max_chunk_size=config['model']['max_chunk_size'],
            chunk_overlap=config['model']['chunk_overlap']
        )
        print(f"✓ Tokenizer loaded: {config['model']['name']}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        print("This may take a moment on first run as the tokenizer is downloaded...")
        sys.exit(1)
    
    # Initialize extractors
    print("\n[3/5] Initializing extractors...")
    repo_extractor = RepoExtractor(
        repo_path=repo_path,
        include_patterns=config['collection']['include_patterns'],
        exclude_patterns=config['collection']['exclude_patterns']
    )
    
    git_extractor = GitExtractor(repo_path=repo_path)
    
    github_extractor = GitHubExtractor(
        token=config['github']['api_token'],
        repo_owner=config['github']['repo_owner'],
        repo_name=config['github']['repo_name']
    )
    
    test_pair_extractor = TestPairExtractor(repo_path=repo_path)
    
    print("✓ All extractors initialized")
    
    # Open output file
    output_path = config['output']['file']
    print(f"\n[4/5] Creating output file: {output_path}")
    
    total_entries = 0
    total_chunks = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        
        # Extract repository files
        print("\n--- Extracting repository files ---")
        for file_data in repo_extractor.extract_files(show_progress=config['performance']['show_progress']):
            # Apply chunking if needed
            chunks = chunker.chunk_text(
                text=file_data['training_content'],
                metadata={k: v for k, v in file_data.items() if k != 'training_content'}
            )
            
            for chunk in chunks:
                write_jsonl(chunk, outfile)
                total_chunks += 1
            
            total_entries += 1
        
        print(f"✓ Extracted {total_entries} files ({total_chunks} chunks)")
        
        # Extract git commits
        print("\n--- Extracting git commits ---")
        commit_count = 0
        for commit_data in git_extractor.extract_commits(
            include_all=config['collection']['commit_settings']['include_all_commits'],
            max_commits=config['collection']['commit_settings']['max_commits'],
            show_progress=config['performance']['show_progress']
        ):
            # Apply chunking if needed
            chunks = chunker.chunk_text(
                text=commit_data['training_content'],
                metadata={k: v for k, v in commit_data.items() if k != 'training_content'}
            )
            
            for chunk in chunks:
                write_jsonl(chunk, outfile)
                total_chunks += 1
            
            commit_count += 1
        
        print(f"✓ Extracted {commit_count} commits")
        total_entries += commit_count
        
        # Extract GitHub PRs
        print("\n--- Extracting GitHub PRs ---")
        pr_count = 0
        try:
            for pr_data in github_extractor.extract_prs(
                include_merged=config['collection']['pr_settings']['include_merged'],
                include_closed=config['collection']['pr_settings']['include_closed'],
                include_reviews=config['collection']['pr_settings']['include_reviews'],
                include_comments=config['collection']['pr_settings']['include_comments'],
                max_prs=config['collection']['pr_settings']['max_prs'],
                show_progress=config['performance']['show_progress']
            ):
                # Apply chunking if needed
                chunks = chunker.chunk_text(
                    text=pr_data['training_content'],
                    metadata={k: v for k, v in pr_data.items() if k != 'training_content'}
                )
                
                for chunk in chunks:
                    write_jsonl(chunk, outfile)
                    total_chunks += 1
                
                pr_count += 1
            
            print(f"✓ Extracted {pr_count} PRs")
            total_entries += pr_count
        except Exception as e:
            print(f"⚠ Warning: Error extracting PRs: {e}")
            print("Continuing with other data sources...")
        
        # Extract test pairs
        print("\n--- Extracting test-implementation pairs ---")
        pair_count = 0
        for pair_data in test_pair_extractor.extract_test_pairs(
            show_progress=config['performance']['show_progress']
        ):
            # Apply chunking if needed
            chunks = chunker.chunk_text(
                text=pair_data['training_content'],
                metadata={k: v for k, v in pair_data.items() if k != 'training_content'}
            )
            
            for chunk in chunks:
                write_jsonl(chunk, outfile)
                total_chunks += 1
            
            pair_count += 1
        
        print(f"✓ Extracted {pair_count} test pairs")
        total_entries += pair_count
    
    # Summary
    print("\n" + "=" * 80)
    print("[5/5] Dataset Generation Complete!")
    print("=" * 80)
    print(f"Total entries: {total_entries}")
    print(f"Total chunks (after tokenization): {total_chunks}")
    print(f"Output file: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / (1024**2):.2f} MB")
    print("\nDataset breakdown:")
    print("  - Repository files")
    print("  - Git commits with diffs")
    print("  - GitHub PRs with reviews/comments")
    print("  - Test-implementation pairs")
    print("\nYou can now use this dataset for continued pre-training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
