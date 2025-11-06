# EulerDatasetGen

A specialized tool for downloading Haskell packages from Hackage and preparing them into training datasets for Large Language Models (LLMs), particularly designed for Continued Pre-Training (CPT) tasks.

## Overview

EulerDatasetGen automates the process of:
1. **Downloading** Haskell packages from Hackage based on a curated list
2. **Processing** the source code into tokenized training samples  
3. **Preparing** structured datasets suitable for language model training

The tool is optimized for memory efficiency and can handle large-scale package collections through streaming processing.

## Features

- **🚀 Concurrent Downloads**: Multi-threaded package downloading with progress tracking
- **🧠 Smart Processing**: Automatic chunking of large files with configurable token limits
- **📊 Comprehensive Statistics**: Detailed dataset metrics and sample type breakdown
- **⚡ Memory Efficient**: Streaming processing to handle large codebases
- **🔧 Configurable**: Flexible configuration via YAML files
- **🎯 Sample Variety**: Generates multiple sample types (full files, chunks, function signatures)

## Quick Start

### Prerequisites

- Python 3.8+
- Internet connection for downloading packages
- Sufficient disk space for package storage

### Installation

1. Clone the repository and navigate to the EulerDatasetGen directory:
```bash
cd EulerDatasetGen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Download Packages
```bash
python download_packages.py
```

This will:
- Read the package list from `packages.txt`
- Download packages concurrently to `HS_packages/` directory
- Skip GHC internal packages and already downloaded packages
- Generate a summary report with success/failure statistics

#### Step 2: Prepare Dataset
```bash
python prepare_dataset.py
```

This will:
- Process all downloaded Haskell source files
- Generate training samples with token counting
- Create a shuffled JSONL dataset
- Save comprehensive statistics

## Configuration

### `config.yaml`

```yaml
repository:
  paths:
    - "HS_packages"              # Path to downloaded packages

dataset:
  output_dir: "haskell_dataset"  # Output directory for dataset
  data_file: "all_data.jsonl"   # Final dataset filename
  max_tokens: 8192              # Maximum tokens per sample
  overlap_tokens: 200           # Overlap for chunking large files
  random_seed: 42               # Seed for shuffling
  include_tests: false          # Whether to include test files

model:
  name: "zai-org/GLM-4.5-Air"   # Tokenizer model for token counting
```

### `packages.txt`

Contains a list of Haskell packages to download in the format:
```
package-name-version
aeson-2.1.3.0
lens-5.2.3
servant-0.19.1
```

## Output Structure

### Dataset Samples

The generated dataset contains JSON samples with the following structure:

```json
{
  "text": "module Data.Example where\n\nexample :: Int -> String\nexample n = show n",
  "file_path": "Data/Example.hs",
  "module": "Data.Example",
  "tokens": 156,
  "type": "full_file"
}
```

### Sample Types

- **`full_file`**: Complete source files under token limit
- **`chunk`**: Overlapping chunks from large files  
- **`function_sig`**: Individual function type signatures

### Generated Files

- `haskell_dataset/all_data.jsonl` - Main training dataset
- `haskell_dataset/dataset_stats.json` - Comprehensive statistics
- `failed_packages.txt` - List of packages that failed to download (if any)

## Project Structure

```
EulerDatasetGen/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── config.yaml             # Configuration file
├── packages.txt            # List of packages to download
├── download_packages.py    # Package downloader
├── prepare_dataset.py      # Dataset preparation script
├── HS_packages/           # Downloaded packages (created)
└── haskell_dataset/       # Generated dataset (created)
```

## Advanced Usage

### Custom Package Lists

Edit `packages.txt` to customize which packages to download. Package names should be in the format `package-version` as they appear on Hackage.

### Memory Optimization

For very large datasets, the tool automatically:
- Processes files incrementally 
- Uses streaming I/O to minimize memory usage
- Provides progress tracking for long-running operations

### Token Counting

The tool supports multiple tokenizer backends:
- **Transformers**: Uses the specified model tokenizer for accurate counting
- **Fallback**: Character-based estimation (1 token ≈ 4 chars) if tokenizer fails

## Troubleshooting

### Common Issues

1. **Download failures**: Check internet connection and Hackage availability
2. **Memory issues**: Reduce `max_tokens` in config or process smaller batches
3. **Tokenizer errors**: Ensure transformers library is properly installed

### Logs and Debugging

The scripts provide detailed logging output. For more verbose logging, check the console output during execution.

## Statistics Example

After processing, you'll see output like:

```
======================================================================
Dataset Summary:
======================================================================
  Total samples: 45,230
    - File-level: 38,450
    - Granular: 6,780
  Total tokens: 12,345,678
  Modules: 1,234
  Files processed: 8,901
======================================================================
```

## Contributing

When adding new features or modifying the processing pipeline:

1. Update the configuration schema in `config.yaml`
2. Ensure backward compatibility with existing datasets
3. Add appropriate error handling and logging
4. Update this README with new features

## License

This project is part of the larger code-trainer repository. Please refer to the main repository license.