# Enhanced Python Dataset Preparator

A comprehensive dataset preparation tool for CPT (Continual Pre-Training) that extracts both code-level details, documentation, and Git commit history from Python repositories to create rich training datasets.

## 🚀 Features

### Code Analysis
- **File-level Processing**: Complete Python files with headers and metadata
- **Granular Extraction**: Individual functions, classes, and methods with detailed metadata
- **AST Analysis**: Deep Abstract Syntax Tree inspection with comprehensive metrics
- **Module Structure**: Imports, constants, variables, type aliases, and module docstrings
- **Smart Chunking**: Intelligent content splitting with configurable overlap
- **Method Analysis**: Detailed extraction of methods, properties, static/class methods with decorators

### Git Commit Analysis 🆕
- **Commit History**: Extracts commit messages, file changes, and development patterns
- **File Change Tracking**: Detailed analysis of added, modified, deleted, and renamed files
- **AST-level Changes**: Tracks function/class additions, removals, and modifications
- **Configurable Time Range**: Analyze commits from specified time periods

### Documentation Processing
- **Multi-format Support**: Processes Markdown, reStructuredText, AsciiDoc, and text files
- **Image Filtering**: Removes images, badges, and visual elements for clean text
- **Documentation Types**: README, Contributing, API docs, and project documentation

### Advanced Features
- **Generated File Filtering**: Skips auto-generated code and build artifacts
- **Token Counting**: Accurate token estimation using transformers or character-based fallback

## 📋 Requirements

```bash
pip install PyYAML markdown beautifulsoup4 tqdm
```

```bash
pip install transformers
```

## 🔧 Configuration

Add `config.yaml`:

```yaml
repository:
  name: "your-repo-name"
  path: "path/to/your/python/repo"
  language: "python"

model:
  name: "your-model-name"  # For tokenizer (optional)

dataset:
  output_dir: "output_directory"
  max_tokens: 8192
  overlap_tokens: 512
  include_tests: true
  include_commits: false    # Enable Git commit analysis
  commit_days: 300        # Number of days of commit history to analyze
```

### Git Commit Analysis Configuration

To enable Git commit analysis, set `include_commits: true` in your config. The analyzer will:
- Extract commits from the last `commit_days` days (default: 300)
- Analyze commit messages and file changes
- Track AST-level modifications for Python files
- Generate structured samples for model training

**Requirements for Git Analysis:**
- Repository must be a Git repository with commit history
- Git command-line tool must be available
- Sufficient permissions to read Git history

## 🏃 Usage

### Usage
```bash
python prepare_py_dataset.py
```



## 📊 Output Structure

The script generates:

### `all_data.jsonl`
Each line contains a JSON object with:
```json
{
  "text": "# Function: example_func\n# Function definition : def example_func(...",
  "file_path": "src/module.py",
  "module": "mypackage",
  "type": "function_definition",
  "tokens": 150,
  "language": "python",
  "function_name": "example_func",
  "function_params": "param1: str, param2: int"
}
```

### Sample Types

#### Python Code Samples
- `full_file`: Complete Python files with metadata headers
- `chunk`: Large file chunks with intelligent splitting and overlap
- `function_definition`: Individual functions with parameters, docstrings, and type hints
- `class_definition`: Class definitions with inheritance, methods count, and decorators
- `method_definition`: Class methods with detailed metadata (static, class, property, async)
- `module_structure`: Imports, constants, variables, and module-level constructs
- `ast_analysis`: Comprehensive AST metrics and analysis
- `ast_analysis_chunk`: Chunked AST analysis for large files

#### Documentation Samples
- `doc_file`: Complete documentation files (Markdown, reStructuredText, AsciiDoc, etc.)
- `doc_chunk`: Chunked documentation for large files

#### Git Commit Samples 🆕
- `commit_summary`: High-level commit information with file change statistics
- `file_change_analysis`: Detailed file-level changes with AST modifications

### `dataset_stats.json`
Comprehensive statistics including:
- Sample counts by type
- Token distribution
- Module coverage
- Documentation breakdown

## 🎯 Example Output

### Function Sample
```
# Function: download_video
# Function Definition:
def download_video(url: str, output_path: str) -> bool:

# Documentation: Downloads a video from YouTube URL

# Parameters:
#   - url: str
#   - output_path: str

# Returns:  -> bool
```

### Documentation Sample
```
# Documentation: README.md
# Type: Doc File
# Lines: 45
# Words: 312

# PyTube - YouTube Video Downloader

A lightweight, dependency-free Python library for downloading YouTube videos.

## Features
- Download videos in various formats
- Extract audio streams
- Playlist support
```

### Git Commit Sample 🆕
```
# Commit Summary
# Date: 2024-01-15 10:30:45 +0000
# Hash: a1b2c3d

# Commit Message:
Add async download support for better performance

Implemented async/await patterns for concurrent downloads.
Added progress tracking and cancellation support.

# Change Summary:
# Files: 5 modified
# Lines: +127 -23
# Net Change: 104 lines

# Files Modified (5):
~ pytube/main.py (+45 -8)
~ pytube/streams.py (+32 -12)
~ pytube/helpers.py (+28 -3)
~ tests/test_async.py (+22 -0)
```

### File Change Analysis Sample 🆕
```
# File Change Analysis
# Commit: a1b2c3d
# Date: 2024-01-15 10:30:45 +0000
# File: pytube/main.py

# Commit Message:
Add async download support for better performance

# Change Statistics:
# Lines Added: 45
# Lines Removed: 8
# Net Change: 37

# Code Structure Changes:
# Functions Added: download_async, track_progress
# Functions Modified: download_video, extract_info
# New Imports:
+ import asyncio
+ from typing import AsyncGenerator
```

  Total samples: 377
    └─ File-level: 43
    └─ Granular: 334
  Total tokens: 115,325

  📂 File-level samples:
    └─ full_file: 40

  🐍 Python granular samples:
    └─ functions: 83
    └─ classes: 25
    └─ methods: 185
    └─ module_structure: 18
    └─ ast_analysis: 23

  🌳 AST Analysis samples: 23

  📚 Documentation samples: 3
    └─ Markdown Files: 3

  📦 Python modules covered: 6
```
## 📈 Dataset Statistics

Example output:
```
📊 Python Dataset Summary:
  Total samples: 542
    └─ File-level: 43
    └─ Granular: 334
    └─ Commit: 165
  Total tokens: 187,456

  📂 File-level samples:
    └─ full_file: 40
    └─ chunk: 3

  🐍 Python granular samples:
    └─ functions: 83
    └─ classes: 25
    └─ methods: 185
    └─ module_structure: 18
    └─ ast_analysis: 23

  🌳 AST Analysis samples: 23
    └─ Chunked AST samples: 5

  📚 Documentation samples: 8
    └─ Doc Files: 6
    └─ Doc Chunks: 2

  🔄 Git commit samples: 165
    └─ Commit Summaries: 45
    └─ File Change Analysis: 120

  📦 Python modules covered: 6
```
======================================================================
  Total samples: 377
    └─ File-level: 43
    └─ Granular: 334
  Total tokens: 115,325

  📂 File-level samples:
    └─ full_file: 40

  🐍 Python granular samples:
    └─ functions: 83
    └─ classes: 25
    └─ methods: 185
    └─ module_structure: 18
    └─ ast_analysis: 23

  🌳 AST Analysis samples: 23

  📚 Documentation samples: 3
    └─ Markdown Files: 3

  📦 Python modules covered: 6
```

## ⚙️ Advanced Configuration

### Exclusion Patterns
The script automatically excludes:
- `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`
- `dist/`, `build/`, `*.egg-info/`
- `.pytest_cache/`, `.tox/`, `venv/`, `.venv/`
- `.mypy_cache/`, `.coverage`, `htmlcov/`

### Test File Detection
Identifies test files by:
- Path patterns: `/tests/`, `/test/`
- Naming: `test_*.py`, `*_test.py`
- Content: `import unittest`, `import pytest`

## � Code Quality Features

### AST Analysis Includes
- Node type distribution
- Control flow constructs
- Function/class definitions
- Expression complexity
- Code quality metrics

### Module Structure Extraction
- Standard library imports
- Third-party imports
- Relative imports
- Constants and variables
- Type aliases


## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError**: Install required dependencies
```bash
pip install PyYAML markdown beautifulsoup4 tqdm
```

**Empty output**: Check repository path in config.yaml

**Memory issues**: Reduce `max_tokens` in configuration

**Encoding errors**: Files are read with UTF-8 and error ignore mode



Perfect for CPT training on Python repositories with rich code-documentation relationships!
