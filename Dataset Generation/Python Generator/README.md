# Enhanced Python Dataset Preparator

A comprehensive dataset preparation tool for CPT (Continual Pre-Training) that extracts both code-level details and documentation from Python repositories to create rich training datasets.

## 🚀 Features

### Code Analysis
- **File-level Processing**: Complete Python files with headers and metadata
- **Granular Extraction**: Individual functions, classes, and methods
- **AST Analysis**: Deep Abstract Syntax Tree inspection and metrics
- **Module Structure**: Imports, constants, variables, and type aliases
- **Smart Chunking**: Intelligent content splitting with configurable overlap

### Documentation Processing
- **Markdown Integration**: Processes README, Contributing, and other docs
- **Image Filtering**: Removes images, badges, and visual elements

### Advanced Features
- **Test File Detection**: Automatically excludes or includes test files
- **Generated File Filtering**: Skips auto-generated code
- **Token Counting**: Accurate token estimation for training

## 📋 Requirements

```bash
pip install PyYAML markdown beautifulsoup4 tqdm
```

Optional (for accurate token counting):
```bash
pip install transformers
```

## 🔧 Configuration

Update `config.yaml`:

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
```

## 🏃 Usage

### Basic Usage
```bash
python prepare_py_dataset.py
```

### Custom Configuration
```bash
python prepare_py_dataset.py --config custom_config.yaml
```

## 📊 Output Structure

The script generates:

### `all_data.jsonl`
Each line contains a JSON object with:
```json
{
  "text": "# Function: example_func\n# File: module.py...",
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
- `full_file`: Complete Python files
- `chunk`: Large file chunks
- `function_definition`: Individual functions
- `class_definition`: Class definitions
- `method_definition`: Class methods
- `module_structure`: Imports and module-level code
- `ast_analysis`: AST metrics and analysis
- `markdown_file`: Documentation files

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
# File: pytube/main.py
# Module: pytube
# Documentation: Downloads a video from YouTube URL

# Function Definition:
def download_video(url: str, output_path: str) -> bool:

# Parameters:
#   - url: str
#   - output_path: str

# Returns:  -> bool
```

### Documentation Sample
```
# Documentation: README.md
# Type: Markdown Documentation
# Lines: 45
# Words: 312

# PyTube - YouTube Video Downloader

A lightweight, dependency-free Python library for downloading YouTube videos.

## Features
- Download videos in various formats
- Extract audio streams
- Playlist support
```

## 📈 Dataset Statistics

Example output:
```
📊 Python Dataset Summary:
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

### Markdown Filtering
Automatically removes:
- Image references: `![alt](url)`
- HTML image tags: `<img>`
- Badge URLs and shields
- Standalone image links

### Test File Detection
Identifies test files by:
- Path patterns: `/tests/`, `/test/`
- Naming: `test_*.py`, `*_test.py`
- Content: `import unittest`, `import pytest`

## 🔍 Code Quality Features

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

## 🚀 Performance

- **Fast Processing**: Optimized for large repositories
- **Memory Efficient**: Streaming file processing
- **Progress Tracking**: Real-time progress bars
- **Error Handling**: Graceful handling of syntax errors

## 📝 Notes

- Uses character-based token estimation (1 token ≈ 4 chars) when transformers unavailable
- Preserves code structure and comments
- Maintains file relationships and module context
- Suitable for both small projects and large codebases

## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError**: Install required dependencies
```bash
pip install PyYAML markdown beautifulsoup4 tqdm
```

**Empty output**: Check repository path in config.yaml

**Memory issues**: Reduce `max_tokens` in configuration

**Encoding errors**: Files are read with UTF-8 and error ignore mode

## 🎯 Use Cases

- **Model Training**: Prepare datasets for code-focused language models
- **Code Analysis**: Extract structured information from codebases
- **Documentation**: Combine code and docs for comprehensive understanding
- **Research**: Study code patterns and repository structures

Perfect for CPT training on Python repositories with rich code-documentation relationships!
