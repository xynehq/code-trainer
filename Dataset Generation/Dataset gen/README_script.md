# Multi-Repository Dataset Generator for Code LLM Training

Generate high-quality datasets for Continual Pre-Training (CPT) from multiple GitHub repositories across multiple programming languages.

## Key Features

### Multi-Repository Support
- Accept multiple GitHub repository URLs via CLI or config file
- Automatic cloning of repositories if not present locally
- Process all repositories together in a single dataset

### Multi-Language Support
- **Rust** (`.rs`) - Full support with granular extraction
- **Python** (`.py`) - Functions and classes extraction
- **JavaScript** (`.js`, `.jsx`) - Functions, arrow functions, classes
- **TypeScript** (`.ts`, `.tsx`) - Functions, arrow functions, classes  
- **HTML** (`.html`, `.htm`) - File-level processing
- **CSS** (`.css`, `.scss`, `.sass`) - File-level processing

### Extraction Modes

#### File-Level Extraction
- Complete file content without metadata headers
- Token-based chunking for large files
- Overlapping chunks for context preservation
- Clean, normalized code ready for training

#### Granular Extraction
- **Rust** (default): Functions, structs, traits, implementations, module exports
- **Python** (with `--granular`): Functions and classes with docstrings
- **JavaScript/TypeScript** (with `--granular`): Functions, arrow functions, classes
- Preserves documentation and type information

### Git Commit History
- Extract commit diffs from all repositories
- Filter by configured languages
- Include commit messages and metadata
- Configurable time range (recent N days) or commit count
- Automatic filtering of irrelevant changes

### Smart Filtering
- Excludes test files, generated code, build artifacts
- Language-specific exclusion patterns
- Filters out `node_modules`, `target`, `__pycache__`, etc.

### Output

**Files Generated:**
- `all_data.jsonl` - All training samples (file-level + granular + commits)
- `dataset_stats.json` - Detailed statistics

### Basic Usage
```bash
# Single repository
python3 shubhangi_dataset.py --repos https://github.com/user/repo.git

# Multiple repositories
python3 shubhangi_dataset.py --repos \
  https://github.com/user/repo1.git \
  https://github.com/user/repo2.git \
  https://github.com/user/repo3.git
```

### With Granular Extraction
```bash
# Enable granular extraction for all languages (default: Rust only)
python3 shubhangi_dataset.py --repos <url1> <url2> --granular
```

### Using Config File
```bash
# Use custom config file
python3 shubhangi_dataset.py --config path/to/config.yaml
```

### Combined Options
```bash
python3 shubhangi_dataset.py \
  --repos https://github.com/user/repo.git \
  --granular \
  --config custom_config.yaml
```
