# Enhanced Multi-Repo & Multi-Language Dataset Preparation Script

This script generates datasets for Continual Pre-Training (CPT) of code LLMs, supporting multiple repositories and programming languages. It extracts file-level and granular samples, and can include git commit history for robust model training.

## Key Features Added in This Script

- **Multi-Repository Support**: Accepts multiple GitHub repo URLs via CLI or config, clones them, and processes all together.
- **Multi-Language Support**: Handles Rust, Python, JavaScript, TypeScript, HTML, and CSS files. Language detection is automatic by file extension.
- **Granular Extraction for All Languages**: With `--granular` flag, extracts functions/classes for Python, JS/TS, in addition to Rust (default: granular for Rust only).
- **Flexible Configuration**: Reads repo paths/URLs, output directory, languages, and other options from `config.yaml` or CLI.
- **File Filtering**: Excludes generated, build, test, and irrelevant files for each language.
- **Token-Based Chunking**: Splits large files into overlapping chunks based on token limits.
- **Granular Extraction**:
  - Rust: Public functions, structs, traits, impls, module exports
  - Python: Functions and classes (with docstrings)
  - JS/TS: Functions, arrow functions, and classes
- **Git Commit Extraction**: Optionally extracts commit history (diffs, messages, metadata) for each repo, filtered by relevant file types and limited by date/count.
- **Dataset Statistics**: Outputs detailed stats by language, type, and repo count.
- **Command-Line Interface**: Supports `--repos`, `--config`, and `--granular` flags for flexible usage.
- **Output**: Writes all samples to `all_data.jsonl` and stats to `dataset_stats.json` in the output directory.

## Dataset Generation Steps (Summary)
1. **Configuration**: Reads settings from `config.yaml` or CLI
2. **Repository Cloning**: Clones all specified repos if not already present
3. **File Collection**: Recursively collects code files for all configured languages
4. **File Processing**: Cleans, normalizes, and chunks files; extracts granular samples if enabled
5. **Git Commit Extraction**: (Optional) Extracts commit diffs/messages for each repo
6. **Sample Aggregation**: Combines all file-level, granular, and commit samples
7. **Saving Output**: Writes dataset and statistics to output directory
8. **Logging & Stats**: Logs progress and prints a summary of dataset statistics

## Usage Example
```bash
python3 shubhangi_dataset.py --repos <repo1> <repo2> ... --granular
```
- Use `--granular` to enable granular extraction for all languages (not just Rust).
- Edit `config.yaml` for more options.

---
For more details, see the script and config file comments.
