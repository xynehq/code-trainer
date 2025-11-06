import os
import json
import yaml
import re
import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import logging
import ast
import markdown

# Fix HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitCommitAnalyzer:
    """
    Git commit analyzer focusing on commit messages, file changes, and AST modifications
    Extracts commits from the last N days with essential development information
    """
    
    def __init__(self, repo_path: Path, max_tokens: int = 8192):
        self.repo_path = repo_path
        self.max_tokens = max_tokens
        self.tokenizer = None
        
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for token counting"""
        self.tokenizer = tokenizer
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            return len(text) // 4
    
    def run_git_command(self, cmd: List[str]) -> Optional[str]:
        """Run a git command and return the output"""
        try:
            result = subprocess.run(
                cmd, 
                cwd=str(self.repo_path), 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore'
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(f"Git command failed: {' '.join(cmd)}")
                return None
        except Exception as e:
            logger.error(f"Error running git command {' '.join(cmd)}: {e}")
            return None
    
    def get_recent_commits(self, days: int = 300) -> List[str]:
        """Get commit hashes from the last N days"""
        since_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        cmd = [
            'git', 'log', 
            f'--since={since_date}', 
            '--pretty=format:%H',
            '--all'
        ]
        
        output = self.run_git_command(cmd)
        if output:
            commits = [line.strip() for line in output.split('\n') if line.strip()]
            logger.info(f"Found {len(commits)} commits in the last {days} days")
            return commits
        return []
    
    def extract_commit_info(self, commit_hash: str) -> Dict:
        """Extract essential commit information"""
        cmd = [
            'git', 'show', '--pretty=format:%H%n%s%n%b%n%ad', 
            '--date=iso', '--no-patch', commit_hash
        ]
        
        output = self.run_git_command(cmd)
        if not output:
            return {}
        
        lines = output.split('\n')
        if len(lines) < 2:
            return {}
        
        full_hash = lines[0]
        subject = lines[1]
        
        # Extract body and date
        body_lines = []
        commit_date = ""
        
        for i, line in enumerate(lines[2:], 2):
            if i == len(lines) - 1:  # Last line should be the date
                commit_date = line
            else:
                body_lines.append(line)
        
        body = '\n'.join(body_lines).strip()
        
        return {
            'hash': full_hash,
            'short_hash': full_hash[:7],
            'subject': subject,
            'body': body,
            'message': f"{subject}\n\n{body}".strip() if body else subject,
            'date': commit_date
        }
    
    def extract_file_changes(self, commit_hash: str) -> Dict:
        """Extract detailed file changes with statistics"""
        # Get file change statistics
        cmd = ['git', 'show', '--numstat', '--format=', commit_hash]
        numstat_output = self.run_git_command(cmd)
        
        # Get file status changes
        cmd = ['git', 'show', '--name-status', commit_hash]
        status_output = self.run_git_command(cmd)
        
        files_added = []
        files_deleted = []
        files_modified = []
        files_renamed = []
        file_stats = {}
        
        total_insertions = 0
        total_deletions = 0
        
        # Parse numeric stats
        if numstat_output:
            for line in numstat_output.split('\n'):
                if not line.strip():
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 3:
                    try:
                        insertions = int(parts[0]) if parts[0] != '-' else 0
                        deletions = int(parts[1]) if parts[1] != '-' else 0
                        filename = parts[2]
                        
                        total_insertions += insertions
                        total_deletions += deletions
                        
                        file_stats[filename] = {
                            'insertions': insertions,
                            'deletions': deletions,
                            'net_change': insertions - deletions
                        }
                    except ValueError:
                        continue
        
        # Parse file status
        if status_output:
            for line in status_output.split('\n'):
                if not line.strip() or line.startswith('commit'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    status = parts[0]
                    filename = parts[1]
                    
                    if status == 'A':
                        files_added.append(filename)
                    elif status == 'D':
                        files_deleted.append(filename)
                    elif status == 'M':
                        files_modified.append(filename)
                    elif status.startswith('R'):
                        if len(parts) >= 3:
                            files_renamed.append({
                                'from': filename,
                                'to': parts[2]
                            })
        
        return {
            'summary': {
                'files_changed': len(file_stats),
                'insertions': total_insertions,
                'deletions': total_deletions,
                'net_change': total_insertions - total_deletions
            },
            'files': {
                'added': files_added,
                'deleted': files_deleted,
                'modified': files_modified,
                'renamed': files_renamed
            },
            'stats': file_stats
        }
    
    def extract_python_file_diff(self, commit_hash: str, file_path: str) -> Optional[Dict]:
        """Extract diff for a specific Python file"""
        cmd = ['git', 'show', commit_hash, '--', file_path]
        diff_output = self.run_git_command(cmd)
        
        if not diff_output:
            return None
        
        # Parse the diff to extract added and removed code sections
        lines = diff_output.split('\n')
        added_lines = []
        removed_lines = []
        context_lines = []
        
        in_diff = False
        for line in lines:
            if line.startswith('@@'):
                in_diff = True
                continue
            
            if not in_diff:
                continue
            
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:])
            elif line.startswith(' '):
                context_lines.append(line[1:])
        
        return {
            'file': file_path,
            'added_lines': added_lines,
            'removed_lines': removed_lines,
            'context_lines': context_lines,
            'total_added': len(added_lines),
            'total_removed': len(removed_lines)
        }
    
    def analyze_ast_changes(self, commit_hash: str, file_path: str) -> Dict:
        """Analyze AST changes for a Python file between commits"""
        # Get file content before the commit
        parent_cmd = ['git', 'show', f'{commit_hash}^:{file_path}']
        old_content = self.run_git_command(parent_cmd)
        
        # Get file content after the commit
        current_cmd = ['git', 'show', f'{commit_hash}:{file_path}']
        new_content = self.run_git_command(current_cmd)
        
        changes = {
            'functions_added': [],
            'functions_removed': [],
            'functions_modified': [],
            'classes_added': [],
            'classes_removed': [],
            'imports_added': [],
            'imports_removed': []
        }
        
        try:
            # Parse old AST
            old_functions = set()
            old_classes = set()
            old_imports = set()
            
            if old_content:
                try:
                    old_tree = ast.parse(old_content)
                    for node in ast.walk(old_tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            old_functions.add(node.name)
                        elif isinstance(node, ast.ClassDef):
                            old_classes.add(node.name)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                old_imports.add(f"import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ''
                            for alias in node.names:
                                old_imports.add(f"from {module} import {alias.name}")
                except SyntaxError:
                    pass
            
            # Parse new AST
            new_functions = set()
            new_classes = set()
            new_imports = set()
            
            if new_content:
                try:
                    new_tree = ast.parse(new_content)
                    for node in ast.walk(new_tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            new_functions.add(node.name)
                        elif isinstance(node, ast.ClassDef):
                            new_classes.add(node.name)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                new_imports.add(f"import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ''
                            for alias in node.names:
                                new_imports.add(f"from {module} import {alias.name}")
                except SyntaxError:
                    pass
            
            # Calculate differences
            changes['functions_added'] = list(new_functions - old_functions)
            changes['functions_removed'] = list(old_functions - new_functions)
            changes['functions_modified'] = list(old_functions & new_functions)  # Potentially modified
            changes['classes_added'] = list(new_classes - old_classes)
            changes['classes_removed'] = list(old_classes - new_classes)
            changes['imports_added'] = list(new_imports - old_imports)
            changes['imports_removed'] = list(old_imports - new_imports)
            
        except Exception as e:
            logger.debug(f"Error analyzing AST changes for {file_path}: {e}")
        
        return changes
    
    def create_commit_summary_sample(self, commit_hash: str) -> Optional[Dict]:
        """Create a focused commit summary sample"""
        commit_info = self.extract_commit_info(commit_hash)
        if not commit_info:
            return None
        
        file_changes = self.extract_file_changes(commit_hash)
        
        # Build structured text focused on essential information
        text = f"""# Commit Summary
# Date: {commit_info['date']}
# Hash: {commit_info['short_hash']}

# Commit Message:
{commit_info['message']}

# Change Summary:
# Files: {file_changes['summary']['files_changed']} modified
# Lines: +{file_changes['summary']['insertions']} -{file_changes['summary']['deletions']}
# Net Change: {file_changes['summary']['net_change']} lines"""
        
        # Add file changes
        if file_changes['files']['added']:
            text += f"\n\n# Files Added ({len(file_changes['files']['added'])}):"
            for file in file_changes['files']['added'][:15]:
                stats = file_changes['stats'].get(file, {})
                text += f"\n+ {file}"
                if stats:
                    text += f" (+{stats['insertions']} lines)"
        
        if file_changes['files']['modified']:
            text += f"\n\n# Files Modified ({len(file_changes['files']['modified'])}):"
            for file in file_changes['files']['modified'][:15]:
                stats = file_changes['stats'].get(file, {})
                text += f"\n~ {file}"
                if stats:
                    text += f" (+{stats['insertions']} -{stats['deletions']})"
        
        if file_changes['files']['deleted']:
            text += f"\n\n# Files Deleted ({len(file_changes['files']['deleted'])}):"
            for file in file_changes['files']['deleted'][:15]:
                text += f"\n- {file}"
        
        if file_changes['files']['renamed']:
            text += f"\n\n# Files Renamed ({len(file_changes['files']['renamed'])}):"
            for rename in file_changes['files']['renamed'][:10]:
                text += f"\n{rename['from']} -> {rename['to']}"
        
        return {
            'text': text,
            'commit_hash': commit_info['hash'],
            'short_hash': commit_info['short_hash'],
            'type': 'commit_summary',
            'commit_date': commit_info['date'],
            'files_changed': file_changes['summary']['files_changed'],
            'lines_added': file_changes['summary']['insertions'],
            'lines_deleted': file_changes['summary']['deletions'],
            'tokens': self.count_tokens(text),
            'language': 'git'
        }
    
    def create_file_change_samples(self, commit_hash: str) -> List[Dict]:
        """Create detailed samples for Python file changes with AST analysis"""
        commit_info = self.extract_commit_info(commit_hash)
        file_changes = self.extract_file_changes(commit_hash)
        samples = []
        
        # Focus on Python files for detailed analysis
        python_files = [f for f in file_changes['files']['modified'] + file_changes['files']['added'] 
                       if f.endswith('.py')]
        
        for file_path in python_files:
            # Get diff information
            diff_info = self.extract_python_file_diff(commit_hash, file_path)
            if not diff_info:
                continue
            
            # Get AST changes
            ast_changes = self.analyze_ast_changes(commit_hash, file_path)
            
            # Create structured text
            text = f"""# File Change Analysis
# Commit: {commit_info['short_hash']}
# Date: {commit_info['date']}
# File: {file_path}

# Commit Message:
{commit_info['subject']}

# Change Statistics:
# Lines Added: {diff_info['total_added']}
# Lines Removed: {diff_info['total_removed']}
# Net Change: {diff_info['total_added'] - diff_info['total_removed']}"""
            
            # Add AST-level changes
            if any(ast_changes.values()):
                text += "\n\n# Code Structure Changes:"
                
                if ast_changes['functions_added']:
                    text += f"\n# Functions Added: {', '.join(ast_changes['functions_added'])}"
                
                if ast_changes['functions_removed']:
                    text += f"\n# Functions Removed: {', '.join(ast_changes['functions_removed'])}"
                
                if ast_changes['classes_added']:
                    text += f"\n# Classes Added: {', '.join(ast_changes['classes_added'])}"
                
                if ast_changes['classes_removed']:
                    text += f"\n# Classes Removed: {', '.join(ast_changes['classes_removed'])}"
                
                if ast_changes['imports_added']:
                    text += f"\n# New Imports:\n"
                    for imp in ast_changes['imports_added'][:10]:
                        text += f"+ {imp}\n"
                
                if ast_changes['imports_removed']:
                    text += f"\n# Removed Imports:\n"
                    for imp in ast_changes['imports_removed'][:10]:
                        text += f"- {imp}\n"
            
            # Add actual code changes (limited)
            if diff_info['removed_lines']:
                text += f"\n\n# Code Removed ({len(diff_info['removed_lines'])} lines):"
                for line in diff_info['removed_lines'][:20]:
                    if line.strip():  # Skip empty lines
                        text += f"\n- {line.strip()}"
                if len(diff_info['removed_lines']) > 20:
                    text += f"\n# ... {len(diff_info['removed_lines']) - 20} more lines removed"
            
            if diff_info['added_lines']:
                text += f"\n\n# Code Added ({len(diff_info['added_lines'])} lines):"
                for line in diff_info['added_lines'][:20]:
                    if line.strip():  # Skip empty lines
                        text += f"\n+ {line.strip()}"
                if len(diff_info['added_lines']) > 20:
                    text += f"\n# ... {len(diff_info['added_lines']) - 20} more lines added"
            
            samples.append({
                'text': text,
                'commit_hash': commit_info['hash'],
                'short_hash': commit_info['short_hash'],
                'type': 'file_change_analysis',
                'file_path': file_path,
                'commit_date': commit_info['date'],
                'lines_added': diff_info['total_added'],
                'lines_removed': diff_info['total_removed'],
                'functions_added': len(ast_changes['functions_added']),
                'functions_removed': len(ast_changes['functions_removed']),
                'classes_added': len(ast_changes['classes_added']),
                'classes_removed': len(ast_changes['classes_removed']),
                'tokens': self.count_tokens(text),
                'language': 'python'
            })
        
        return samples
    
    def analyze_all_commits(self, days: int = 300) -> List[Dict]:
        """Analyze all commits from the last N days and create focused samples"""
        logger.info(f"🔍 Analyzing Git commits from the last {days} days...")
        
        commits = self.get_recent_commits(days)
        if not commits:
            logger.warning("No recent commits found")
            return []
        
        all_samples = []
        
        for commit_hash in tqdm(commits, desc="Processing commits", unit="commits"):
            # Create commit summary
            summary_sample = self.create_commit_summary_sample(commit_hash)
            if summary_sample:
                all_samples.append(summary_sample)
            
            # Create detailed file change samples
            file_samples = self.create_file_change_samples(commit_hash)
            all_samples.extend(file_samples)
        
        logger.info(f"✓ Created {len(all_samples)} commit-related samples")
        return all_samples


class EnhancedPythonDatasetPreparator:
    """
    Enhanced dataset preparator for CPT training on Python codebases
    Generates both file-level chunks AND granular function/class/decorator data
    Also processes documentation files for comprehensive understanding
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.repo_path = Path(self.config['repository']['path'])
        self.output_dir = Path(self.config['dataset']['output_dir'])
        self.max_tokens = self.config['dataset']['max_tokens']
        self.overlap_tokens = self.config['dataset']['overlap_tokens']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.git_analyzer = None
        
        # Check if commit analysis is enabled
        self.include_commits = self.config.get('dataset', {}).get('include_commits', False)
        self.commit_days = self.config.get('dataset', {}).get('commit_days', 300)
        
    def load_tokenizer(self):
        """Load tokenizer for token counting"""
        try:
            from transformers import AutoTokenizer
            tokenizer_name = self.config['model']['name']
            logger.info(f"Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            logger.info("✓ Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            logger.warning("Using character-based estimation (1 token ≈ 4 chars)")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            return len(text) // 4
    
    def initialize_git_analyzer(self):
        """Initialize Git commit analyzer if enabled"""
        if self.include_commits:
            try:
                self.git_analyzer = GitCommitAnalyzer(self.repo_path, self.max_tokens)
                self.git_analyzer.set_tokenizer(self.tokenizer)
                logger.info("✓ Git commit analyzer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Git analyzer: {e}")
                self.include_commits = False
                self.git_analyzer = None
    
    def is_excluded_path(self, path: Path) -> bool:
        """Check if path should be excluded"""
        path_str = str(path)
        exclude_patterns = [
            '__pycache__/', '*.pyc', '*.pyo', '*.pyd', 'dist/', 'build/', 
            '*.egg-info/', '.pytest_cache/', '.tox/', 'venv/', '.venv/', 
            'env/', '.env/', '.mypy_cache/', '.coverage', 'htmlcov/', 
            'node_modules/', '.DS_Store', '.git/'
        ]
        return any(pattern in path_str for pattern in exclude_patterns)
    
    def is_test_file(self, path: Path, content: str) -> bool:
        """Check if file is a test file"""
        path_str = str(path)
        
        # Check for test file patterns
        if '/tests/' in path_str or '/test/' in path_str:
            return True
        
        # Check for test file naming conventions
        filename = path_str.split('/')[-1]
        if filename.startswith('test_') or filename.endswith('_test.py'):
            return True
        
        # Check for test content markers
        test_markers = [
            'import unittest',
            'import pytest',
            'from unittest',
            'from pytest',
            'def test_',
            'class Test',
            '@pytest.',
            'unittest.TestCase',
            '@pytest.fixture'
        ]
        
        for marker in test_markers:
            if marker in content:
                return True
        
        return False
        
    def is_generated_file(self, content: str) -> bool:
        """Check if file appears to be generated"""
        first_lines = '\n'.join(content.split('\n')[:10])
        generation_markers = [
            '# generated',
            '# auto-generated',
            '# this file is generated',
            '# do not edit',
            'automatically generated',
            'autogenerated'
        ]
        return any(marker in first_lines.lower() for marker in generation_markers)
    
    def extract_module_name(self, rel_path: Path) -> str:
        """Extract Python module/package name from path"""
        parts = rel_path.parts
        
        # Handle common Python project structures
        
        # Check for src layout: src/package_name/...
        if 'src' in parts:
            src_idx = parts.index('src')
            if len(parts) > src_idx + 1:
                return parts[src_idx + 1]
        
        # Check for lib/libs layout: lib/package_name/...
        if 'lib' in parts or 'libs' in parts:
            lib_idx = parts.index('lib') if 'lib' in parts else parts.index('libs')
            if len(parts) > lib_idx + 1:
                return parts[lib_idx + 1]
        
        # Skip common non-module directories and get the next meaningful part
        skip_dirs = {'tests', 'test', 'docs', 'scripts', 'examples', 'tools', 'utils'}
        
        for i, part in enumerate(parts):
            if part not in skip_dirs:
                # For files directly in project root
                if i == len(parts) - 1:  # This is the filename
                    if len(parts) > 1:
                        return parts[0]  # Use parent directory
                    else:
                        return "root"  # Top-level file
                else:
                    return part  # First meaningful directory
        
        # If all parts are in skip_dirs, use the first one anyway
        if len(parts) > 1:
            return parts[0]
        
        return "root"
    
    def clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        content = content.replace('\r\n', '\n')
        content = re.sub(r'\n{3,}', '\n\n', content)
        lines = [line.rstrip() for line in content.split('\n')]
        return '\n'.join(lines)
    
    def clean_doc_content(self, content: str) -> str:
        """Clean doc content by removing images and other non-useful elements"""
        # Remove image references: ![alt text](image_url) and ![alt text][ref]
        content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', content)
        content = re.sub(r'!\[([^\]]*)\]\[[^\]]+\]', '', content)
        
        # Remove HTML image tags
        content = re.sub(r'<img[^>]*>', '', content, flags=re.IGNORECASE)
        
        # Remove badge/shield URLs that are typically images
        content = re.sub(r'\[!\[[^\]]*\]\([^\)]+\)\]\([^\)]+\)', '', content)
        
        # Remove standalone image URLs that might be on their own lines
        content = re.sub(r'^https?://[^\s]*\.(png|jpg|jpeg|gif|svg|webp).*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up multiple empty lines that might result from image removal
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove lines that are just whitespace
        lines = [line for line in content.split('\n') if line.strip() or line == '']
        
        return '\n'.join(lines)

    def get_comments(self, content: str) -> Dict[int, List[str]]:
        """Extract comments from the content"""
        comments_map = {}
        lines = content.split('\n')
        current_comments = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#'):
                current_comments.append(stripped)
            elif stripped:  # Non-empty line
                if current_comments:
                    comments_map[i + 1] = current_comments
                    current_comments = []
                else:
                    current_comments = []
        
        return comments_map

    def extract_quick_metadata(self, tree: ast.AST) -> Dict:
        """Quick metadata extraction for Python files using AST"""
        functions = 0
        classes = 0
        methods = 0
        decorators = 0
        
        # Count top-level items directly from module body
        for node in tree.body:
            # Count top-level functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions += 1
                decorators += len(node.decorator_list)
            # Count classes and their methods
            elif isinstance(node, ast.ClassDef):
                classes += 1
                decorators += len(node.decorator_list)
                # Count methods within classes
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods += 1
                        decorators += len(item.decorator_list)

        return {
            'functions': functions,
            'classes': classes,
            'methods': methods,
            'decorators': decorators
        }
    
    def extract_functions(self, tree: ast.AST, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract function definitions with documentation, comments, and parameters using AST"""
        samples = []
        
        # Get leading comments for each line
        comments_map = self.get_comments(content)
        
        # Only process top-level function definitions (not methods inside classes)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_name = node.name
                
                # Get decorators
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(f"@{decorator.id}")
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorators.append(f"@{decorator.func.id}")
                
                # Get docstring
                docstring = ast.get_docstring(node) or ""
                
                # Get function parameters
                params = []
                for arg in node.args.args:
                    param_str = arg.arg
                    if arg.annotation:
                        param_str += f": {ast.unparse(arg.annotation)}"
                    params.append(param_str)
                
                if node.args.vararg:
                    params.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    params.append(f"**{node.args.kwarg.arg}")
                
                # Get function return type
                returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                
                # Get comments before the function
                leading_comments = comments_map.get(node.lineno, [])
                
                # Build the function definition string
                is_async = isinstance(node, ast.AsyncFunctionDef)
                fn_def = "# Function Definition:\n" + f"{'async ' if is_async else ''}def {fn_name}({', '.join(params)}){returns}:"
                
                # Prepare structured text sample
                text = f"""# Function: {fn_name}"""
                if leading_comments:
                    text += "# Comments:\n" + "\n".join(leading_comments) + "\n\n"
                
                if decorators:
                    text += "# Decorators:\n" + "\n".join(decorators) + "\n"
                
                if docstring:
                    text += f"# Documentation: {docstring}\n\n"
                
                text += fn_def + "\n"
                
                text += "\n# Parameters:\n"
                if params:
                    for param in params:
                        text += f"#   - {param}\n"
                else:
                    text += "#   (none)\n"
                
                if returns:
                    text += f"\n# Returns: {returns}\n"

                samples.append({
                    "text": text,
                    "file_path": file_path,
                    "module": module,
                    "type": "function_definition",
                    "function_name": fn_name,
                    "function_params": ", ".join(params),
                    "tokens": self.count_tokens(text),
                    "language": "python"
                })
        
        return samples
    
    def extract_methods(self, tree: ast.AST, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract individual method definitions from classes with detailed information using AST"""
        samples = []
        
        # Get leading comments for each line
        comments_map = self.get_comments(content)
        
        # Find all classes and their methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # Process each method in the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = item.name
                        
                        # Get decorators
                        decorators = []
                        method_type = "method"  # default
                        
                        for decorator in item.decorator_list:
                            if isinstance(decorator, ast.Name):
                                decorators.append(f"@{decorator.id}")
                                # Determine method type based on decorators
                                if decorator.id == 'staticmethod':
                                    method_type = "static_method"
                                elif decorator.id == 'classmethod':
                                    method_type = "class_method"
                                elif decorator.id == 'property':
                                    method_type = "property"
                            elif isinstance(decorator, ast.Call):
                                if isinstance(decorator.func, ast.Name):
                                    decorators.append(f"@{decorator.func.id}")
                                else:
                                    decorators.append(f"@{ast.unparse(decorator)}")
                        
                        # Get docstring
                        docstring = ast.get_docstring(item) or ""
                        
                        # Get method parameters
                        params = []
                        for arg in item.args.args:
                            param_str = arg.arg
                            if arg.annotation:
                                param_str += f": {ast.unparse(arg.annotation)}"
                            params.append(param_str)
                        
                        if item.args.vararg:
                            params.append(f"*{item.args.vararg.arg}")
                        if item.args.kwarg:
                            params.append(f"**{item.args.kwarg.arg}")
                        
                        # Get return type
                        returns = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                        
                        # Get comments before the method
                        leading_comments = comments_map.get(item.lineno, [])
                        
                        # Determine if async
                        is_async = isinstance(item, ast.AsyncFunctionDef)
                        if is_async:
                            method_type = f"async_{method_type}"
                        
                        # Build method signature
                        method_def = f"{'async ' if is_async else ''}def {method_name}({', '.join(params)}){returns}:"
                        
                        # Prepare structured text sample
                        text = f"""# Method: {method_name}
# Class: {class_name}
# Type: {method_type}
"""
                        
                        if leading_comments:
                            text += "# Comments:\n" + "\n".join(leading_comments) + "\n\n"
                        
                        if decorators:
                            text += "# Decorators:\n"
                            for dec in decorators:
                                text += f"#   {dec}\n"
                            text += "\n"
                        
                        if docstring:
                            text += f"# Documentation:\n{docstring}\n\n"
                        
                        text += f"# Method Definition:\n{method_def}\n"
                        
                        text += "\n# Parameters:\n"
                        if params:
                            for param in params:
                                text += f"#   - {param}\n"
                        else:
                            text += "#   (none)\n"
                        
                        if returns:
                            text += f"\n# Returns: {returns[4:]}\n"
                        
                        samples.append({
                            "text": text,
                            "file_path": file_path,
                            "module": module,
                            "class_name": class_name,
                            "method_name": method_name,
                            "method_type": method_type,
                            "type": "method_definition",
                            "is_async": is_async,
                            "has_decorators": len(decorators) > 0,
                            "has_docstring": bool(docstring),
                            "parameter_count": len(params),
                            "has_return_type": bool(returns),
                            "decorators": decorators,
                            "tokens": self.count_tokens(text),
                            "language": "python"
                        })
        
        return samples

    def extract_classes(self, tree: ast.AST, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract class definitions with methods and docstrings using AST"""
        samples = []
        
        # Get leading comments for each line
        comments_map = self.get_comments(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # Get base classes if any
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(ast.unparse(base))
                
                # Get decorators
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(f"@{decorator.id}")
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorators.append(f"@{decorator.func.id}")
                        else:
                            decorators.append(f"@{ast.unparse(decorator)}")
                
                # Get docstring
                docstring = ast.get_docstring(node) or ""
                
                # Count methods and their types in class
                method_count = 0
                async_method_count = 0
                property_count = 0
                static_method_count = 0
                class_method_count = 0
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_count += 1
                        # Check for property/static/classmethod decorators
                        for dec in item.decorator_list:
                            if isinstance(dec, ast.Name):
                                if dec.id == 'property':
                                    property_count += 1
                                    method_count -= 1
                                elif dec.id == 'staticmethod':
                                    static_method_count += 1
                                    method_count -= 1
                                elif dec.id == 'classmethod':
                                    class_method_count += 1
                                    method_count -= 1
                    elif isinstance(item, ast.AsyncFunctionDef):
                        async_method_count += 1
                        # Check for decorators on async methods too
                        for dec in item.decorator_list:
                            if isinstance(dec, ast.Name):
                                if dec.id == 'property':
                                    property_count += 1
                                    async_method_count -= 1
                                elif dec.id == 'staticmethod':
                                    static_method_count += 1
                                    async_method_count -= 1
                                elif dec.id == 'classmethod':
                                    class_method_count += 1
                                    async_method_count -= 1
                
                # Get leading comments for the class
                leading_comments = comments_map.get(node.lineno, [])
                
                # Build class definition string
                class_def = f"class {class_name}"
                if bases:
                    class_def += f"({', '.join(bases)})"
                class_def += ":"
                
                # Prepare structured text sample
                text = f"""# Class: {class_name}
# Methods: {method_count + async_method_count}
# Properties: {property_count}
# Static Methods: {static_method_count}
# Class Methods: {class_method_count}
"""
                if bases:
                    text += f"# Base Classes: {', '.join(bases)}\n"
                
                if leading_comments:
                    text += "# Comments:\n" + "\n".join(leading_comments) + "\n\n"
                
                if decorators:
                    text += "\n".join(decorators) + "\n"
                
                if docstring:
                    text += f"# Documentation: {docstring}\n\n"
                
                text += class_def
                
                samples.append({
                    "text": text,
                    "file_path": file_path,
                    "module": module,
                    "type": "class_definition",
                    "class_name": class_name,
                    "method_count": method_count + async_method_count,
                    "property_count": property_count,
                    "static_method_count": static_method_count,
                    "class_method_count": class_method_count,
                    "has_base_classes": len(bases) > 0,
                    "base_classes": bases,
                    "tokens": self.count_tokens(text),
                    "language": "python"
                })
        
        return samples
    
    def extract_module_structure(self, tree: ast.AST, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract imports, module-level variables, and other module-level constructs using AST"""
        samples = []
        
        # Extract module docstring only if it's the first thing in the module
        module_docstring = ""
        if tree.body and isinstance(tree.body[0], ast.Expr):
            value_node = tree.body[0].value
            # Handle both old ast.Str (Python < 3.8) and new ast.Constant (Python >= 3.8)
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                module_docstring = value_node.value
            elif hasattr(ast, 'Str') and isinstance(value_node, ast.Str):
                module_docstring = value_node.s
        
        # Collect imports
        standard_imports = []
        third_party_imports = []
        relative_imports = []
        
        # Collect module-level definitions
        constants = []
        variables = []
        type_aliases = []
        
        for node in tree.body:
            # Handle imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    import_str = f"import {name.name}"
                    if name.asname:
                        import_str += f" as {name.asname}"
                    standard_imports.append(import_str)
            
            elif isinstance(node, ast.ImportFrom):
                import_str = f"from {'.' * node.level if node.level else ''}{node.module or ''} import "
                names = []
                for name in node.names:
                    alias = f"{name.name}"
                    if name.asname:
                        alias += f" as {name.asname}"
                    names.append(alias)
                import_str += ", ".join(names)
                
                if node.level > 0:  # Relative import
                    relative_imports.append(import_str)
                else:
                    third_party_imports.append(import_str)
            
            # Handle assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():  # Constants
                            value = ast.unparse(node.value)
                            constants.append(f"{target.id} = {value}")
                        else:  # Variables
                            value = ast.unparse(node.value)
                            variables.append(f"{target.id} = {value}")
            
            # Handle annotated assignments
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    annotation = ast.unparse(node.annotation)
                    value = ast.unparse(node.value) if node.value else None
                    if name.isupper():  # Constants with type hints
                        if value:
                            constants.append(f"{name}: {annotation} = {value}")
                        else:
                            constants.append(f"{name}: {annotation}")
                    else:  # Variables with type hints
                        if value:
                            variables.append(f"{name}: {annotation} = {value}")
                        else:
                            variables.append(f"{name}: {annotation}")
            
            # Handle type aliases
            elif (isinstance(node, ast.Assign) and 
                  isinstance(node.targets[0], ast.Name) and 
                  isinstance(node.value, (ast.Subscript, ast.Name))):
                type_aliases.append(ast.unparse(node))
        
        # Prepare the text output
        text = f"""# Module Structure
# File: {file_path}
# Module: {module}
"""
        if module_docstring:
            text += f"\n# Module Documentation:\n{module_docstring}\n"
        
        if standard_imports:
            text += "\n# Standard Library Imports:\n"
            for imp in standard_imports[:10]:
                text += f"{imp}\n"
            if len(standard_imports) > 10:
                text += f"# ... and {len(standard_imports) - 10} more\n"
        
        if third_party_imports:
            text += "\n# Third-Party Imports:\n"
            for imp in third_party_imports[:10]:
                text += f"{imp}\n"
            if len(third_party_imports) > 10:
                text += f"# ... and {len(third_party_imports) - 10} more\n"
        
        if relative_imports:
            text += "\n# Relative Imports:\n"
            for imp in relative_imports[:10]:
                text += f"{imp}\n"
            if len(relative_imports) > 10:
                text += f"# ... and {len(relative_imports) - 10} more\n"
        
        if type_aliases:
            text += "\n# Type Aliases:\n"
            for alias in type_aliases[:10]:
                text += f"{alias}\n"
            if len(type_aliases) > 10:
                text += f"# ... and {len(type_aliases) - 10} more\n"
        
        if constants:
            text += "\n# Constants:\n"
            for const in constants[:10]:
                text += f"{const}\n"
            if len(constants) > 10:
                text += f"# ... and {len(constants) - 10} more\n"
        
        if variables:
            text += "\n# Module-Level Variables:\n"
            for var in variables[:10]:
                text += f"{var}\n"
            if len(variables) > 10:
                text += f"# ... and {len(variables) - 10} more\n"
        
        samples.append({
            "text": text,
            "file_path": file_path,
            "module": module,
            "type": "module_structure",
            "import_count": len(standard_imports) + len(third_party_imports) + len(relative_imports),
            "constant_count": len(constants),
            "variable_count": len(variables),
            "type_alias_count": len(type_aliases),
            "has_docstring": bool(module_docstring),
            "tokens": self.count_tokens(text),
            "language": "python",
        })
        
        return samples

    def extract_ast_information(self, tree: ast.AST, file_path: str, module: str) -> List[Dict]:
        """Extract comprehensive AST information with detailed node analysis and chunking"""
        samples = []
        
        # Collect comprehensive AST information
        ast_info = self._analyze_ast_tree(tree)
        
        # Create the comprehensive AST text
        ast_text = self._format_ast_analysis(ast_info, file_path, module)
        
        # Check if chunking is needed
        total_tokens = self.count_tokens(ast_text)
        
        if total_tokens <= self.max_tokens:
            # Single AST sample
            samples.append({
                "text": ast_text,
                "file_path": file_path,
                "module": module,
                "type": "ast_analysis",
                "node_count": ast_info['total_nodes'],
                "node_types": list(ast_info['node_types'].keys()),
                "max_depth": ast_info['max_depth'],
                "tokens": total_tokens,
                "language": "python"
            })
        else:
            # Chunk the AST information
            chunks = self._chunk_ast_information(ast_text, file_path, module, ast_info)
            samples.extend(chunks)
        
        return samples
    
    def _analyze_ast_tree(self, tree: ast.AST) -> Dict:
        """Analyze the AST tree and extract comprehensive information"""
        node_types = {}
        total_nodes = 0
        max_depth = 0
        
        # Lists to store different types of constructs
        control_flow = []
        expressions = []
        statements = []
        definitions = []
        
        def analyze_node(node, depth=0):
            nonlocal total_nodes, max_depth
            
            total_nodes += 1
            max_depth = max(max_depth, depth)
            
            node_type = type(node).__name__
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Categorize nodes and extract information
            if isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                control_flow.append({
                    'type': node_type,
                    'line': getattr(node, 'lineno', 0),
                    'depth': depth
                })
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.append({
                    'type': node_type,
                    'name': node.name,
                    'line': getattr(node, 'lineno', 0),
                    'depth': depth,
                    'decorators': len(node.decorator_list) if hasattr(node, 'decorator_list') else 0
                })
            
            elif isinstance(node, (ast.Call, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                expressions.append({
                    'type': node_type,
                    'line': getattr(node, 'lineno', 0),
                    'depth': depth
                })
            
            elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Raise, ast.Assert)):
                statements.append({
                    'type': node_type,
                    'line': getattr(node, 'lineno', 0),
                    'depth': depth
                })
            
            # Recursively analyze child nodes
            for child in ast.iter_child_nodes(node):
                analyze_node(child, depth + 1)
        
        analyze_node(tree)
        
        return {
            'total_nodes': total_nodes,
            'node_types': node_types,
            'max_depth': max_depth,
            'control_flow': control_flow,
            'expressions': expressions,
            'statements': statements,
            'definitions': definitions
        }
    
    def _format_ast_analysis(self, ast_info: Dict, file_path: str, module: str) -> str:
        """Format AST analysis into structured text"""
        text = f"""# AST Analysis
# File: {file_path}
# Module: {module}
# Total Nodes: {ast_info['total_nodes']}
# Max Depth: {ast_info['max_depth']}

# Node Type Distribution:
"""
        
        # Sort node types by frequency
        sorted_types = sorted(ast_info['node_types'].items(), key=lambda x: x[1], reverse=True)
        for node_type, count in sorted_types:
            percentage = (count / ast_info['total_nodes']) * 100
            text += f"#   {node_type}: {count} ({percentage:.1f}%)\n"
        
        # Control Flow Analysis
        if ast_info['control_flow']:
            text += "\n# Control Flow Constructs:\n"
            for item in ast_info['control_flow'][:20]:  # Limit to first 20
                text += f"#   Line {item['line']}: {item['type']} (depth: {item['depth']})\n"
            if len(ast_info['control_flow']) > 20:
                text += f"#   ... and {len(ast_info['control_flow']) - 20} more\n"
        
        # Function and Class Definitions
        if ast_info['definitions']:
            text += "\n# Definitions:\n"
            for item in ast_info['definitions'][:30]:  # Limit to first 30
                decorator_info = f" [{item['decorators']} decorators]" if item['decorators'] > 0 else ""
                text += f"#   Line {item['line']}: {item['type']} '{item['name']}' (depth: {item['depth']}){decorator_info}\n"
            if len(ast_info['definitions']) > 30:
                text += f"#   ... and {len(ast_info['definitions']) - 30} more\n"
        
        # Expression Analysis
        if ast_info['expressions']:
            text += "\n# Complex Expressions:\n"
            expr_summary = {}
            for item in ast_info['expressions']:
                expr_type = item['type']
                expr_summary[expr_type] = expr_summary.get(expr_type, 0) + 1
            
            for expr_type, count in sorted(expr_summary.items(), key=lambda x: x[1], reverse=True):
                text += f"#   {expr_type}: {count} occurrences\n"
        
        # Statement Analysis
        if ast_info['statements']:
            text += "\n# Control Statements:\n"
            stmt_summary = {}
            for item in ast_info['statements']:
                stmt_type = item['type']
                stmt_summary[stmt_type] = stmt_summary.get(stmt_type, 0) + 1
            
            for stmt_type, count in sorted(stmt_summary.items(), key=lambda x: x[1], reverse=True):
                text += f"#   {stmt_type}: {count} occurrences\n"
        
        # Code Quality Metrics
        text += f"\n# Code Quality Metrics:\n"
        text += f"#   Average depth: {ast_info['max_depth'] / max(len(ast_info['definitions']), 1):.1f}\n"
        text += f"#   Node type diversity: {len(ast_info['node_types'])} unique types\n"
        
        return text
    
    def _chunk_ast_information(self, ast_text: str, file_path: str, module: str, ast_info: Dict) -> List[Dict]:
        """Break AST information into manageable chunks"""
        chunks = []
        lines = ast_text.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        
        # Always include the header in each chunk
        header_lines = []
        for i, line in enumerate(lines):
            if line.startswith('# Node Type Distribution:'):
                header_lines = lines[:i]
                remaining_lines = lines[i:]
                break
        else:
            header_lines = lines[:10] if len(lines) > 10 else lines
            remaining_lines = lines[10:] if len(lines) > 10 else []
        
        header_text = '\n'.join(header_lines)
        header_tokens = self.count_tokens(header_text)
        
        for line in remaining_lines:
            line_tokens = self.count_tokens(line + '\n')
            
            if current_tokens + line_tokens + header_tokens > self.max_tokens and current_chunk:
                # Create chunk with header
                chunk_text = header_text + '\n' + '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'file_path': f"{file_path}#ast_chunk{chunk_idx}",
                    'module': module,
                    'type': 'ast_analysis_chunk',
                    'chunk_index': chunk_idx,
                    'total_ast_nodes': ast_info['total_nodes'],
                    'tokens': self.count_tokens(chunk_text),
                    "language": "python"
                })
                
                # Keep some overlap for context
                overlap_lines = current_chunk[-5:] if len(current_chunk) > 5 else current_chunk
                current_chunk = overlap_lines
                current_tokens = self.count_tokens('\n'.join(overlap_lines))
                chunk_idx += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add final chunk if there's remaining content
        if current_chunk:
            chunk_text = header_text + '\n' + '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'file_path': f"{file_path}#ast_chunk{chunk_idx}",
                'module': module,
                'type': 'ast_analysis_chunk',
                'chunk_index': chunk_idx,
                'total_ast_nodes': ast_info['total_nodes'],
                'tokens': self.count_tokens(chunk_text),
                "language": "python"
            })
        
        return chunks
    
    def collect_doc_files(self) -> List[Path]:
        """Collect relevant documentation files from repository"""
        logger.info(f"Collecting documentation files from {self.repo_path}")
        documentation_files = []
        
        # Common documentation file extensions
        for pattern in ["*.md", "*.markdown", "*.mdown", "*.mkd", "*.mdx", "*.txt", "*.rst", "*.adoc", "*.asciidoc", "*.org"]:
            documentation_files.extend(self.repo_path.rglob(pattern))
        
        # Filter out excluded paths and irrelevant files
        relevant_files = []
        for f in documentation_files:
            if self.is_excluded_path(f):
                continue
            if self.is_relevant_doc(f):
                relevant_files.append(f)
        
        logger.info(f"✓ Found {len(relevant_files)} relevant documentation files")
        return relevant_files
    
    def is_relevant_doc(self, file_path: Path) -> bool:
        """Check if doc file is relevant for the dataset"""
        filename = file_path.name.lower()
        path_str = str(file_path).lower()
        
        # Exclude irrelevant patterns
        irrelevant_patterns = [
            'node_modules', '.github', 'vendor', 'build', 'dist',
            'test', 'spec', 'mock', '.git', 'cache'
        ]
        
        # Check if path contains irrelevant patterns
        for pattern in irrelevant_patterns:
            if pattern in path_str:
                return False
    
        
        return True
    
    def process_doc_file(self, file_path: Path) -> List[Dict]:
        """Process a single doc file and extract content with chunking"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = file_path.relative_to(self.repo_path)
            
            # Clean doc content and filter out images
            content = self.clean_doc_content(content)
            
            # Create file header
            file_header = f"""# Documentation: {rel_path}
# Type: Doc File
# Lines: {len(content.split('\n'))}
# Words: {len(content.split())}

"""
            
            full_content = file_header + content
            
            # Chunk the content similar to Python files
            return self.chunk_doc_content(full_content, str(rel_path))
            
        except Exception as e:
            logger.debug(f"Error processing doc file {file_path}: {e}")
            return []
    
    def chunk_doc_content(self, content: str, file_path: str) -> List[Dict]:
        """Split doc content into chunks with overlap if needed"""
        tokens = self.count_tokens(content)
        
        if tokens <= self.max_tokens:
            return [{
                'text': content,
                'file_path': file_path,
                'type': 'doc_file',
                'tokens': tokens,
                'language': 'doc'
            }]
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line + '\n')
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'file_path': f"{file_path}#chunk{chunk_idx}",
                    'type': 'doc_chunk',
                    'tokens': current_tokens,
                    'language': 'doc'
                })
                
                # Keep some overlap for context (similar to Python chunking)
                overlap_lines = current_chunk[-20:] if len(current_chunk) > 20 else current_chunk
                current_chunk = overlap_lines
                current_tokens = self.count_tokens('\n'.join(overlap_lines))
                chunk_idx += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'file_path': f"{file_path}#chunk{chunk_idx}",
                'type': 'doc_chunk',
                'tokens': current_tokens,
                'language': 'doc'
            })
        
        return chunks
    
    def create_file_header(self, rel_path: str, module: str, metadata: Dict) -> str:
        """Create context header for file"""
        header = f"# File: {rel_path}\n"
        header += f"# Module: {module}\n"
        
        if metadata['functions'] > 0:
            header += f"# Functions: {metadata['functions']}\n"
        if metadata['classes'] > 0:
            header += f"# Classes: {metadata['classes']}\n"
        if metadata['decorators'] > 0:
            header += f"# Decorators: {metadata['decorators']}\n"
        if metadata['methods'] > 0:
            header += f"# Methods: {metadata['methods']}\n"
        header += "\n"
        return header
    
    def chunk_content(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Split content into chunks if needed"""
        tokens = self.count_tokens(content)
        
        if tokens <= self.max_tokens:
            return [{
                'text': content,
                'file_path': file_path,
                'module': module,
                'tokens': tokens,
                'type': 'full_file'
            }]
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line + '\n')
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'file_path': f"{file_path}#chunk{chunk_idx}",
                    'module': module,
                    'tokens': current_tokens,
                    'type': 'chunk',
                    "language": "python"
                })
                
                # Keep some overlap
                overlap_lines = current_chunk[-20:] if len(current_chunk) > 20 else current_chunk
                current_chunk = overlap_lines
                current_tokens = self.count_tokens('\n'.join(overlap_lines))
                chunk_idx += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'file_path': f"{file_path}#chunk{chunk_idx}",
                'module': module,
                'tokens': current_tokens,
                'type': 'chunk',
                "language": "python"
            })
        
        return chunks
    
    def process_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a single Python file
        Returns: (file_level_samples, granular_samples)
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = file_path.relative_to(self.repo_path)
            
            if self.is_excluded_path(file_path):
                return [], []
            
            if self.is_generated_file(content):
                return [], []
            
            is_test = self.is_test_file(file_path, content)
            if is_test and not self.config['dataset']['include_tests']:
                return [], []
            
            content = self.clean_content(content)
            module = self.extract_module_name(rel_path)
            
            # Parse AST once with error handling
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                logger.warning(f"Syntax error in file {file_path}: {e}")
                return [], []
            
            # Extract metadata using the parsed tree
            metadata = self.extract_quick_metadata(tree)
            
            # Create file header and full content
            header = self.create_file_header(str(rel_path), module, metadata)
            full_content = header + content
            
            # Create file-level chunks
            file_samples = self.chunk_content(full_content, str(rel_path), module)
            
            # Extract granular samples using the parsed tree
            granular_samples = []
            
            if not is_test and (metadata['functions'] > 0 or metadata['classes'] > 0):
                granular_samples.extend(
                    self.extract_functions(tree, content, str(rel_path), module)
                )
                
                granular_samples.extend(
                    self.extract_classes(tree, content, str(rel_path), module)
                )
                
                granular_samples.extend(
                    self.extract_methods(tree, content, str(rel_path), module)
                )
                
                granular_samples.extend(
                    self.extract_module_structure(tree, content, str(rel_path), module)
                )
            
            # Always extract AST information for non-test files (regardless of functions/classes)
            if not is_test:
                granular_samples.extend(
                    self.extract_ast_information(tree, str(rel_path), module)
                )
            
            return file_samples, granular_samples
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return [], []
    
    def collect_python_files(self) -> List[Path]:
        """Collect all Python files from repository"""
        logger.info(f"Collecting Python files from {self.repo_path}")
        python_files = list(self.repo_path.rglob("*.py"))
        logger.info(f"✓ Found {len(python_files)} Python files")
        return python_files
    
    def process_all_files(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all Python and doc files
        Returns: (file_level_samples, granular_samples)
        """
        python_files = self.collect_python_files()
        doc_files = self.collect_doc_files()
        
        all_file_samples = []
        all_granular_samples = []
        
        # Process Python files
        logger.info("Processing Python files for file-level AND granular data...")
        for file_path in tqdm(python_files, desc="Processing Python", unit="files"):
            file_samples, granular_samples = self.process_file(file_path)
            all_file_samples.extend(file_samples)
            all_granular_samples.extend(granular_samples)
        
        # Process doc files
        logger.info("Processing doc files for documentation data...")
        for file_path in tqdm(doc_files, desc="Processing doc", unit="files"):
            doc_samples = self.process_doc_file(file_path)
            all_file_samples.extend(doc_samples)  # Doc samples go into file_samples
        
        logger.info(f"✓ Created {len(all_file_samples)} file-level samples (including doc)")
        logger.info(f"✓ Created {len(all_granular_samples)} granular samples")
        logger.info(f"✓ Total: {len(all_file_samples) + len(all_granular_samples)} samples")
        
        return all_file_samples, all_granular_samples
    
    def save_dataset(self, file_samples: List[Dict], granular_samples: List[Dict], commit_samples: List[Dict] = None):
        """Save combined dataset to a single all_data.jsonl file"""
        
        if commit_samples is None:
            commit_samples = []
        
        all_samples = file_samples + granular_samples + commit_samples
        random.shuffle(all_samples)
        
        output_path = self.output_dir / "all_data.jsonl"
        logger.info(f"Saving combined dataset to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Generate statistics
        stats = {
            'total_samples': len(all_samples),
            'file_level_samples': len(file_samples),
            'granular_samples': len(granular_samples),
            'commit_samples': len(commit_samples),
            'total_tokens': sum(s.get('tokens', 0) for s in all_samples),
            'sample_types': {},
            'modules': list(set(s.get('module', 'unknown') for s in all_samples)),
        }
        
        for sample in all_samples:
            stype = sample.get('type', 'unknown')
            stats['sample_types'][stype] = stats['sample_types'].get(stype, 0) + 1
        
        stats['granular_breakdown'] = {
            'functions': stats['sample_types'].get('function_definition', 0),
            'classes': stats['sample_types'].get('class_definition', 0),
            'methods': stats['sample_types'].get('method_definition', 0),
            'module_structure': stats['sample_types'].get('module_structure', 0),
            'ast_analysis': stats['sample_types'].get('ast_analysis', 0),
            'ast_analysis_chunks': stats['sample_types'].get('ast_analysis_chunk', 0)
        }
        
        stats['documentation_breakdown'] = {
            'doc_files': stats['sample_types'].get('doc_file', 0),
            'doc_chunks': stats['sample_types'].get('doc_chunk', 0)
        }
        
        stats['commit_breakdown'] = {
            'commit_summaries': stats['sample_types'].get('commit_summary', 0),
            'file_change_analysis': stats['sample_types'].get('file_change_analysis', 0)
        }
        
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Dataset statistics saved to {stats_path}")
        self._print_stats(stats)
        
        return output_path
    
    def _print_stats(self, stats: Dict):
        """Print dataset statistics"""
        logger.info(f"\n{'='*70}")
        logger.info("📊 Python Dataset Summary:")
        logger.info(f"{'='*70}")
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"    └─ File-level: {stats['file_level_samples']:,}")
        logger.info(f"    └─ Granular: {stats['granular_samples']:,}")
        if stats['commit_samples'] > 0:
            logger.info(f"    └─ Commit: {stats['commit_samples']:,}")
        logger.info(f"  Total tokens: {stats['total_tokens']:,}")
        
        logger.info(f"\n  📂 File-level samples:")
        for stype in ['full_file', 'chunk']:
            count = stats['sample_types'].get(stype, 0)
            if count > 0:
                logger.info(f"    └─ {stype}: {count:,}")
        
        logger.info(f"\n  🐍 Python granular samples:")
        for name, count in stats['granular_breakdown'].items():
            if count > 0:
                logger.info(f"    └─ {name}: {count:,}")
        
        # Show AST-specific information
        ast_total = stats['granular_breakdown']['ast_analysis'] + stats['granular_breakdown']['ast_analysis_chunks']
        if ast_total > 0:
            logger.info(f"\n  🌳 AST Analysis samples: {ast_total:,}")
            if stats['granular_breakdown']['ast_analysis_chunks'] > 0:
                logger.info(f"    └─ Chunked AST samples: {stats['granular_breakdown']['ast_analysis_chunks']:,}")
        
        # Show documentation-specific information
        if 'documentation_breakdown' in stats:
            doc_total = sum(stats['documentation_breakdown'].values())
            if doc_total > 0:
                logger.info(f"\n  📚 Documentation samples: {doc_total:,}")
                for name, count in stats['documentation_breakdown'].items():
                    if count > 0:
                        logger.info(f"    └─ {name.replace('_', ' ').title()}: {count:,}")
        
        # Show commit-specific information
        if 'commit_breakdown' in stats:
            commit_total = sum(stats['commit_breakdown'].values())
            if commit_total > 0:
                logger.info(f"\n  🔄 Git commit samples: {commit_total:,}")
                for name, count in stats['commit_breakdown'].items():
                    if count > 0:
                        logger.info(f"    └─ {name.replace('_', ' ').title()}: {count:,}")
        
        logger.info(f"\n  📦 Python modules covered: {len(stats['modules'])}")
        logger.info(f"{'='*70}")
    
    def prepare(self):
        """Main preparation pipeline - generates both file-level and granular data"""
        logger.info("=" * 70)
        logger.info("🚀 Enhanced Dataset Preparation")
        logger.info("   Generating: File-level chunks + Granular function/class/module/AST data + Documentation")
        logger.info("=" * 70)
        
        self.load_tokenizer()
        self.initialize_git_analyzer()
        
        file_samples, granular_samples = self.process_all_files()
        
        # Process Git commits if enabled
        commit_samples = []
        if self.include_commits and self.git_analyzer:
            commit_samples = self.git_analyzer.analyze_all_commits(self.commit_days)
        
        if not file_samples and not granular_samples and not commit_samples:
            logger.error("No samples generated! Check your configuration.")
            return
        
        output_path = self.save_dataset(file_samples, granular_samples, commit_samples)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Dataset preparation complete!")
        logger.info(f"   📄 Output: {output_path}")
        logger.info(f"   📊 Total: {len(file_samples) + len(granular_samples)+len(commit_samples):,} samples")
        logger.info("=" * 70)
        logger.info("\n🎯 Next step: Run training with ./train_direct.sh")


def main():
    """Main entry point"""
    preparator = EnhancedPythonDatasetPreparator("config.yaml")
    preparator.prepare()


if __name__ == "__main__":
    main()
