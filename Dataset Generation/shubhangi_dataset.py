import os
import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import logging
import subprocess
import argparse

# Suppress tokenizer parallelism warnings when using git subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDatasetPreparator:
    """
    Enhanced dataset preparator for CPT training - supports multiple repos and languages
    Generates both file-level chunks AND granular function/struct/trait data (Rust only)
    """
    
    def __init__(self, config_path: str = "config.yaml", repo_urls: List[str] = None, enable_granular_all: bool = False):
        """Initialize dataset preparator with optional repo URLs and granular flag"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        repo_config = self.config['repository']
        
        if repo_urls:
            self.repo_paths = []
            for url in repo_urls:
                repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
                repo_path = Path(repo_name)
                self.clone_repository(url, repo_path)
                self.repo_paths.append(repo_path)
        elif 'paths' in repo_config and repo_config['paths']:
            self.repo_paths = [Path(p) for p in repo_config['paths']]
        elif 'urls' in repo_config and repo_config['urls']:
            self.repo_paths = []
            for url in repo_config['urls']:
                repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
                repo_path = Path(repo_name)
                self.clone_repository(url, repo_path)
                self.repo_paths.append(repo_path)
        elif 'path' in repo_config:
            self.repo_paths = [Path(repo_config['path'])]
        else:
            raise ValueError("No repository path(s) or URL(s) found in config or command-line")
        
        self.languages = self.config['dataset'].get('languages', ['rust'])
        
        self.enable_granular_all = enable_granular_all
        
        self.output_dir = Path(self.config['dataset']['output_dir'])
        self.max_tokens = self.config['dataset']['max_tokens']
        self.overlap_tokens = self.config['dataset']['overlap_tokens']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        
        logger.info(f"Configured for {len(self.repo_paths)} repository(ies)")
        logger.info(f"Processing languages: {', '.join(self.languages)}")
        if self.enable_granular_all:
            logger.info(f"Granular extraction: ENABLED for all languages")
        else:
            logger.info(f"Granular extraction: Rust only (use --granular for all languages)")

    
    def clone_repository(self, repo_url: str, target_dir: Path) -> bool:
        """Clone a GitHub repository to the target directory"""
        try:
            if target_dir.exists():
                logger.info(f"Repository already exists at {target_dir}, skipping clone...")
                return True
            
            logger.info(f"Cloning {repo_url} into {target_dir}...")
            subprocess.run(['git', 'clone', repo_url, str(target_dir)], check=True, capture_output=True)
            logger.info(f"✓ Successfully cloned {repo_url}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {repo_url}: {e}")
            logger.error(f"Error output: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error cloning {repo_url}: {e}")
            return False
    
    def detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        path_str = str(file_path).lower()
        
        if path_str.endswith('.rs'):
            return 'rust'
        elif path_str.endswith('.py'):
            return 'python'
        elif path_str.endswith(('.js', '.jsx')):
            return 'javascript'
        elif path_str.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif path_str.endswith(('.html', '.htm')):
            return 'html'
        elif path_str.endswith(('.css', '.scss', '.sass', '.less')):
            return 'css'
        else:
            return 'unknown'
        
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
    
    def is_excluded_path(self, path: Path, language: str = 'rust') -> bool:
        """Check if path should be excluded (language-aware)"""
        path_str = str(path)
        
        common_excludes = ['/build/', '/.git/', '/dist/']
        
        if language == 'rust':
            excludes = common_excludes + ['/target/', '/generated/', '.gen.rs']
        elif language == 'python':
            excludes = common_excludes + ['__pycache__', '.pyc', '/venv/', '/.venv/']
        elif language in ['javascript', 'typescript']:
            excludes = common_excludes + ['/node_modules/', '/coverage/', '/.next/']
        elif language in ['html', 'css']:
            excludes = common_excludes + ['/vendor/', '/bower_components/']
        else:
            excludes = common_excludes
        
        return any(pattern in path_str for pattern in excludes)
    
    def is_test_file(self, path: Path, content: str) -> bool:
        """Check if file is a test file"""
        path_str = str(path)
        if '/tests/' in path_str or path_str.endswith('_test.rs'):
            return True
        if '#[cfg(test)]' in content or 'mod tests {' in content:
            return True
        return False
    
    def is_generated_file(self, content: str) -> bool:
        """Check if file appears to be generated"""
        first_lines = '\n'.join(content.split('\n')[:10])
        generation_markers = [
            '@generated',
            'auto-generated',
            'this file is generated',
            'do not edit',
        ]
        return any(marker in first_lines.lower() for marker in generation_markers)
    
    def extract_module_name(self, rel_path: Path) -> str:
        """Extract module/crate name from path"""
        parts = rel_path.parts
        if 'crates' in parts:
            crate_idx = parts.index('crates')
            if len(parts) > crate_idx + 1:
                return parts[crate_idx + 1]
        if len(parts) > 1:
            return parts[0]
        return "root"
    
    def clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        content = content.replace('\r\n', '\n')
        content = re.sub(r'\n{3,}', '\n\n', content)
        lines = [line.rstrip() for line in content.split('\n')]
        return '\n'.join(lines)
    
    def extract_quick_metadata(self, content: str) -> Dict:
        """Quick metadata extraction without deep parsing"""
        pub_fn_count = len(re.findall(r'\bpub\s+(?:async\s+)?fn\s+\w+', content))
        pub_struct_count = len(re.findall(r'\bpub\s+struct\s+\w+', content))
        impl_count = len(re.findall(r'\bimpl\b', content))
        
        return {
            'pub_functions': pub_fn_count,
            'pub_structs': pub_struct_count,
            'impls': impl_count,
        }
    
    def extract_public_functions(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract public function signatures with documentation"""
        samples = []
        
        pattern = r'((?:///[^\n]*\n)+)?\s*(pub(?:\([^)]+\))?\s+(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?fn\s+(\w+)(<[^>]+>)?\s*\(([^)]*)\)(?:\s*->\s*([^{;]+))?)'
        
        for match in re.finditer(pattern, content):
            doc = match.group(1)
            fn_signature = match.group(2)
            fn_name = match.group(3)
            generics = match.group(4) or ""
            params = match.group(5)
            return_type = match.group(6) or "()"
            
            doc_text = ""
            if doc:
                doc_lines = [line.strip()[3:].strip() for line in doc.split('\n') if line.strip().startswith('///')]
                doc_text = ' '.join(doc_lines)
            
            text = ""
            if doc_text:
                text += f"// Documentation: {doc_text}\n\n"
            
            text += f"pub fn {fn_name}{generics}({params})"
            if return_type.strip() != "()":
                text += f" -> {return_type.strip()}"
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "function_signature",
                "function_name": fn_name,
                "tokens": self.count_tokens(text)
            })
        
        return samples
    
    def extract_public_structs(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract public struct definitions with impl counts"""
        samples = []
        
        struct_pattern = r'((?:///[^\n]*\n)+)?\s*(pub(?:\([^)]+\))?\s+struct\s+(\w+)(<[^>]+>)?)'
        
        for match in re.finditer(struct_pattern, content):
            doc = match.group(1)
            struct_name = match.group(3)
            generics = match.group(4) or ""
            
            doc_text = ""
            if doc:
                doc_lines = [line.strip()[3:].strip() for line in doc.split('\n') if line.strip().startswith('///')]
                doc_text = ' '.join(doc_lines)
            
            impl_pattern = fr'impl(?:<[^>]+>)?\s+(?:(\w+(?:::\w+)*)\s+for\s+)?{re.escape(struct_name)}(?:<[^>]+>)?\s*(?:where[^{{]*)?{{'
            impl_matches = list(re.finditer(impl_pattern, content))
            impl_count = len(impl_matches)
            
            traits = []
            for impl_match in impl_matches:
                trait_name = impl_match.group(1)
                if trait_name:
                    traits.append(trait_name)
            
            text = ""
            
            if doc_text:
                text += f"// Documentation: {doc_text}\n\n"
            
            text += f"pub struct {struct_name}{generics}"
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "struct_definition",
                "struct_name": struct_name,
                "impl_count": impl_count,
                "traits": traits,
                "tokens": self.count_tokens(text)
            })
        
        return samples
    
    def extract_public_traits(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract public trait definitions"""
        samples = []
        
        trait_pattern = r'((?:///[^\n]*\n)+)?\s*(pub(?:\([^)]+\))?\s+trait\s+(\w+)(<[^>]+>)?(?:\s*:\s*([^{]+))?)'
        
        for match in re.finditer(trait_pattern, content):
            doc = match.group(1)
            trait_name = match.group(3)
            generics = match.group(4) or ""
            bounds = match.group(5) or ""
            
            doc_text = ""
            if doc:
                doc_lines = [line.strip()[3:].strip() for line in doc.split('\n') if line.strip().startswith('///')]
                doc_text = ' '.join(doc_lines)
            
            text = ""
            if doc_text:
                text += f"// Documentation: {doc_text}\n\n"
            
            text += f"pub trait {trait_name}{generics}"
            if bounds.strip():
                text += f": {bounds.strip()}"
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "trait_definition",
                "trait_name": trait_name,
                "tokens": self.count_tokens(text)
            })
        
        return samples
    
    def extract_impl_blocks(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract impl blocks with their methods"""
        samples = []
        
        impl_pattern = r'impl(?:<[^>]+>)?\s+((?:\w+(?:::\w+)*)\s+for\s+)?(\w+)(?:<[^>]+>)?\s*(?:where[^{]*)?\s*\{'
        
        for match in re.finditer(impl_pattern, content):
            trait_name = match.group(1)
            type_name = match.group(2)
            
            start_pos = match.end()
            brace_count = 1
            end_pos = start_pos
            
            for i in range(start_pos, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            impl_content = content[start_pos:end_pos]
            
            fn_count = len(re.findall(r'\bfn\s+\w+', impl_content))
            pub_fn_count = len(re.findall(r'\bpub\s+fn\s+\w+', impl_content))
            
            if trait_name:
                impl_type = f"impl {trait_name.strip()} for {type_name}"
            else:
                impl_type = f"impl {type_name}"
            
            text = f"{impl_type}"
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "impl_block",
                "type_name": type_name,
                "trait_name": trait_name.strip() if trait_name else None,
                "method_count": fn_count,
                "public_method_count": pub_fn_count,
                "tokens": self.count_tokens(text)
            })
        
        return samples
    
    def extract_module_exports(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract pub mod and pub use declarations"""
        samples = []
        
        mod_pattern = r'^\s*pub\s+mod\s+(\w+)\s*;'
        mods = re.findall(mod_pattern, content, re.MULTILINE)
        
        use_pattern = r'^\s*pub\s+use\s+([^;]+);'
        uses = re.findall(use_pattern, content, re.MULTILINE)
        
        if mods or uses:
            text = ""
            if mods:
                for mod in mods[:15]:  
                    text += f"pub mod {mod};\n"
                if len(mods) > 15:
                    text += f"// ... and {len(mods) - 15} more\n"
                if uses:
                    text += "\n"
            
            if uses:
                for use in uses[:15]:  
                    text += f"pub use {use};\n"
                if len(uses) > 15:
                    text += f"// ... and {len(uses) - 15} more\n"
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "module_structure",
                "submodule_count": len(mods),
                "export_count": len(uses),
                "tokens": self.count_tokens(text)
            })
        
        return samples
    
    def create_file_header(self, rel_path: str, module: str, metadata: Dict) -> str:
        """Create context header for file"""
        header = f"// File: {rel_path}\n"
        header += f"// Module: {module}\n"
        
        if metadata['pub_functions'] > 0:
            header += f"// Public functions: {metadata['pub_functions']}\n"
        if metadata['pub_structs'] > 0:
            header += f"// Public structs: {metadata['pub_structs']}\n"
        
        header += "\n"
        return header
    
    def chunk_content(self, content: str, file_path: str, module: str, language: str = None) -> List[Dict]:
        """Split content into chunks if needed"""
        tokens = self.count_tokens(content)
        
        sample = {
            'text': content,
            'file_path': file_path,
            'module': module,
            'tokens': tokens,
            'type': 'full_file'
        }
        if language:
            sample['language'] = language
            
        if tokens <= self.max_tokens:
            return [sample]
        
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line + '\n')
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunk_sample = {
                    'text': chunk_text,
                    'file_path': f"{file_path}#chunk{chunk_idx}",
                    'module': module,
                    'tokens': current_tokens,
                    'type': 'chunk'
                }
                if language:
                    chunk_sample['language'] = language
                chunks.append(chunk_sample)
                
                overlap_lines = current_chunk[-20:] if len(current_chunk) > 20 else current_chunk
                current_chunk = overlap_lines
                current_tokens = self.count_tokens('\n'.join(overlap_lines))
                chunk_idx += 1
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            final_chunk = {
                'text': chunk_text,
                'file_path': f"{file_path}#chunk{chunk_idx}",
                'module': module,
                'tokens': current_tokens,
                'type': 'chunk'
            }
            if language:
                final_chunk['language'] = language
            chunks.append(final_chunk)
        
        return chunks
    
    def extract_python_definitions(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract Python functions and classes"""
        samples = []
        
        function_pattern = r'^(def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?:\s*(?:"""[^"]*"""|\'\'\'[^\']*\'\'\')?(?:\n(?:    .*|\n))*?)(?=\n(?:def |class |\S|$))'
        functions = re.finditer(function_pattern, content, re.MULTILINE)
        
        for match in functions:
            func_code = match.group(1).strip()
            name_match = re.search(r'def\s+(\w+)\s*\(', func_code)
            if name_match:
                func_name = name_match.group(1)
                
                if func_name.startswith('_') and not func_name.startswith('__'):
                    continue
                
                text = func_code
                
                samples.append({
                    "text": text,
                    "file_path": file_path,
                    "module": module,
                    "type": "python_function",
                    "name": func_name,
                    "tokens": self.count_tokens(text)
                })
        
        class_pattern = r'^(class\s+\w+(?:\([^)]*\))?:\s*(?:"""[^"]*"""|\'\'\'[^\']*\'\'\')?(?:\n(?:    .*|\n))*?)(?=\n(?:def |class |\S|$))'
        classes = re.finditer(class_pattern, content, re.MULTILINE)
        
        for match in classes:
            class_code = match.group(1).strip()
            name_match = re.search(r'class\s+(\w+)', class_code)
            if name_match:
                class_name = name_match.group(1)
                method_count = len(re.findall(r'^\s{4}def\s+\w+', class_code, re.MULTILINE))
                
                text = class_code
                
                samples.append({
                    "text": text,
                    "file_path": file_path,
                    "module": module,
                    "type": "python_class",
                    "name": class_name,
                    "method_count": method_count,
                    "tokens": self.count_tokens(text)
                })
        
        return samples
    
    def extract_js_definitions(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract JavaScript/TypeScript functions and classes"""
        samples = []
        
        func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        functions = re.finditer(func_pattern, content, re.DOTALL)
        
        for match in functions:
            func_name = match.group(1)
            func_code = match.group(0)
            
            text = func_code
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "js_function",
                "name": func_name,
                "tokens": self.count_tokens(text)
            })
        
        arrow_pattern = r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        arrows = re.finditer(arrow_pattern, content, re.DOTALL)
        
        for match in arrows:
            func_name = match.group(1)
            func_code = match.group(0)
            
            text = func_code
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "js_arrow_function",
                "name": func_name,
                "tokens": self.count_tokens(text)
            })
        
        class_pattern = r'(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        classes = re.finditer(class_pattern, content, re.DOTALL)
        
        for match in classes:
            class_name = match.group(1)
            class_code = match.group(0)
            
            method_count = len(re.findall(r'(?:async\s+)?(?:\w+)\s*\([^)]*\)\s*(?::\s*[^{]+)?\s*\{', class_code))
            
            text = f"""// File: {file_path} | Class: {class_name} | Methods: {method_count}

{class_code}"""
            
            samples.append({
                "text": text,
                "file_path": file_path,
                "module": module,
                "type": "js_class",
                "name": class_name,
                "method_count": method_count,
                "tokens": self.count_tokens(text)
            })
        
        return samples
    
    def process_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a file based on its language
        Returns: (file_level_samples, granular_samples)
        """
        language = self.detect_language(file_path)
        
        if language == 'unknown':
            logger.debug(f"Unknown file type: {file_path}")
            return [], []
        
        if language == 'rust':
            return self.process_rust_file(file_path)
        else:
            return self.process_simple_file(file_path, language)
    
    def process_rust_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Process a single Rust file (original logic)"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            rel_path = None
            for repo_path in self.repo_paths:
                try:
                    rel_path = file_path.relative_to(repo_path)
                    break
                except ValueError:
                    continue
            
            if rel_path is None:
                logger.warning(f"Could not determine relative path for {file_path}")
                return [], []
            
            if self.is_excluded_path(file_path, 'rust'):
                return [], []
            
            if self.is_generated_file(content):
                return [], []
            
            is_test = self.is_test_file(file_path, content)
            if is_test and not self.config['dataset']['include_tests']:
                return [], []
            
            content = self.clean_content(content)
            module = self.extract_module_name(rel_path)
            
            metadata = self.extract_quick_metadata(content)
            
            file_samples = self.chunk_content(content, str(rel_path), module, 'rust')
            
            granular_samples = []
            
            if not is_test and (metadata['pub_functions'] > 0 or metadata['pub_structs'] > 0):
                granular_samples.extend(
                    self.extract_public_functions(content, str(rel_path), module)
                )
                
                granular_samples.extend(
                    self.extract_public_structs(content, str(rel_path), module)
                )
                
                granular_samples.extend(
                    self.extract_public_traits(content, str(rel_path), module)
                )
                
                granular_samples.extend(
                    self.extract_impl_blocks(content, str(rel_path), module)
                )
                
                granular_samples.extend(
                    self.extract_module_exports(content, str(rel_path), module)
                )
            
            for sample in granular_samples:
                sample['language'] = 'rust'
            
            return file_samples, granular_samples
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return [], []
    
    def process_simple_file(self, file_path: Path, language: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Simple processing for non-Rust files
        Returns: (file_level_samples, granular_samples)
        
        Granular extraction depends on enable_granular_all flag
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            rel_path = None
            repo_name = None
            for repo_path in self.repo_paths:
                try:
                    rel_path = file_path.relative_to(repo_path)
                    repo_name = repo_path.name
                    break
                except ValueError:
                    continue
            
            if rel_path is None:
                logger.warning(f"Could not determine relative path for {file_path}")
                return [], []
            
            if self.is_excluded_path(file_path, language):
                return [], []
            
            module = str(rel_path.parent) if rel_path.parent != Path('.') else str(rel_path.stem)
            content = self.clean_content(content)
            
            # Use content directly without header - metadata is in JSON fields
            file_samples = self.chunk_content(content, str(rel_path), module, language)
            
            granular_samples = []
            if self.enable_granular_all:
                if language == 'python':
                    granular_samples = self.extract_python_definitions(content, str(rel_path), module)
                    for sample in granular_samples:
                        sample['language'] = language
                elif language in ['javascript', 'typescript']:
                    granular_samples = self.extract_js_definitions(content, str(rel_path), module)
                    for sample in granular_samples:
                        sample['language'] = language
            
            return file_samples, granular_samples
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return [], []
    
    def extract_git_commits(self, repo_path: Path, max_commits: int = None) -> List[Dict]:
        """
        Extract git commit history from repository
        Returns list of commit samples with message, diff, and metadata
        """
        samples = []
        
        try:
            # Get commit history with stats
            cmd = ['git', '-C', str(repo_path), 'log', '--pretty=format:%H|%an|%ae|%ad|%s', '--date=iso']
            if max_commits:
                cmd.extend(['-n', str(max_commits)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commits_raw = result.stdout.strip()
            
            if not commits_raw:
                logger.warning(f"No commits found in {repo_path}")
                return []
            
            commit_lines = commits_raw.split('\n')
            logger.info(f"  Processing {len(commit_lines)} commits from {repo_path.name}...")
            
            for line in tqdm(commit_lines, desc=f"  Extracting commits from {repo_path.name}", leave=False):
                if not line:
                    continue
                
                # Parse commit header: hash|author|email|date|message
                header_parts = line.split('|', 4)
                if len(header_parts) < 5:
                    continue
                
                commit_hash = header_parts[0]
                author = header_parts[1]
                email = header_parts[2]
                date = header_parts[3]
                message = header_parts[4]
                
                # Get commit diff
                diff_cmd = ['git', '-C', str(repo_path), 'show', '--pretty=', '--unified=2', commit_hash]
                diff_result = subprocess.run(diff_cmd, capture_output=True, text=True)
                diff_text = diff_result.stdout
                
                # Filter by file types (only include relevant languages)
                if not self._is_relevant_commit_diff(diff_text):
                    continue
                
                # Clean and limit diff size
                diff_lines = diff_text.split('\n')
                if len(diff_lines) > 200:  # Limit diff size
                    diff_text = '\n'.join(diff_lines[:200]) + f"\n... ({len(diff_lines) - 200} more lines truncated)"
                
                # Create training sample
                text = f"""// Commit: {commit_hash[:8]}
// Author: {author}
// Date: {date}
// Message: {message}

{diff_text}"""
                
                # Count tokens
                tokens = self.count_tokens(text)
                
                # Skip extremely large commits
                if tokens > self.max_tokens * 2:
                    logger.debug(f"Skipping large commit {commit_hash[:8]} ({tokens} tokens)")
                    continue
                
                samples.append({
                    "text": text,
                    "file_path": f"{repo_path.name}/commit/{commit_hash[:8]}",
                    "module": f"{repo_path.name}_history",
                    "type": "git_commit",
                    "language": "git",
                    "commit_hash": commit_hash,
                    "author": author,
                    "date": date,
                    "message": message,
                    "tokens": tokens
                })
            
            logger.info(f"  ✓ Extracted {len(samples)} relevant commits from {repo_path.name}")
            return samples
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed for {repo_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting commits from {repo_path}: {e}")
            return []
    
    def extract_recent_commits(self, repo_path: Path, days: int = 365) -> List[Dict]:
        """
        Extract recent commits from the last N days
        Useful for focusing on recent development patterns
        """
        samples = []
        
        try:
            from datetime import datetime, timedelta
            
            # Calculate date cutoff
            cutoff_date = datetime.now() - timedelta(days=days)
            date_str = cutoff_date.strftime('%Y-%m-%d')
            
            # Get recent commits
            cmd = [
                'git', '-C', str(repo_path), 'log',
                f'--since={date_str}',
                '--pretty=format:%H|%an|%ae|%ad|%s',
                '--date=iso'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commit_lines = result.stdout.strip().split('\n')
            
            if not commit_lines or not commit_lines[0]:
                logger.info(f"  No commits found in last {days} days for {repo_path.name}")
                return []
            
            logger.info(f"  Processing {len(commit_lines)} commits from last {days} days in {repo_path.name}...")
            
            for line in tqdm(commit_lines, desc=f"  Extracting recent commits from {repo_path.name}", leave=False):
                if not line:
                    continue
                
                parts = line.split('|', 4)
                if len(parts) < 5:
                    continue
                
                commit_hash = parts[0]
                author = parts[1]
                email = parts[2]
                date = parts[3]
                message = parts[4]
                
                # Get diff
                diff_cmd = ['git', '-C', str(repo_path), 'show', '--pretty=', '--unified=2', commit_hash]
                diff_result = subprocess.run(diff_cmd, capture_output=True, text=True)
                diff_text = diff_result.stdout
                
                # Filter by file types
                if not self._is_relevant_commit_diff(diff_text):
                    continue
                
                # Limit diff size
                diff_lines = diff_text.split('\n')
                if len(diff_lines) > 150:
                    diff_text = '\n'.join(diff_lines[:150]) + f"\n... (truncated)"
                
                text = f"""// Commit: {commit_hash[:8]}
// Author: {author}
// Date: {date}
// Message: {message}

{diff_text}"""
                
                tokens = self.count_tokens(text)
                
                if tokens > self.max_tokens * 2:
                    continue
                
                samples.append({
                    "text": text,
                    "file_path": f"{repo_path.name}/commit/{commit_hash[:8]}",
                    "module": f"{repo_path.name}_history",
                    "type": "git_commit",
                    "language": "git",
                    "commit_hash": commit_hash,
                    "author": author,
                    "date": date,
                    "message": message,
                    "tokens": tokens
                })
            
            logger.info(f"  ✓ Extracted {len(samples)} relevant commits from last {days} days")
            return samples
            
        except Exception as e:
            logger.error(f"Error extracting recent commits: {e}")
            return []
    
    def _is_relevant_commit_diff(self, diff_text: str) -> bool:
        """Check if commit diff contains files in configured languages"""
        for language in self.languages:
            if language == 'rust' and '.rs' in diff_text:
                return True
            elif language == 'python' and '.py' in diff_text:
                return True
            elif language == 'javascript' and ('.js' in diff_text or '.jsx' in diff_text):
                return True
            elif language == 'typescript' and ('.ts' in diff_text or '.tsx' in diff_text):
                return True
            elif language == 'html' and '.html' in diff_text:
                return True
            elif language == 'css' and '.css' in diff_text:
                return True
        return False
    
    def collect_code_files(self) -> List[Path]:
        """Collect all code files based on configured languages"""
        all_files = []
        
        for repo_path in self.repo_paths:
            logger.info(f"Collecting files from {repo_path}")
            
            for language in self.languages:
                if language == 'rust':
                    files = list(repo_path.rglob("*.rs"))
                elif language == 'python':
                    files = list(repo_path.rglob("*.py"))
                elif language == 'javascript':
                    files = list(repo_path.rglob("*.js"))
                    files.extend(list(repo_path.rglob("*.jsx")))
                elif language == 'typescript':
                    files = list(repo_path.rglob("*.ts"))
                    files.extend(list(repo_path.rglob("*.tsx")))
                elif language == 'html':
                    files = list(repo_path.rglob("*.html"))
                    files.extend(list(repo_path.rglob("*.htm")))
                elif language == 'css':
                    files = list(repo_path.rglob("*.css"))
                    files.extend(list(repo_path.rglob("*.scss")))
                    files.extend(list(repo_path.rglob("*.sass")))
                else:
                    logger.warning(f"Unknown language: {language}")
                    continue
                
                all_files.extend(files)
                logger.info(f"  ✓ {language}: {len(files)} files")
        
        logger.info(f"✓ Total: {len(all_files)} files across all repos and languages")
        return all_files
    
    def process_all_files(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all code files across all repositories and languages
        Returns: (file_level_samples, granular_samples)
        """
        code_files = self.collect_code_files()
        all_file_samples = []
        all_granular_samples = []
        
        logger.info("Processing files for file-level AND granular data...")
        for file_path in tqdm(code_files, desc="Processing", unit="files"):
            file_samples, granular_samples = self.process_file(file_path)
            all_file_samples.extend(file_samples)
            all_granular_samples.extend(granular_samples)
        
        logger.info(f"✓ Created {len(all_file_samples)} file-level samples")
        logger.info(f"✓ Created {len(all_granular_samples)} granular samples")
        logger.info(f"✓ Total: {len(all_file_samples) + len(all_granular_samples)} samples")
        
        # Extract git commit history if configured
        if self.config['dataset'].get('include_commits', False):
            logger.info("\n=== Extracting Git Commit History ===")
            commit_days = self.config['dataset'].get('commit_history_days', 365)
            max_commits = self.config['dataset'].get('max_commits_per_repo', 1000)
            
            for repo_path in self.repo_paths:
                if (repo_path / '.git').exists():
                    if commit_days:
                        # Extract recent commits
                        commit_samples = self.extract_recent_commits(repo_path, days=commit_days)
                    else:
                        # Extract all commits up to max
                        commit_samples = self.extract_git_commits(repo_path, max_commits=max_commits)
                    
                    all_file_samples.extend(commit_samples)
                else:
                    logger.warning(f"No .git directory found in {repo_path}")
            
            commit_count = sum(1 for s in all_file_samples if s.get('type') == 'git_commit')
            logger.info(f"✓ Extracted {commit_count} commit samples")
            logger.info(f"✓ New total: {len(all_file_samples) + len(all_granular_samples)} samples")
        
        return all_file_samples, all_granular_samples
    
    def split_dataset(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split into train and validation sets"""
        random.seed(self.config['dataset']['random_seed'])
        random.shuffle(samples)
        
        train_ratio = self.config['dataset']['train_split']
        split_idx = int(len(samples) * train_ratio)
        
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        logger.info(f"✓ Train samples: {len(train_samples)}")
        logger.info(f"✓ Validation samples: {len(val_samples)}")
        
        return train_samples, val_samples
    
    def save_dataset(self, file_samples: List[Dict], granular_samples: List[Dict]):
        """Save combined dataset to a single all_data.jsonl file"""
        
        all_samples = file_samples + granular_samples
        random.shuffle(all_samples)
        
        output_path = self.output_dir / "all_data.jsonl"
        logger.info(f"Saving combined dataset to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        language_stats = {}
        for sample in all_samples:
            lang = sample['language']
            language_stats[lang] = language_stats.get(lang, 0) + 1
        
        stats = {
            'total_samples': len(all_samples),
            'file_level_samples': len(file_samples),
            'granular_samples': len(granular_samples),
            'total_tokens': sum(s.get('tokens', 0) for s in all_samples),
            'sample_types': {},
            'modules': list(set(s.get('module', 'unknown') for s in all_samples)),
            'languages': language_stats,
            'repositories': len(self.repo_paths),
        }
        
        for sample in all_samples:
            stype = sample.get('type', 'unknown')
            stats['sample_types'][stype] = stats['sample_types'].get(stype, 0) + 1
        
        stats['granular_breakdown'] = {
            'functions': stats['sample_types'].get('function_signature', 0),
            'structs': stats['sample_types'].get('struct_definition', 0),
            'traits': stats['sample_types'].get('trait_definition', 0),
            'impls': stats['sample_types'].get('impl_block', 0),
            'modules': stats['sample_types'].get('module_structure', 0),
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
        logger.info("📊 Dataset Summary:")
        logger.info(f"{'='*70}")
        logger.info(f"  Repositories processed: {stats.get('repositories', 1)}")
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"    └─ File-level: {stats['file_level_samples']:,}")
        logger.info(f"    └─ Granular: {stats['granular_samples']:,}")
        
        # Show commit samples if present
        commit_count = stats['sample_types'].get('git_commit', 0)
        if commit_count > 0:
            logger.info(f"    └─ Git commits: {commit_count:,}")
        
        logger.info(f"  Total tokens: {stats['total_tokens']:,}")
        
        if 'languages' in stats:
            logger.info(f"\n   Samples by language:")
            for lang, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    └─ {lang}: {count:,}")
        
        logger.info(f"\n  📂 File-level samples:")
        for stype in ['full_file', 'chunk']:
            count = stats['sample_types'].get(stype, 0)
            if count > 0:
                logger.info(f"    └─ {stype}: {count:,}")
        
        if self.enable_granular_all:
            logger.info(f"\n  Granular samples (all languages):")
        else:
            logger.info(f"\n  Granular samples (Rust only):")
        for name, count in stats['granular_breakdown'].items():
            if count > 0:
                logger.info(f"    └─ {name}: {count:,}")
        
        logger.info(f"\n  📦 Modules covered: {len(stats['modules'])}")
        logger.info(f"{'='*70}")
    
    def prepare(self):
        """Main preparation pipeline - generates both file-level and granular data"""
        logger.info("=" * 70)
        logger.info("🚀 Enhanced Dataset Preparation for CPT Training")
        logger.info("   Multi-Repo & Multi-Language Support")
        if self.enable_granular_all:
            logger.info("   Generating: File-level chunks + Granular data (ALL languages)")
        else:
            logger.info("   Generating: File-level chunks + Granular data (Rust only)")
        logger.info("=" * 70)
        
        self.load_tokenizer()
        file_samples, granular_samples = self.process_all_files()
        
        if not file_samples and not granular_samples:
            logger.error("No samples generated! Check your configuration.")
            return
        
        output_path = self.save_dataset(file_samples, granular_samples)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Dataset preparation complete!")
        logger.info(f"   📄 Output: {output_path}")
        logger.info(f"   📊 Total: {len(file_samples) + len(granular_samples):,} samples")
        logger.info("=" * 70)
        logger.info("\n🎯 Next step: Run training with ./train_direct.sh")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate dataset from GitHub repositories for CPT training'
    )
    
    parser.add_argument(
        '--repos',
        nargs='+',
        help='GitHub repository URL(s) to clone and process (overrides config)',
        metavar='URL'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config.yaml (default: config.yaml)',
        metavar='PATH'
    )
    
    parser.add_argument(
        '--granular',
        action='store_true',
        help='Enable granular extraction (functions/classes) for all languages (default: Rust only)'
    )
    
    args = parser.parse_args()
    
    # Initialize preparator with CLI arguments
    preparator = EnhancedDatasetPreparator(
        config_path=args.config,
        repo_urls=args.repos,
        enable_granular_all=args.granular
    )
    preparator.prepare()


if __name__ == "__main__":
    main()

