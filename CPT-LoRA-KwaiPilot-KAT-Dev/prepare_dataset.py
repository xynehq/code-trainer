import os
import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRustDatasetPreparator:
    """
    Enhanced dataset preparator for CPT training
    Generates both file-level chunks AND granular function/struct/trait data
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
    
    def is_excluded_path(self, path: Path) -> bool:
        """Check if path should be excluded"""
        path_str = str(path)
        exclude_patterns = ['/target/', '/generated/', '.gen.rs', '/build/']
        return any(pattern in path_str for pattern in exclude_patterns)
    
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
            
            text = f"""// Function: {fn_name}
// File: {file_path}
// Module: {module}

"""
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
            
            text = f"""// Struct: {struct_name}
// File: {file_path}
// Module: {module}
// Implementations: {impl_count}
"""
            if traits:
                text += f"// Traits: {', '.join(traits)}\n"
            text += "\n"
            
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
            
            text = f"""// Trait: {trait_name}
// File: {file_path}
// Module: {module}

"""
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
            
            text = f"""// Implementation: {impl_type}
// File: {file_path}
// Module: {module}
// Methods: {fn_count} total ({pub_fn_count} public)

{impl_type}"""
            
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
            text = f"""// Module Structure
// File: {file_path}
// Module: {module}

"""
            if mods:
                text += "// Public submodules:\n"
                for mod in mods[:15]:  
                    text += f"pub mod {mod};\n"
                if len(mods) > 15:
                    text += f"// ... and {len(mods) - 15} more\n"
                text += "\n"
            
            if uses:
                text += "// Public exports:\n"
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
                    'type': 'chunk'
                })
                
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
                'type': 'chunk'
            })
        
        return chunks
    
    def process_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a single Rust file
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
            
            metadata = self.extract_quick_metadata(content)
            
            header = self.create_file_header(str(rel_path), module, metadata)
            full_content = header + content
            
            file_samples = self.chunk_content(full_content, str(rel_path), module)
            
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
            
            return file_samples, granular_samples
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return [], []
    
    def collect_rust_files(self) -> List[Path]:
        """Collect all Rust files from repository"""
        logger.info(f"Collecting Rust files from {self.repo_path}")
        rust_files = list(self.repo_path.rglob("*.rs"))
        logger.info(f"✓ Found {len(rust_files)} Rust files")
        return rust_files
    
    def process_all_files(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all Rust files
        Returns: (file_level_samples, granular_samples)
        """
        rust_files = self.collect_rust_files()
        all_file_samples = []
        all_granular_samples = []
        
        logger.info("Processing files for file-level AND granular data...")
        for file_path in tqdm(rust_files, desc="Processing", unit="files"):
            file_samples, granular_samples = self.process_file(file_path)
            all_file_samples.extend(file_samples)
            all_granular_samples.extend(granular_samples)
        
        logger.info(f"✓ Created {len(all_file_samples)} file-level samples")
        logger.info(f"✓ Created {len(all_granular_samples)} granular samples")
        logger.info(f"✓ Total: {len(all_file_samples) + len(all_granular_samples)} samples")
        
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
        
        stats = {
            'total_samples': len(all_samples),
            'file_level_samples': len(file_samples),
            'granular_samples': len(granular_samples),
            'total_tokens': sum(s.get('tokens', 0) for s in all_samples),
            'sample_types': {},
            'modules': list(set(s.get('module', 'unknown') for s in all_samples)),
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
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"    └─ File-level: {stats['file_level_samples']:,}")
        logger.info(f"    └─ Granular: {stats['granular_samples']:,}")
        logger.info(f"  Total tokens: {stats['total_tokens']:,}")
        logger.info(f"\n  📂 File-level samples:")
        for stype in ['full_file', 'chunk']:
            count = stats['sample_types'].get(stype, 0)
            if count > 0:
                logger.info(f"    └─ {stype}: {count:,}")
        
        logger.info(f"\n  🔬 Granular samples:")
        for name, count in stats['granular_breakdown'].items():
            if count > 0:
                logger.info(f"    └─ {name}: {count:,}")
        
        logger.info(f"\n  📦 Modules covered: {len(stats['modules'])}")
        logger.info(f"{'='*70}")
    
    def prepare(self):
        """Main preparation pipeline - generates both file-level and granular data"""
        logger.info("=" * 70)
        logger.info("🚀 Enhanced Dataset Preparation for Hyperswitch CPT")
        logger.info("   Generating: File-level chunks + Granular function/struct/trait data")
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
    preparator = EnhancedRustDatasetPreparator("config.yaml")
    preparator.prepare()


if __name__ == "__main__":
    main()
