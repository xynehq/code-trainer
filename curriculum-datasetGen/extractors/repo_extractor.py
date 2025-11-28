"""
Repository file extractor.
Walks the repository and extracts all matching files.
"""
import os
from pathlib import Path
from typing import List, Dict, Generator
import pathspec
from tqdm import tqdm


class RepoExtractor:
    """Extracts files from repository based on patterns."""
    
    def __init__(self, repo_path: str, include_patterns: List[str], exclude_patterns: List[str]):
        """
        Initialize repo extractor.
        
        Args:
            repo_path: Path to repository root
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
        """
        self.repo_path = Path(repo_path)
        self.include_spec = pathspec.PathSpec.from_lines('gitwildmatch', include_patterns)
        self.exclude_spec = pathspec.PathSpec.from_lines('gitwildmatch', exclude_patterns)
    
    def extract_files(self, show_progress: bool = True) -> Generator[Dict, None, None]:
        """
        Extract all matching files from repository.
        
        Yields:
            Dict with file metadata and content
        """
        # Collect all files first for progress bar
        all_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.repo_path)
                
                if self._should_include(rel_path):
                    all_files.append(file_path)
        
        # Process files with progress bar
        iterator = tqdm(all_files, desc="Extracting files") if show_progress else all_files
        
        for file_path in iterator:
            try:
                rel_path = file_path.relative_to(self.repo_path)
                
                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Skip binary files
                    continue
                
                # Format training content with file header
                training_content = f"// File: {rel_path}\n\n{content}"
                
                yield {
                    "type": "file",
                    "path": str(rel_path),
                    "size_bytes": file_path.stat().st_size,
                    "training_content": training_content
                }
                
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error processing {file_path}: {e}")
                continue
    
    def _should_include(self, rel_path: Path) -> bool:
        """Check if file should be included."""
        path_str = str(rel_path)
        return (self.include_spec.match_file(path_str) and 
                not self.exclude_spec.match_file(path_str))
    
    def _should_exclude(self, path: str) -> bool:
        """Check if directory should be excluded."""
        rel_path = Path(path).relative_to(self.repo_path)
        return self.exclude_spec.match_file(str(rel_path))
