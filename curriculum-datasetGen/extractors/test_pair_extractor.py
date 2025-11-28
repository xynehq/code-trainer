"""
Test-code pairing extractor.
Identifies test files and pairs them with implementation files.
"""
from pathlib import Path
from typing import List, Dict, Generator, Optional
import re


class TestPairExtractor:
    """Pairs test files with their implementation counterparts."""
    
    def __init__(self, repo_path: str):
        """
        Initialize test pair extractor.
        
        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
    
    def extract_test_pairs(self, show_progress: bool = True) -> Generator[Dict, None, None]:
        """
        Find test files and pair with implementation files.
        
        Yields:
            Dict with paired test and implementation content
        """
        # Find all test files
        test_files = list(self._find_test_files())
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(test_files, desc="Pairing tests")
        else:
            iterator = test_files
        
        for test_file in iterator:
            try:
                # Find corresponding implementation file
                impl_file = self._find_implementation_file(test_file)
                
                if impl_file and impl_file.exists():
                    # Read both files
                    test_content = test_file.read_text(encoding='utf-8')
                    impl_content = impl_file.read_text(encoding='utf-8')
                    
                    # Create training content
                    training_content = self._format_test_pair_training_content(
                        test_file, impl_file, test_content, impl_content
                    )
                    
                    yield {
                        "type": "test_pair",
                        "test_file": str(test_file.relative_to(self.repo_path)),
                        "impl_file": str(impl_file.relative_to(self.repo_path)),
                        "training_content": training_content
                    }
            
            except Exception as e:
                if show_progress:
                    from tqdm import tqdm
                    tqdm.write(f"Error pairing {test_file}: {e}")
                continue
    
    def _find_test_files(self) -> Generator[Path, None, None]:
        """Find all test files in repository."""
        # Pattern 1: Files in tests/ directories
        for test_dir in self.repo_path.rglob('tests'):
            if test_dir.is_dir():
                for test_file in test_dir.rglob('*.rs'):
                    yield test_file
        
        # Pattern 2: Files with _test.rs or test_.rs suffix
        for test_file in self.repo_path.rglob('*_test.rs'):
            yield test_file
        for test_file in self.repo_path.rglob('test_*.rs'):
            yield test_file
        
        # Pattern 3: Files containing #[cfg(test)] modules
        for rs_file in self.repo_path.rglob('*.rs'):
            if self._contains_test_module(rs_file):
                yield rs_file
    
    def _contains_test_module(self, file_path: Path) -> bool:
        """Check if file contains #[cfg(test)] module."""
        try:
            content = file_path.read_text(encoding='utf-8')
            return '#[cfg(test)]' in content or '#[test]' in content
        except:
            return False
    
    def _find_implementation_file(self, test_file: Path) -> Optional[Path]:
        """
        Find the implementation file corresponding to a test file.
        
        Common patterns:
        - tests/payments.rs -> src/payments.rs
        - src/payments_test.rs -> src/payments.rs
        - crates/router/tests/payments.rs -> crates/router/src/payments.rs
        """
        rel_path = test_file.relative_to(self.repo_path)
        parts = list(rel_path.parts)
        
        # Pattern 1: tests/ directory -> src/ directory
        if 'tests' in parts:
            # Replace 'tests' with 'src'
            src_parts = [p if p != 'tests' else 'src' for p in parts]
            impl_path = self.repo_path / Path(*src_parts)
            if impl_path.exists():
                return impl_path
        
        # Pattern 2: *_test.rs -> *.rs
        if test_file.name.endswith('_test.rs'):
            impl_name = test_file.name.replace('_test.rs', '.rs')
            impl_path = test_file.parent / impl_name
            if impl_path.exists():
                return impl_path
        
        # Pattern 3: test_*.rs -> *.rs
        if test_file.name.startswith('test_'):
            impl_name = test_file.name.replace('test_', '', 1)
            impl_path = test_file.parent / impl_name
            if impl_path.exists():
                return impl_path
        
        # Pattern 4: If file contains #[cfg(test)], the impl is in the same file
        # (we can return the same file, but extract non-test content)
        if self._contains_test_module(test_file):
            return test_file
        
        return None
    
    def _format_test_pair_training_content(
        self, 
        test_file: Path, 
        impl_file: Path, 
        test_content: str, 
        impl_content: str
    ) -> str:
        """Format test pair for training."""
        test_rel = test_file.relative_to(self.repo_path)
        impl_rel = impl_file.relative_to(self.repo_path)
        
        content = f"Test-Implementation Pair\n\n"
        content += f"Implementation: {impl_rel}\n"
        content += "=" * 80 + "\n"
        content += impl_content
        content += "\n\n"
        content += f"Tests: {test_rel}\n"
        content += "=" * 80 + "\n"
        content += test_content
        
        return content
