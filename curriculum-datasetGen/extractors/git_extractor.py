"""
Git history extractor.
Extracts commits and diffs from local git repository.
"""
import subprocess
from typing import List, Dict, Generator, Optional
from datetime import datetime
from tqdm import tqdm
import re


class GitExtractor:
    """Extracts commit history and diffs from git repository."""
    
    def __init__(self, repo_path: str):
        """
        Initialize git extractor.
        
        Args:
            repo_path: Path to git repository
        """
        self.repo_path = repo_path
    
    def extract_commits(
        self, 
        include_all: bool = True, 
        max_commits: Optional[int] = None,
        show_progress: bool = True
    ) -> Generator[Dict, None, None]:
        """
        Extract all commits with diffs.
        
        Args:
            include_all: Include all commits (not just merges)
            max_commits: Maximum number of commits to process
            show_progress: Show progress bar
            
        Yields:
            Dict with commit metadata and diff
        """
        # Get commit hashes
        cmd = ['git', '-C', self.repo_path, 'log', '--all', '--format=%H']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commit_hashes = result.stdout.strip().split('\n')
        
        if max_commits:
            commit_hashes = commit_hashes[:max_commits]
        
        iterator = tqdm(commit_hashes, desc="Extracting commits") if show_progress else commit_hashes
        
        for commit_hash in iterator:
            try:
                commit_data = self._get_commit_data(commit_hash)
                
                # Skip if no diff content
                if not commit_data['diff'].strip():
                    continue
                
                # Format training content
                training_content = self._format_commit_training_content(commit_data)
                
                yield {
                    "type": "commit",
                    "commit_hash": commit_hash,
                    "author": commit_data['author'],
                    "date": commit_data['date'],
                    "message": commit_data['message'],
                    "training_content": training_content
                }
                
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error processing commit {commit_hash}: {e}")
                continue
    
    def _get_commit_data(self, commit_hash: str) -> Dict:
        """Get detailed commit information."""
        # Get commit metadata
        cmd = [
            'git', '-C', self.repo_path, 'show', 
            '--format=%an%n%aI%n%B', '--no-patch', commit_hash
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        author = lines[0] if lines else "Unknown"
        date = lines[1] if len(lines) > 1 else ""
        message = '\n'.join(lines[2:]) if len(lines) > 2 else ""
        
        # Get diff
        cmd = ['git', '-C', self.repo_path, 'show', '--format=', commit_hash]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        diff = result.stdout
        
        return {
            'author': author,
            'date': date,
            'message': message.strip(),
            'diff': diff
        }
    
    def _format_commit_training_content(self, commit_data: Dict) -> str:
        """Format commit data for training."""
        content = f"Commit: {commit_data['message']}\n"
        content += f"Author: {commit_data['author']}\n"
        content += f"Date: {commit_data['date']}\n\n"
        content += "Diff:\n"
        content += commit_data['diff']
        return content
    
    def extract_pr_numbers_from_commits(self) -> List[int]:
        """
        Extract PR numbers from commit messages.
        Useful for cross-referencing with GitHub API.
        
        Returns:
            List of PR numbers found in commit messages
        """
        cmd = ['git', '-C', self.repo_path, 'log', '--all', '--format=%s']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        pr_numbers = set()
        # Common patterns: "Merge pull request #123", "(#123)", "PR #123"
        patterns = [
            r'#(\d+)',
            r'pull request #(\d+)',
            r'PR #(\d+)',
        ]
        
        for line in result.stdout.split('\n'):
            for pattern in patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                pr_numbers.update(int(m) for m in matches)
        
        return sorted(pr_numbers)
