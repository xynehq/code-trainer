"""
GitHub API extractor for PRs, reviews, and comments.
"""
import requests
from typing import List, Dict, Generator, Optional
from tqdm import tqdm
import time


class GitHubExtractor:
    """Extracts PR data from GitHub API."""
    
    def __init__(self, token: str, repo_owner: str, repo_name: str):
        """
        Initialize GitHub API extractor.
        
        Args:
            token: GitHub API token
            repo_owner: Repository owner (e.g., "juspay")
            repo_name: Repository name (e.g., "hyperswitch")
        """
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def extract_prs(
        self,
        include_merged: bool = True,
        include_closed: bool = True,
        include_reviews: bool = True,
        include_comments: bool = True,
        max_prs: Optional[int] = None,
        show_progress: bool = True
    ) -> Generator[Dict, None, None]:
        """
        Extract PRs with reviews and comments.
        
        Args:
            include_merged: Include merged PRs
            include_closed: Include closed (non-merged) PRs
            include_reviews: Include review comments
            include_comments: Include PR comments
            max_prs: Maximum PRs to fetch
            show_progress: Show progress bar
            
        Yields:
            Dict with PR data formatted for training
        """
        # Fetch all PRs
        prs = self._fetch_all_prs(include_merged, include_closed, max_prs, show_progress)
        
        iterator = tqdm(prs, desc="Processing PRs") if show_progress else prs
        
        for pr in iterator:
            try:
                pr_data = self._get_pr_details(
                    pr, 
                    include_reviews, 
                    include_comments
                )
                
                # Format training content
                training_content = self._format_pr_training_content(pr_data)
                
                yield {
                    "type": "pr_diff",
                    "pr_number": pr['number'],
                    "title": pr['title'],
                    "state": pr['state'],
                    "merged": pr.get('merged', False),
                    "training_content": training_content
                }
                
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error processing PR #{pr['number']}: {e}")
                continue
    
    def _fetch_all_prs(
        self, 
        include_merged: bool, 
        include_closed: bool, 
        max_prs: Optional[int],
        show_progress: bool
    ) -> List[Dict]:
        """Fetch all PRs matching criteria."""
        prs = []
        page = 1
        per_page = 100
        
        while True:
            # Fetch both open and closed PRs
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls"
            params = {
                'state': 'all',  # Get both open and closed
                'per_page': per_page,
                'page': page,
                'sort': 'created',
                'direction': 'desc'
            }
            
            response = self._make_request(url, params)
            if not response:
                break
            
            page_prs = response
            if not page_prs:
                break
            
            # Filter based on criteria
            for pr in page_prs:
                is_merged = pr.get('merged_at') is not None
                is_closed = pr['state'] == 'closed'
                
                should_include = False
                if is_merged and include_merged:
                    should_include = True
                elif is_closed and not is_merged and include_closed:
                    should_include = True
                elif pr['state'] == 'open':
                    should_include = True
                
                if should_include:
                    prs.append(pr)
            
            if show_progress:
                tqdm.write(f"Fetched {len(prs)} PRs so far...")
            
            # Check if we've hit max or end of results
            if max_prs and len(prs) >= max_prs:
                prs = prs[:max_prs]
                break
            
            if len(page_prs) < per_page:
                break
            
            page += 1
            time.sleep(0.5)  # Rate limiting
        
        return prs
    
    def _get_pr_details(
        self, 
        pr: Dict, 
        include_reviews: bool, 
        include_comments: bool
    ) -> Dict:
        """Get detailed PR information including diff, reviews, comments."""
        pr_number = pr['number']
        
        # Get PR diff
        diff_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}"
        headers = {**self.headers, "Accept": "application/vnd.github.v3.diff"}
        response = requests.get(diff_url, headers=headers)
        diff = response.text if response.status_code == 200 else ""
        
        # Get reviews
        reviews = []
        if include_reviews:
            reviews_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}/reviews"
            reviews_data = self._make_request(reviews_url)
            if reviews_data:
                reviews = reviews_data
        
        # Get comments (both issue comments and review comments)
        comments = []
        if include_comments:
            # Issue comments
            comments_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{pr_number}/comments"
            comments_data = self._make_request(comments_url)
            if comments_data:
                comments.extend(comments_data)
            
            # Review comments (inline code comments)
            review_comments_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}/comments"
            review_comments_data = self._make_request(review_comments_url)
            if review_comments_data:
                comments.extend(review_comments_data)
        
        return {
            'pr': pr,
            'diff': diff,
            'reviews': reviews,
            'comments': comments
        }
    
    def _format_pr_training_content(self, pr_data: Dict) -> str:
        """Format PR data for training."""
        pr = pr_data['pr']
        
        content = f"Pull Request #{pr['number']}: {pr['title']}\n\n"
        
        # PR description
        if pr.get('body'):
            content += f"Description:\n{pr['body']}\n\n"
        
        # Diff
        content += "Diff:\n"
        content += pr_data['diff']
        content += "\n\n"
        
        # Reviews
        if pr_data['reviews']:
            content += "Reviews:\n"
            for review in pr_data['reviews']:
                user = review.get('user', {}).get('login', 'Unknown')
                state = review.get('state', 'COMMENTED')
                body = review.get('body', '')
                content += f"- {user} ({state}): {body}\n"
            content += "\n"
        
        # Comments
        if pr_data['comments']:
            content += "Comments:\n"
            for comment in pr_data['comments']:
                user = comment.get('user', {}).get('login', 'Unknown')
                body = comment.get('body', '')
                # Include file context if it's a review comment
                if 'path' in comment:
                    path = comment['path']
                    content += f"- {user} on {path}: {body}\n"
                else:
                    content += f"- {user}: {body}\n"
        
        return content
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[List[Dict]]:
        """Make API request with error handling."""
        try:
            response = self.session.get(url, params=params)
            
            # Handle rate limiting
            if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                if int(response.headers['X-RateLimit-Remaining']) == 0:
                    reset_time = int(response.headers['X-RateLimit-Reset'])
                    wait_time = reset_time - time.time() + 10
                    print(f"Rate limited. Waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time)
                    response = self.session.get(url, params=params)
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"API request failed: {e}")
            return None
