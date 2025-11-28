"""
Tokenizer-based chunking utility for large files.
"""
from typing import List, Dict
from transformers import AutoTokenizer


class TokenizerChunker:
    """Handles chunking of large text files using model tokenizer."""
    
    def __init__(self, model_name: str, max_chunk_size: int, chunk_overlap: int):
        """
        Initialize the chunker with model tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            max_chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into overlapping segments based on token count.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to include in each chunk
            
        Returns:
            List of chunks with metadata
        """
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # If text fits in one chunk, return as-is
        if len(tokens) <= self.max_chunk_size:
            return [{"training_content": text, **(metadata or {})}]
        
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.max_chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Create chunk with metadata
            chunk_metadata = {**(metadata or {})}
            if len(tokens) > self.max_chunk_size:
                chunk_metadata["chunk_index"] = chunk_num
                chunk_metadata["total_chunks"] = (len(tokens) - 1) // (self.max_chunk_size - self.chunk_overlap) + 1
            
            chunks.append({
                "training_content": chunk_text,
                **chunk_metadata
            })
            
            # Move to next chunk with overlap
            start_idx += self.max_chunk_size - self.chunk_overlap
            chunk_num += 1
        
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
