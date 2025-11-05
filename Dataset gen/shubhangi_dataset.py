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

# ...existing code from shubhangi_dataset.py...
