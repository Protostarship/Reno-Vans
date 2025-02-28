import os
import json
import logging
import argparse
import time
import random
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import psutil

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria
)
from safetensors.torch import save_file

# For TF-IDF and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
