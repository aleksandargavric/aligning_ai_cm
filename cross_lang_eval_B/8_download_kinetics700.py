from datasets import load_dataset
import numpy as np

local_dir = "./datasets/webvid"


# Load the dataset
dataset = load_dataset("iejMac/CLIP-WebVid", split='train', cache_dir=local_dir)
