import argparse

import torch
import torch.nn as nn

from utils.config import load_config
from utils.data import load_data
from trainer import Trainer

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, default="config/config.yaml")

args = parser.parse_args()

# Load config
config = load_config(args.config)

# Load data
num_users, num_items, train_adj, train_unique_users, test_adj, test_unique_users = load_data(config["data_path"])

# 여기부터 하믄 됨
# trainer = Trainer(config, num_users, num_items, train_adj, test_adj) 