import argparse

import torch
import torch.nn as nn

from utils.config import load_config
from utils.data import load_data, load_data_matrix
from trainer import Trainer

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/config.yaml")

args = parser.parse_args()

# Load config
config = load_config(args.config)

load_function = load_data_matrix
if config["use_dict"]:
    load_function = load_data

# Load data
num_users, num_items, train_adj, train_unique_users, valid_adj, valid_unique_users, test_adj, test_unique_users = load_function(config["data_path"])
num_users, num_items, train_adj, train_unique_users, valid_dict, valid_unique_users, test_dict, test_unique_users = load_data(config["data_path"])

# 여기부터 하믄 됨
trainer = Trainer(config, num_users, num_items, train_adj=train_adj, valid_adj=valid_adj, test_adj=test_adj, valid_dict=valid_dict, test_dict=test_dict)
trainer.train()