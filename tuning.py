import argparse

import torch
import torch.nn as nn

from utils.config import load_config
from utils.data import load_data
from trainer import Trainer
from itertools import product

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/config.yaml")

args = parser.parse_args()

# Load config
config = load_config(args.config)

learning_rate = [0.01, 0.005, 0.001, 0.0005]
dropout_rate = [0.0]#[0.1, 0.2, 0.3]
model_size = [64, 128, 256, 512]

# Load data
num_users, num_items, train_adj, train_unique_users, valid_data, test_data = load_data(config["data_path"])

for lr, dr, ms in product(learning_rate, dropout_rate, model_size):
    trainer = Trainer(config, num_users, num_items, train_adj=train_adj, valid_adj=valid_data, test_adj=test_data)
    trainer.train()

    trainer.save_model()
    trainer.test()