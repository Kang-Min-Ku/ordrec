import yaml

def load_config(file, in_channel_dim, out_channel_dim):
    with open(file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["in_channel"] = in_channel_dim
    config["out_channel"] = out_channel_dim
    return config