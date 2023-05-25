import yaml

def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config