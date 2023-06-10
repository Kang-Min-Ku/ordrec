import random

data_sets = ["gowalla", "yelp2018", "amazon-book"]
basedir = "dataset"
for dataset in data_sets:
    with open(f"{basedir}/{dataset}/train.txt", 'r') as f:
        lines = f.readlines()
    with open(f"{basedir}/{dataset}/valid.txt", 'w') as validf:
            with open(f"{basedir}/{dataset}/splitTrain.txt", 'w') as splitedf:
                for l in lines:
                    tmp = l.split()
                    x = tmp[1:]
                    random.shuffle(x)
                    validf.write(f"{tmp[0]} {' '.join(x[:max(1, int(len(x) * 0.1))])}\n")
                    splitedf.write(f"{tmp[0]} {' '.join(x[max(1, int(len(x) * 0.1)):])}\n")
        