import pandas as pd
import numpy as np


for id in range(1, 12):
    train = pd.read_csv(f"./data{id}_train.csv")
    test = pd.read_csv(f"./data{id}_test.csv")

    for i in range(len(train)):
        rand_val = np.random.uniform(-0.5, 0.5)
        train.iloc[i,0] += rand_val
        train.iloc[i, 1] += rand_val

    for i in range(len(test)):
        rand_val = np.random.uniform(-0.5, 0.5)
        test.iloc[i,0] += rand_val
        test.iloc[i, 1] += rand_val

    train.to_csv(f"data{id+1}_train.csv", index=False)
    test.to_csv(f"data{id+1}_test.csv", index=False)






