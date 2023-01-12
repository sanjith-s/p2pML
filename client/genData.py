import pandas as pd
import numpy as np


count = 20000

arr = []

for i in range(count):
    arr.append([i, i*i])
    arr.append([-i, i*i])

df = pd.DataFrame(arr, columns=['x', 'y'])

df = df.sample(frac=1)
print(df.to_csv('data.csv', index=False))
