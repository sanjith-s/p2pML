import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Rational, Integer, Array, lambdify
from sympy.tensor.array import derive_by_array
from sympy.vector import CoordSys3D, Del


from pyIPC import send


# DATASET = 'Datasets/func1.csv'
DATASET = 'Datasets/func2.csv'  # 3x + 6

# FUNCTIONS = [lambda x: 1, lambda x: sp.cos(x), lambda x: 1/(x ** 2 + 1), lambda x: x**4]

ALPHA = 0.005
GAMMA = 0.01

df = pd.read_csv(DATASET)
S = len(df)


x = df['x'].values.reshape((S, 1))
y = df['y'].values

ATTRS = 1 + 1

print(df.head(6))

S = len(df)


def error(X, Y, W):
    predicted = W[0] + sum([X[i-1] * W[i] for i in range(1, ATTRS)])
    base = (Y - predicted) ** 2
    
#     norm = GAMMA / Integer(2) * sum([x ** 2 for x in W])
    
    return base# + norm


weights = np.random.normal(size=ATTRS)

Y = sp.Symbol('y')
X = [sp.Symbol(f'x{i}') for i in range(1, ATTRS)]
W = [sp.Symbol(f'w{i}') for i in range(ATTRS)]

f = error(X, Y, W)


# print(weights)
# print(X)
# print(Y)
# print(W)
# print(f)


gradient_eq = derive_by_array(f, W)
hessian_eq = derive_by_array(derive_by_array(f, W), W)
    
gradient_general = lambdify(W + X + [Y], gradient_eq, "numpy")
hessian_general = lambdify(W + X + [Y], hessian_eq, "numpy")

for _ in range(100):    
    final_grad = (np.zeros((ATTRS)))
    final_hess = (np.zeros((ATTRS, ATTRS)))
    
    grad = lambda *X, Y: gradient_general(*weights, *X, Y)
    hess = lambda *X, Y: hessian_general(*weights, *X, Y)

    for i, j in zip(x, y):
        final_grad += grad(*i, Y=j)
        final_hess += hess(*i, Y=j)
        
    final_grad /= S
    final_hess /= S
    
    update = np.matmul(np.linalg.inv(final_hess), final_grad)
    
    alpha = ALPHA  # TODO: use formula
    
    weights -= alpha * update

    send(weights)






