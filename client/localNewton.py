import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Rational, Integer, Array, lambdify
from sympy.tensor.array import derive_by_array
from sympy.vector import CoordSys3D, Del


from pyIPC import send, recv, connectCPP


DATASET = '/home/mona/Desktop/Datasets/cosdataset.csv'
# DATASET = 'node1_train.csv'  # 3x + 6

# FUNCTIONS = [lambda x: 1, lambda x: sp.cos(x), lambda x: 1/(x ** 2 + 1), lambda x: x**4]

ALPHA = 0.005
GAMMA = 0.01

MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.5

df = pd.read_csv(DATASET)


S = len(df)


x = df.iloc[:, 0].values.reshape((S, 1))
y = df.iloc[:,-1].values

dfTest = pd.read_csv(DATASET)
testX = dfTest.iloc[:,:-1].values.reshape((S, 1))
testY = dfTest.iloc[:,-1].values

ATTRS = 1 + 1


S = len(df)


def error(X, Y, W):
    predicted = W[0] + sum([X[i-1] * W[i] for i in range(1, 2)])

    base = (Y-predicted)**2
    #     norm = GAMMA / Integer(2) * sum([x ** 2 for x in W])

    return base  #+ norm

def errorProp(X, Y, W):
    predicted = W[0] + sum([X[:,i-1] * W[i] for i in range(1, 2)])

    base = (Y-predicted)**2
    #     norm = GAMMA / Integer(2) * sum([x ** 2 for x in W])

    return np.sum(base)/len(X)

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

py_socket = connectCPP()

add_to_blockchain = True
for _ in range(1000000):
    if (_ + 1) % 1000 == 0:
        # Ensure all nodes are at this phase
        send(py_socket, '[]', 'ACK')
        recv(py_socket, 'ACK')

        # Get weights of all nodes
        send(py_socket, json.dumps(weights.tolist()), 'WEIGHTS')
        receieved_weights = np.array(recv(py_socket, 'WEIGHTS'))

        total = weights
        total += np.sum(receieved_weights, axis=0)
        weights = total / (len(receieved_weights) + 1)
        print('UPDATED')

        loss = errorProp(testX, testY, weights)
        print("Loss: ", loss)
        # Get loss of all nodes
        send(py_socket, str(loss), 'LOSS')  # TODO: replace get_loss with real loss function
        received_loss = recv(py_socket, 'LOSS')
        max_diff = np.ptp(received_loss)

        if max_diff < MIN_THRESHOLD:
            break
        elif max_diff > MAX_THRESHOLD:
            add_to_blockchain = False
            break

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

    if (_ + 1) % 100 == 0:
        print('WEIGHTS', weights)
        # send(py_socket, weights, 'WEIGHTS')

print("Final Weigths: ", weights)

