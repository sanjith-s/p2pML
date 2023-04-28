import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Rational, Integer, Array, lambdify
from sympy.tensor.array import derive_by_array
from sympy.vector import CoordSys3D, Del


from pyIPC import send, recv, connectCPP

LOCALHOST = '127.0.0.1'
# DATASET = 'Datasets/func1.csv'
DATASET = 'node2_train.csv'  # 3x + 6
TEST_DATASET = 'node2_test.csv'
# DATASET = '/home/mona/Desktop/Datasets/cosdataset_train.csv'
# TEST_DATASET = '/home/mona/Desktop/Datasets/cosdataset_test.csv'

# FUNCTIONS = [lambda x: 1, lambda x: sp.cos(x), lambda x: 1/(x ** 2 + 1), lambda x: x**4]

ALPHA = 0.005
# GAMMA = 0.01

MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.5

K = 2.5

df = pd.read_csv(DATASET)

S = len(df)

x = df.iloc[:, :-1].values.reshape((S, 1))
y = df.iloc[:,-1].values


dfTest = pd.read_csv(TEST_DATASET)

S_test = len(dfTest)

testX = dfTest.iloc[:, :-1].values.reshape((S_test, 1))
testY = dfTest.iloc[:, -1].values

ATTRS = 1 + 1


S = len(df)


def find_outliers(points, k):
    mean = np.mean(points)
    std_dev = np.std(points)

    z_scores = [(point - mean) / std_dev for point in points]

    outliers = [i for i, z in enumerate(z_scores) if abs(z) > k]

    return outliers


def error(X, Y, W):
    predicted = W[0] + sum([X[i-1] * W[i] for i in range(1, ATTRS)])
    base = (Y - predicted) ** 2

    #     norm = GAMMA / Integer(2) * sum([x ** 2 for x in W])

    return base # + norm


def errorProp(X, Y, W):
    predicted = W[0] + sum([X[:,i-1] * W[i] for i in range(1, ATTRS)])
    base = (Y - predicted) ** 2

    #     norm = GAMMA / Integer(2) * sum([x ** 2 for x in W])

    return np.sum(base)/len(X) # + norm


py_socket = connectCPP()

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


def local_newton(weights, whitelist=None):
    if whitelist is None:
        ignore_list = []

    for _ in range(1000000):
        if (_ + 1) % 1000 == 0:
            # Ensure all nodes are at this phase
            send(py_socket, '[]', 'ACK')
            recv(py_socket, 'ACK')
            print('ACKs received')

            # Get weights of all nodes
            print(weights)
            send(py_socket, json.dumps(weights.tolist()), 'WEIGHTS')
            receieved_weights_all = recv(py_socket, 'WEIGHTS')
            receieved_weights_ip = ['127.0.0.1'] if LOCALHOST in whitelist else []
            receieved_weights = [weights] if LOCALHOST in whitelist else []
            for ip, weights in receieved_weights_all:
                if ip in whitelist:
                    receieved_weights_ip += [ip]
                    receieved_weights += [weights]

            total = np.sum(receieved_weights, axis=0)
            weights = total / (len(receieved_weights) + 1)
            print(weights)
            print('UPDATED')

            # Get loss of all nodes
            loss = errorProp(testX, testY, weights)
            print(loss)
            send(py_socket, str(loss), 'LOSS')
            received_loss_all = recv(py_socket, 'LOSS')
            received_loss_ip = ['127.0.0.1'] if LOCALHOST in whitelist else []
            received_loss = [loss] if LOCALHOST in whitelist else []

            for ip, loss in received_loss_all:
                if ip in whitelist:
                    received_loss_ip += [ip]
                    received_loss += [loss]

            outliers = find_outliers(received_loss, K)

            if len(outliers) == 0:
                return weights
            else:
                for index in sorted(outliers, reverse=True):
                    del whitelist[index]

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


weights = local_newton(weights, None)

is_leader = False

loss = errorProp(testX, testY, weights)
send(py_socket, str(loss) if is_leader else -1, 'LOSS')

ips = None
if is_leader:
    received_loss_all = recv(py_socket, 'LOSS')

    ips = ['127.0.0.1']
    total_loss = [loss]

    for ip, loss in received_loss_all:
        if loss != -1:
            ips += [ip]
            total_loss += [loss]

    outliers = find_outliers(total_loss, K)
    for index in sorted(outliers, reverse=True):
        del ips[index]

local_newton(weights, ips)

print('Adding to blockchain')
# TODO: add weights to blockchain
