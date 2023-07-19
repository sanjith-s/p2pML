import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Rational, Integer, Array, lambdify
from sympy.tensor.array import derive_by_array
from sympy.vector import CoordSys3D, Del
import itertools

from blockchain import Blockchain

from pyIPC import send, recv, connectCPP

import socket

LOCALHOST = socket.gethostbyname(socket.gethostname())  # '127.0.0.1'
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

K = 1.2

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
    print(points)

    mean = np.mean(points)
    std_dev = np.std(points)

    if std_dev == 0:
        return []

    z_scores = [(point - mean) / std_dev for point in points]

    outliers = [i for i, z in enumerate(z_scores) if abs(z) > k]

    print('Z', z_scores)
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


right, wrong = 0, 0

cluster1 = []
cluster2 = []

file1_path = 'cluster_ips_a'
file2_path = 'cluster_ips_b'


with open(file1_path, 'r') as cluster1_data:
    for ip in cluster1_data:
        cluster1 += [ip[:-1]]

with open(file2_path, 'r') as cluster2_data:
    for ip in cluster2_data:
        cluster2 += [ip[:-1]]

all_nodes = cluster1 + cluster2


def swap_rows(cluster_a, cluster_b, k):

    temp = cluster_a[len(cluster_a) - 1 - k]
    cluster_a[len(cluster_a) - 1 - k] = cluster_b[len(cluster_b) - 1 - k]
    cluster_b[len(cluster_b) - 1 - k] = temp
    # rows1 = lines1[len(lines1) - 1 - k].strip().split('.')
    # rows2 = lines2[len(lines2) - 1 - k].strip().split('.')
    # lines1[len(lines1) - 1 - k] = '.'.join(rows2) + '\n'
    # lines2[len(lines2) - 1 - k] = '.'.join(rows1) + '\n'

    # with open(file1_path, 'w') as file1:
    #     file1.writelines(lines1)
    #
    # with open(file2_path, 'w') as file2:
    #     file2.writelines(lines2)
    print("DONE!!!!")


def local_newton(weights, whitelist=None):
    global right, wrong

    if whitelist is None:
        whitelist = []

    is_outlier = False
    for _ in range(1000000):
        if (_ + 1) % 1000 == 0:
            print('WHITELIST', whitelist)

            # Ensure all nodes are at this phase
            send(py_socket, '[]', 'ACK')
            recv(py_socket, 'ACK')
            print('ACKs received')

            # Get weights of all nodes
            print(weights)
            send(py_socket, json.dumps(weights.tolist()), 'WEIGHTS')
            receieved_weights_all = recv(py_socket, 'WEIGHTS')
            receieved_weights_ip = []
            receieved_weights = []
            for allowed in whitelist:
                if allowed == LOCALHOST:
                    receieved_weights_ip += [LOCALHOST]
                    receieved_weights += [weights]
                    continue

                for ip, recv_weights in receieved_weights_all:
                    if ip == allowed:
                        receieved_weights_ip += [ip]
                        receieved_weights += [recv_weights]
                        break

            total = np.sum(receieved_weights, axis=0)
            weights = total / (len(receieved_weights) + 1)
            print(weights)
            print('UPDATED')

            # Get loss of all nodes
            loss = errorProp(testX, testY, weights)
            print(loss)
            send(py_socket, str(loss), 'LOSS')
            received_loss_all = recv(py_socket, 'LOSS')
            received_loss_ip = []
            received_loss = []

            for allowed in whitelist:
                if allowed == LOCALHOST:
                    received_loss_ip += [LOCALHOST]
                    received_loss += [loss]
                    continue

                for ip, recv_loss in received_loss_all:
                    if ip == allowed:
                        received_loss_ip += [ip]
                        received_loss += [recv_loss]
                        break

            outliers = find_outliers(received_loss, K)
            print(outliers)

            if len(outliers) == 0:
                if is_outlier:
                    wrong += 1
                else:
                    right += 1

                return weights
            else:
                for index in sorted(outliers, reverse=True):
                    if whitelist[index] == LOCALHOST:
                        is_outlier = True

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


nodes = len(all_nodes)  # len(cluster1) + len(cluster2)
if nodes % 2 == 0:
    all_nodes += [-1]

count = 0
for combination in itertools.combinations(all_nodes, nodes // 2):
    if -1 in combination:
        continue

    count += 1

    cluster_ips = list(combination) if LOCALHOST in combination else [ip for ip in all_nodes
                                                                      if ip not in combination and ip != -1]
    # print(count, '\t', cluster_ips, tuple(cluster_ips) == combination)
    weights = local_newton(weights, cluster_ips.copy())

    send(py_socket, '[]', 'CLUSTER_ACK')
    recv(py_socket, 'CLUSTER_ACK')

print(right, wrong)

is_leader = right >= wrong

loss = errorProp(testX, testY, weights)
send(py_socket, str(loss) if is_leader else '-1', 'LOSS')
ips = ['127.0.0.1']
# ips = []
if is_leader:
    received_loss_all = recv(py_socket, 'LOSS')

    for ip, loss in received_loss_all:
        if loss != -1:
            ips += [ip]

    local_newton(weights, ips.copy())

send(py_socket, '[]', 'CLUSTER_ACK')
recv(py_socket, 'CLUSTER_ACK')

if is_leader:
    print("Final Weigths: ", weights)

    blockchain = Blockchain()
    blockchain.addBlock(str(weights))
    print('Adding to blockchain')
