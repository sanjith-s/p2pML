import copy
import json
import itertools
from blockchain import Blockchain
from pyIPC import send, recv, connectCPP
import socket
import pandas as pd
import numpy as np
from network import ANN

LOCALHOST = socket.gethostbyname(socket.gethostname())  # '127.0.0.1'

DATASET = 'node1_train.csv'  # 3x + 6
TEST_DATASET = 'node1_test.csv'

df = pd.read_csv(DATASET)

trainX = df.iloc[:, :-1].values
trainY = df.iloc[:, -1].values

trainY = trainY.reshape(-1, 1)

ALPHA = 0.005  # Learning rate

MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.5

K = 1.2

df1 = pd.read_csv(TEST_DATASET)

testX = df1.iloc[:, :-1].values
testY = df1.iloc[:, -1].values

testY = testY.reshape(-1, 1)

py_socket = connectCPP()

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

model = ANN(dim=(1, 2, 1), lr=ALPHA)  # Initializing NeuralNet
print(model.weights)


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


def swap_rows(cluster_a, cluster_b, k):
    temp = cluster_a[len(cluster_a) - 1 - k]
    cluster_a[len(cluster_a) - 1 - k] = cluster_b[len(cluster_b) - 1 - k]
    cluster_b[len(cluster_b) - 1 - k] = temp

    print("DONE!!!!")


def calc_error(x, y):
    error = 0
    for x, y in zip(testX, testY):
        error += model.my_error(x, y)

    return (error / len(testX))[0]


def local_newton(whitelist=None):
    global right, wrong

    if whitelist is None:
        whitelist = []

    is_outlier = False
    for _ in range(1000):
        if (_ + 1) % 100 == 0:
            print('WHITELIST', whitelist)

            # Ensure all nodes are at this phase
            send(py_socket, '[]', 'ACK')
            recv(py_socket, 'ACK')
            print('ACKs received')

            # Get weights of all nodes
            my_Weights = copy.deepcopy(model.weights)
            my_weight_list = []

            for arr in my_Weights:
                my_weight_list.append(arr.tolist())

            print(my_weight_list)

            send(py_socket, json.dumps(my_weight_list), 'WEIGHTS')
            receieved_weights_all = recv(py_socket, 'WEIGHTS')
            receieved_weights_ip = []
            receieved_weights = []
            for allowed in whitelist:
                if allowed == LOCALHOST:
                    receieved_weights_ip += [LOCALHOST]
                    receieved_weights += [model.weights]
                    continue

                for ip, recv_weights in receieved_weights_all:
                    if ip == allowed:
                        receieved_weights_ip += [ip]
                        receieved_weights += [recv_weights]
                        break

            weight_sum = []
            for arr in model.weights:
                weight_sum.append(np.zeros_like(arr))

            for weight_arr in receieved_weights:
                for i in range(len(weight_sum)):
                    for j in range(len(weight_sum[i])):
                        for k in range(len(weight_sum[i][j])):
                            weight_sum[i][j][k] += weight_arr[i][j][k]

            for i in range(len(weight_sum)):
                for j in range(len(weight_sum[i])):
                    for k in range(len(weight_sum[i][j])):
                        weight_sum[i][j][k] /= (len(receieved_weights) + 1)

            print(weight_sum)

            # total = np.sum(receieved_weights, axis=0)
            # weights = total / (len(receieved_weights) + 1)

            model.weights = weight_sum
            print(model.weights)
            print('UPDATED')

            # Get loss of all nodes
            loss = calc_error(testX, testY)
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

                return model.weights
            else:
                for index in sorted(outliers, reverse=True):
                    if whitelist[index] == LOCALHOST:
                        is_outlier = True

                    del whitelist[index]

        model.train(trainX, trainY, 1)

        if (_ + 1) % 100 == 0:
            print('WEIGHTS', model.weights)


nodes = len(all_nodes)  # len(cluster1) + len(cluster2)
if nodes % 2 == 0:
    all_nodes += [-1]

count = 0
for combination in itertools.combinations(all_nodes, nodes // 2):
    if -1 in combination:
        continue
    print(f"------------------ Combination {count} -----------------")
    count += 1

    cluster_ips = list(combination) if LOCALHOST in combination else [ip for ip in all_nodes
                                                                      if ip not in combination and ip != -1]
    # print(count, '\t', cluster_ips, tuple(cluster_ips) == combination)
    model.weights = local_newton(cluster_ips.copy())

    send(py_socket, '[]', 'CLUSTER_ACK')
    recv(py_socket, 'CLUSTER_ACK')

print(right, wrong)

is_leader = right >= wrong

loss = calc_error(testX, testY)
send(py_socket, str(loss) if is_leader else '-1', 'LOSS')
ips = [LOCALHOST]

if is_leader:
    received_loss_all = recv(py_socket, 'LOSS')

    for ip, loss in received_loss_all:
        if loss != -1:
            ips += [ip]

    local_newton(ips.copy())

send(py_socket, '[]', 'CLUSTER_ACK')
recv(py_socket, 'CLUSTER_ACK')

if is_leader:
    print("Final Weigths: ", model.weights)

    blockchain = Blockchain()
    blockchain.addBlock(str(model.weights))
    print('Adding to blockchain')
