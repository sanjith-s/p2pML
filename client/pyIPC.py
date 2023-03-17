import zmq
import time
import json
import ast


# import message_pb2

def connectCPP():
    context = zmq.Context()
    print("Connecting to CPP…")
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://127.0.0.1:1101")

    return socket


def send(socket: zmq.Socket, data, data_type):

    # print("Sending message..")
    socket.send_string('SEND')
    socket.send_string(data_type)

    socket.send_string(data)
    # print(msg_str)
    # print(message)
    # print("Sent Message")


def recv(socket: zmq.Socket, data_type: str):

    socket.send_string('GET')
    socket.send_string(data_type)

    # print("Receiving message..")

    length = int(socket.recv_string())

    output = []
    for _ in range(length):
        output += [ast.literal_eval(socket.recv_string())]

    # message = socket.recv_string()
    # output = ast.literal_eval(message)

    print(output)
    # print("Received Message")

    return output
