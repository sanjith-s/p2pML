import zmq
import time

import message_pb2


def send(weights=[3.14, 6.28]):
    context = zmq.Context()

#  Socket to talk to server
    print("Connecting to server…")
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://127.0.0.1:1101")
#socket.bind("tcp://*:1101")

    print("Sending message..")

    msg = message_pb2.Values()
# msg.param.append(3.14)
    msg.param.extend(weights)

    msgStr = msg.SerializeToString()

    # print(type(msgStr))

    message = socket.send_string(str(msgStr))
    print("Sent Message")

#send()
