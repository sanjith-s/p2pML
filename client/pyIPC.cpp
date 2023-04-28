//
// Created by ubuntu on 18/12/22.
//
#include "pyIPC.h"
#include "client.h"
#include <zmqpp/zmqpp.hpp>
#include <string>
#include <iostream>
//#include "message.pb.h"

Client::MessageDirection contactIPC(zmqpp::socket &socket, const Client &client, Client::MessageType &type, std::string &message) {
    socket.receive(message, true);

    Client::MessageDirection direction{};

    if (message == "GET") direction = Client::GET;
    else if (message == "SEND") direction = Client::SEND;
    else return Client::ERROR;

    socket.receive(message);

    if (message == "WEIGHTS") type = Client::WEIGHTS;
    else if (message == "LOSS") type = Client::LOSS;
    else if (message == "ACK") type = Client::ACK;
    else if (message == "CLUSTER_ACK") type = Client::CLUSTER_ACK;
    else return Client::ERROR;

    if (direction == Client::SEND) socket.receive(message);
    else socket.send(std::to_string(client.peers.size()));

    return direction;
}

void sendIPC(zmqpp::socket &socket, const std::string &message) {
//    std::cout << message << '\n';
    socket.send(message);
}
