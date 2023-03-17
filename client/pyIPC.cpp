//
// Created by ubuntu on 18/12/22.
//
#include "pyIPC.h"
#include "client.h"
#include <zmqpp/zmqpp.hpp>
#include <string>
#include <iostream>
//#include "message.pb.h"

std::pair<zmqpp::context, zmqpp::socket_type> connectPy() {
    zmqpp::context context{};
    zmqpp::socket_type type {zmqpp::socket_type::pair};

    return std::pair{std::move(context), type};
}

std::string vector_to_string(const std::vector<std::string> &vector) {
    std::string ret_val{"["};
    bool first{true};
    for (const auto &i : vector) {
        if (!first) ret_val.append(",");
        else first = false;
        ret_val.append(i);
    }
    ret_val.append("]");

    return ret_val;
}

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
    else return Client::ERROR;

    if (direction == Client::SEND) socket.receive(message);
    else socket.send(std::to_string(client.peers.size()));

    return direction;

//    if (message == "GET_WEIGHTS") {
//        socket.send(vector_to_string(client.stored));
//        socket.send(std::to_string(client.peers.size()));
//        return Client::GET_WEIGHTS;
//    } else if (message == "GET_LOSS") {
//        socket.send(std::to_string(client.peers.size()));
//        return Client::GET_LOSS;
//    } else if (message == "GET_ACK") {
//        socket.send(std::to_string(client.peers.size()));
//        return Client::GET_ACK;
//    } else if (message == "SEND_WEIGHTS") {
//        socket.receive(message);
//        return Client::SEND_WEIGHTS;
//    } else if (message == "SEND_LOSS") {
//        socket.receive(message);
//        return Client::SEND_LOSS;
//    } else if (message == "SEND_ACK") {
//        socket.receive(message);
//        return Client::SEND_ACK;
//    } else {
//        return Client::ERROR;
//    }
}

void sendIPC(zmqpp::socket &socket, const std::string &message) {
    std::cout << message << '\n';
    socket.send(message);
}
