//
// Created by Sanjith on 18/12/22.
//

#ifndef P2P_PYIPC_H
#define P2P_PYIPC_H

#endif //P2P_PYIPC_H

#include "client.h"
#include <zmqpp/zmqpp.hpp>
#include <vector>
#include <string>

Client::MessageDirection contactIPC(zmqpp::socket &socket, const Client &client, Client::MessageType &type, std::string &message);
void sendIPC(zmqpp::socket &socket, const std::string &message);