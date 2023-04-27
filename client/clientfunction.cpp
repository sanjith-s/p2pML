#include "network_headers.h"

#include "client.h"
#include <string>
#include <iostream>
#include <chrono>
#include <future>
#include <cstring>
#include <vector>
#include "pyIPC.h"

std::pair<int, Client::output_type> Client::get_output() {
    std::string params{};

    int type;
    while (params.length() == 0) {
//        std::getline(std::cin >> std::ws, params);
//        params = contactIPC(py_socket, *this, type);
//        std::cout << params;
    }

    return std::make_pair(type, params);
}

void Client::operate() {
    std::string message{};
    MessageType type{};
    Client::MessageDirection command {contactIPC(py_socket, *this, type, message)};

    if (command == Client::SEND) {
        int len = static_cast<int>(message.size()) + 1;
        //        std::cout << len << "\n";

        for (auto i: connections) {
//            std::cout << "SEND " << type << " " << message << std::endl;

            send(i, reinterpret_cast<const char *>(&type), sizeof(len), 0);
            send(i, reinterpret_cast<const char *>(&len), sizeof(len), 0);
            send(i, message.c_str(), len, 0);
        }

//            future = std::async(&Client::get_output, this);
    } else if (command == Client::GET) {
        std::size_t length{connections.size()};
        bool *done = new bool[length]{false};

        bool flag{false};
        while (!flag) {
            for (std::size_t i = 0; i < length; i++) {
                if (done[i]) continue;

                SOCKET socket = connections[i];
                Client::MessageType recv_type{};
                int len{};
                //            int status = recv(socket, reinterpret_cast<char *>(&recv_type), sizeof(len), 0);
                if (recv(socket, reinterpret_cast<char *>(&recv_type), sizeof(recv_type), 0) != -1) {
                    if (recv_type != type) std::exit(1);

                    while (recv(socket, reinterpret_cast<char *>(&len), sizeof(len), 0) == -1);

                    char buffer[len];
                    std::memset(buffer, 0, len);
                    while (recv(socket, buffer, len, 0) == -1);

                    std::string output{buffer};
                    output = "[\"" + peers[i] + "\", " + output + "]";
//                    std::cout << "RECV " << output << std::endl;
                    sendIPC(py_socket, output);

                    done[i] = true;
                }
            }

            flag = true;
            for (std::size_t i = 0; i < length; i++) flag &= done[i];
        }
    }
}

