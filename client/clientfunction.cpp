#include "network_headers.h"

#include "client.h"
#include <string>
#include <iostream>
#include <chrono>
#include <future>
#include <cstring>
#include <vector>
#include "pyIPC.h"

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
        if (length <= 0) return;

        bool *done = new bool[length]{false};

        if (type == Client::CLUSTER_ACK) {
            while (!cluster_acks.empty()) {
                std::pair<int, Client::output_type> clack = cluster_acks.back();
                cluster_acks.pop_back();
                done[clack.first] = true;
                sendIPC(py_socket, clack.second);
            }
        }

        bool flag{true};
        for (std::size_t i = 0; i < length; i++) flag &= done[i];

        while (!flag) {
            for (std::size_t i = 0; i < length; i++) {
                if (done[i]) continue;

                for (const auto& clack : cluster_acks) {
                    if (i == clack.first) {
                        sendIPC(py_socket, "[\"" + peers[i] + "\", 0]");
                        done[i] = true;
                        continue;
                    }
                }

                SOCKET socket = connections[i];
                Client::MessageType recv_type{};
                int len{};
                //            int status = recv(socket, reinterpret_cast<char *>(&recv_type), sizeof(len), 0);
                if (recv(socket, reinterpret_cast<char *>(&recv_type), sizeof(recv_type), 0) != -1) {
                    while (recv(socket, reinterpret_cast<char *>(&len), sizeof(len), 0) == -1);

                    char buffer[len];
                    std::memset(buffer, 0, len);
                    while (recv(socket, buffer, len, 0) == -1);

                    std::string output{"[\"" + peers[i] + "\", " + buffer + "]"};

                    if (recv_type != type) {
                        if (recv_type == Client::CLUSTER_ACK) {
                            cluster_acks.emplace_back(i, output);
                            sendIPC(py_socket, "[\"" + peers[i] + "\", 0]");
                            done[i] = true;
                        } else if (type != Client::CLUSTER_ACK) {
                            std::cout << "Incorrect type received: expected " << type
                            << " got " << recv_type << std::endl;
                        }

                        continue;
                    }

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

