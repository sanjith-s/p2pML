#include "network_headers.h"

#include "connection.h"
#include "client.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

void Client::initialize_socket(int domain, int service, int protocol) {
    sockfd = socket(domain, service, protocol);
    if (!sockfd)
        throw std::runtime_error{"Failed to connect to sockfd"};

    if (connect(sockfd,
                reinterpret_cast<sockaddr *>(&address),
                sizeof(address)
    ) < 0)
        throw std::runtime_error{"Unable to connect"};

    std::cout << "Connected\n";
    Connection::make_nonblocking(sockfd);

    recv_sockfd = socket(domain, service, protocol);
}

void Client::accept(int count) {
    recv_address.sin_family = AF_INET;
    recv_address.sin_port = htons(constants::recv_port);
    recv_address.sin_addr.s_addr = INADDR_ANY;

    if (bind(
            recv_sockfd,
            reinterpret_cast<struct sockaddr *>(&recv_address),
            sizeof(recv_address)
    ) < 0)
        throw std::runtime_error{"Failed to bind to recv_sockfd"};

    if (listen(recv_sockfd, count) < 0)
        throw std::runtime_error{"Failed to start listening (recv_sockfd)"};

    socklen_t len{sizeof(recv_address)};
    for (int i = 0; i < count; i++) {
        SOCKET connection = ::accept(recv_sockfd, reinterpret_cast<sockaddr *>(&recv_address), &len);
        Connection::make_nonblocking(connection);

        std::string ip{inet_ntoa(recv_address.sin_addr)};

        connections.push_back(connection);
        peers.emplace_back(ip);

        std::cout << "Accepted " << ip << '\n';
    }
}

void Client::accept_new_node() {
    int adding{0};
    int data_exists = recv(sockfd, reinterpret_cast<char *>(&adding), sizeof(adding), 0);
    if (data_exists != -1) {
        if (adding == 1) {
            SOCKET sock = socket(domain, service, protocol);

            char new_ip[constants::ip_size];
            std::memset(new_ip, 0, constants::ip_size);
            int port{0};
            while (recv(sockfd, new_ip, constants::ip_size, 0) == -1);

            while (recv(sockfd, reinterpret_cast<char *>(&port), sizeof(port), 0) == -1);

            sockaddr_in new_addr{};
            new_addr.sin_family = AF_INET;
            new_addr.sin_port = htons(constants::recv_port);
            new_addr.sin_addr.s_addr = inet_addr(new_ip);

            if (connect(sock,
                        reinterpret_cast<sockaddr *>(&new_addr),
                        sizeof(new_addr)) < 0) {
                std::string error{"Unable to connect to "};
                error += new_ip;
                error += ":" + std::to_string(port) + '\n';
                throw std::runtime_error{error};
            }

            Connection::make_nonblocking(sock);
            connections.push_back(sock);
            peers.emplace_back(new_ip);

            std::cout << "Added " << new_ip << '\n';
        }
    }
}