#include "network_headers.h"

#include "server.h"
#include <stdexcept>
#include <algorithm>

void Server::initialize_socket(int domain, int service, int protocol) {
    sockfd = socket(domain, service, protocol);
    if (!sockfd)
        throw std::runtime_error("Failed to connect to sockfd");

    if (bind(
            sockfd,
            reinterpret_cast<struct sockaddr *>(&address),
            sizeof(address)
    ) < 0)
        throw std::runtime_error("Failed to bind to sockfd");

    if (listen(sockfd, backlog) < 0)
        throw std::runtime_error("Failed to start listening");
}

bool Server::add_if_new(std::string &new_client, int new_port, SOCKET connection) {
    bool is_new = std::find(clients.begin(), clients.end(), new_client) == clients.end();
    if (is_new) {
        int len = static_cast<int>(clients.size());
        send(connection, reinterpret_cast<const char *>(&len), sizeof(len), 0);

        for (auto i : connections) {
            int adding = 1;
            const char *client = new_client.c_str();
            send(i, reinterpret_cast<const char *>(&adding), sizeof(bool), 0);
            send(i, client, constants::ip_size, 0);
            send(i, reinterpret_cast<const char *>(&new_port), sizeof(int), 0);
        }

        connections.push_back(connection);
        clients.push_back(std::move(new_client));
    }

    return is_new;
}

void Server::remove(std::string client, int port) {
    return;
}