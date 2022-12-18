#ifndef P2P_CLIENT_H
#define P2P_CLIENT_H

#include "network_headers.h"
#include "constants.h"

#include "connection.h"
#include <utility>
#include <vector>
#include <functional>
#include <future>
#include <iostream>

class Client: public Connection {
private:
    using output_type = std::string;

    SOCKET recv_sockfd{};
    sockaddr_in recv_address{};

    std::vector<SOCKET> connections{};
    std::future<output_type> future;

    void initialize_socket(int domain, int service, int protocol);
    output_type get_output();

public:
    std::vector<std::string> peers{};

    Client(int domain, int service, int protocol, unsigned long ip, int port):
        Connection{domain, service, protocol, ip, port} {
        address.sin_family = static_cast<short>(domain);
        address.sin_port = htons(constants::port);
        address.sin_addr.s_addr = ip;

        future = std::async(&Client::get_output, this);

        initialize_socket(domain, service, protocol);
    }

    ~Client() {
        std::cout << "deleted\n";
        for (auto i : connections) closesocket(i);
    }

    [[nodiscard]] SOCKET get_socket() const {return sockfd;}

    void accept(int count);
    void accept_new_node();
    void operate();
};

#endif //P2P_CLIENT_H
