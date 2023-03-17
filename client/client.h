#ifndef P2P_CLIENT_H
#define P2P_CLIENT_H

#include "network_headers.h"
#include "constants.h"
#include "connection.h"

#include <zmqpp/zmqpp.hpp>
#include <utility>
#include <vector>
#include <functional>
#include <future>
#include <iostream>


class Client: public Connection {
public:
    using output_type = std::string;

    enum MessageDirection {
        GET = 0,
        SEND = 1,
        ERROR = 2
    };

    enum MessageType {
        WEIGHTS = 0,
        LOSS = 1,
        ACK = 2,
    };

private:
//    output_type default_stored{"[]"};

    SOCKET recv_sockfd{};
    sockaddr_in recv_address{};

    std::vector<SOCKET> connections{};
//    std::future<std::pair<int, output_type>> future;

    zmqpp::context context{};
    zmqpp::socket py_socket;

    void initialize_socket(int domain, int service, int protocol);
    std::pair<int, output_type> get_output();

public:
    std::vector<std::string> peers{};
//    std::vector<output_type> stored{};

    Client(int domain, int service, int protocol, unsigned long ip, int port, zmqpp::socket_type socket_type):
        Connection{domain, service, protocol, ip, port},
        py_socket{context, socket_type} {
        address.sin_family = static_cast<short>(domain);
        address.sin_port = htons(constants::port);
        address.sin_addr.s_addr = ip;

//        future = std::async(&Client::get_output, this);

        py_socket.bind(std::string{constants::endpoint});

        initialize_socket(domain, service, protocol);
    }

    ~Client() {
        std::cout << "deleted\n";
        for (auto i : connections) closesocket(i);
        py_socket.close();
    }

    [[nodiscard]] SOCKET get_socket() const {return sockfd;}

    void accept(int count);
    void accept_new_node();
    void operate();
};

#endif //P2P_CLIENT_H
