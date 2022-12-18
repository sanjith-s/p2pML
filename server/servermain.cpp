#include "network_headers.h"

#include "connection.h"
#include "constants.h"
#include "server.h"

#include <string>
#include <iostream>

void accept_clients(Server &server) {
    auto *address{reinterpret_cast<sockaddr *>(&server.address)};
    socklen_t addrlen{sizeof(server.address)};

    std::string self_ip{};

    while (true) {
        SOCKET client{accept(server.get_socket(), address, &addrlen)};
        Connection::make_nonblocking(client);

        std::string ip{inet_ntoa(server.address.sin_addr)};
        int port{ntohs(server.address.sin_port)};
        std::cout << "Accepted " << ip << '\n';

        if (ip == constants::local_ip && !self_ip.empty()) ip = self_ip;
        else if (ip != constants::local_ip && self_ip.empty()) {
            sockaddr_in local_addr{};
            socklen_t len = sizeof(local_addr);
            char local_ip[constants::ip_size];

            getsockname(client, reinterpret_cast<sockaddr *>(&local_addr), &len);
            inet_ntop(AF_INET, &local_addr.sin_addr, local_ip, constants::ip_size);

            self_ip = local_ip;
        }

        server.add_if_new(ip, port, client);
    }
}

int main() {
#ifdef WINDOWS
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif

    Server server{
        AF_INET,
        SOCK_STREAM,
        0,
        INADDR_ANY,
        constants::port
    };

    accept_clients(server);

#ifdef WINDOWS
    WSACleanup();
#endif
    return 0;
}
