#include "network_headers.h"
#include "constants.h"
#include "client.h"

void chat(Client &client) {
    while (true) {
        client.accept_new_node();
        client.operate();
    }
}

int main(int argc, char *argv[]) {
#ifdef WINDOWS
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif

    std::string_view server_ip{constants::local_ip};

    if (argc > 1) server_ip = argv[1];

    Client client{
            AF_INET,
            SOCK_STREAM,
            0,
            inet_addr(server_ip.data()),
            constants::port,
    };

    int nodes{};
    while (recv(client.get_socket(), reinterpret_cast<char *>(&nodes), sizeof(nodes), 0) == -1);
    client.accept(nodes);
    chat(client);

#ifdef WINDOWS
    WSACleanup();
#endif
    return 0;
}