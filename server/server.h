#ifndef P2P_SERVER_H
#define P2P_SERVER_H

#include "network_headers.h"
#include "connection.h"
#include "constants.h"
#include <unistd.h>
#include <vector>
#include <string>

class Server: public Connection {
private:
    int backlog{constants::backlog};
    std::vector<SOCKET> connections{};

    void initialize_socket(int domain, int service, int protocol);

public:
    std::vector<std::string> clients{};

    Server(int domain, int service, int protocol, unsigned long ip, int port):
            Connection{domain, service, protocol, ip, port} {
        address.sin_family = static_cast<short>(domain);
        address.sin_port = htons(port);
        address.sin_addr.s_addr = htonl(ip);

        initialize_socket(domain, service, protocol);
    }

    [[nodiscard]] SOCKET get_socket() const {return sockfd;}

    bool add_if_new(std::string &new_client, int new_port, SOCKET connection);
    void remove(std::string client, int port);

};

#endif //P2P_SERVER_H
