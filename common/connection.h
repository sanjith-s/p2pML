#ifndef P2P_CONNECTION_H
#define P2P_CONNECTION_H

#include "network_headers.h"

class Connection {
private:
    unsigned long ip{};
    int port{};

protected:
    SOCKET sockfd{};
    int domain{};
    int service{};
    int protocol{};

public:
    sockaddr_in address{};

    Connection(int domain, int service, int protocol, unsigned long ip, int port):
        domain{domain}, service{service}, protocol{protocol}, ip{ip}, port{port} {
    }

    ~Connection() {
        closesocket(sockfd);
    }

    static void make_nonblocking(SOCKET socket);
};

#endif //P2P_CONNECTION_H
