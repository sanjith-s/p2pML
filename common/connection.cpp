#include "network_headers.h"

#include "connection.h"

void Connection::make_nonblocking(SOCKET socket) {
#ifdef WINDOWS
    u_long mode = 1;
    ioctlsocket(socket, FIONBIO, &mode);
#else
    int flags = fcntl(socket, F_GETFL);
    fcntl(socket, F_SETFL, flags | O_NONBLOCK);
#endif
}