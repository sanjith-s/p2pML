#ifndef P2P_NETWORK_HEADERS_H
#define P2P_NETWORK_HEADERS_H

//#define WINDOWS

#ifdef WINDOWS
#include <winsock2.h>
#include <ws2tcpip.h>

#else
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

#define closesocket(fd) close(fd)
using SOCKET = int;

#endif

#endif //P2P_NETWORK_HEADERS_H
