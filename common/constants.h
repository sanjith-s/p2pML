#ifndef P2P_CONSTANTS_H
#define P2P_CONSTANTS_H

#include <string_view>

namespace constants {
    inline constexpr int port{9092};
    inline constexpr int recv_port{9093};
    inline constexpr int backlog{64};
    inline constexpr int ip_size{16};
    inline constexpr std::string_view local_ip{"127.0.0.1"};
    inline constexpr std::string_view endpoint{"tcp://*:1101"};
}

#endif //P2P_CONSTANTS_H
