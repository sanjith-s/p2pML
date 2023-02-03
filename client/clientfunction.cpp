#include "network_headers.h"

#include "client.h"
#include <string>
#include <iostream>
#include <chrono>
#include <future>
#include <cstring>
#include <vector>
#include "pyIPC.h"

std::string Client::get_output() {
    std::string input{};
    std::string params;
    while (input.length() == 0) {
        //std::getline(std::cin >> std::ws, input);
        params = getParams();
        std::cout << params << std::endl;
    }

    return params;
}

void Client::operate() {
    if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        output_type output = future.get();
        int len = static_cast<int>(output.size()) + 1;

        for (auto i : connections) {
            send(i, reinterpret_cast<const char *>(&len), sizeof(len), 0);
            send(i, output.c_str(), len, 0);
        }

        future = std::async(&Client::get_output, this);
    }

    for (int i = 0; i < connections.size(); i++) {
        SOCKET socket = connections[i];
        int len{0};
        int status = recv(socket, reinterpret_cast<char *>(&len), sizeof(len), 0);
        if (status != -1) {
            char buffer[len];
            std::memset(buffer, 0, len);
            while (recv(socket, buffer, len, 0) == -1);
            std::cout << peers[i] << " says: " << buffer << std::endl;
        }
    }
}

