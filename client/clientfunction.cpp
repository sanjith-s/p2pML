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
    std::string params{};
    while (params.length() == 0) {
//        std::getline(std::cin >> std::ws, params);
        params = getParams();
//        std::cout << params << "THIS IS PARAMETERS" << std::endl;
    }

    return params;
}

int ack_timeout() {
    usleep(100000);
    return 1;
}

void Client::operate() {
    if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        output_type output = future.get();
        int len = static_cast<int>(output.size()) + 1;
        std::cout << len << "\n";

        for (auto i : connections) {
            std::cout << output << std::endl;

            send(i, reinterpret_cast<const char *>(&len), sizeof(len), 0);

            std::future<int> ack = std::async(ack_timeout);

            int ack_store{};
            while (
                    recv(i, reinterpret_cast<char *>(&ack_store), sizeof(int), 0) == -1 &&
                    ack.wait_for(std::chrono::seconds(0)) != std::future_status::ready
                    );

            if (ack.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                std::cout << "NOT ACKED" << std::endl;
                std::exit(1);
            }

            send(i, output.c_str(), len, 0);
        }

        future = std::async(&Client::get_output, this);
    }

    for (int i = 0; i < connections.size(); i++) {
        SOCKET socket = connections[i];
        int len{0};
        int status = recv(socket, reinterpret_cast<char *>(&len), sizeof(len), 0);
        if (status != -1) {
            for (auto j : peers) std::cout << connections.size() << "\n";

            std::cout << len << std::endl;

            send(socket, reinterpret_cast<const char *>(&len), sizeof(len), 0);

            char buffer[len];
            std::memset(buffer, 0, len);
            while (recv(socket, buffer, len, 0) == -1);
            std::cout << peers[i] << " says: " << buffer << std::endl;
        }
    }
}

