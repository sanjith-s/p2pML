cmake_minimum_required(VERSION 3.22)
project(P2P)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

link_directories("/usr/local/lib")

add_executable(P2PServer server/servermain.cpp common/connection.h common/constants.h server/server.cpp server/server.h common/network_headers.h common/connection.cpp )

#add_executable(P2PClient client/clientmain.cpp common/connection.h common/constants.h
#        common/network_headers.h client/client.cpp client/client.h common/connection.cpp client/clientfunction.cpp client/pyIPC.h client/pyIPC.cpp client/message.pb.cc)

add_executable(P2PClient client/clientmain.cpp common/connection.h common/constants.h
        common/network_headers.h client/client.cpp client/client.h common/connection.cpp client/clientfunction.cpp client/pyIPC.h client/pyIPC.cpp)
include_directories(common server)

#target_link_libraries(P2PServer wsock32 ws2_32)
#target_link_libraries(P2PClient wsock32 ws2_32)

target_link_libraries(P2PClient pthread zmq zmqpp sodium)

#target_link_libraries(P2PClient pthread zmq zmqpp sodium protobuf)