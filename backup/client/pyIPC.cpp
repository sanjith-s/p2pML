//
// Created by ubuntu on 18/12/22.
//
#include <zmqpp/zmqpp.hpp>
#include <string>
#include <iostream>
//#include "message.pb.h"

using namespace std;

string getParams() {
    const string endpoint = "tcp://*:1101";

    zmqpp::context context;

    zmqpp::socket_type type = zmqpp::socket_type::pull;
    zmqpp::socket socket (context, type);

//    cout << "Binding to " << endpoint << "..." << endl;
    socket.bind(endpoint);


//    cout << "Receiving message..." << endl;

//    char buff [256];
//    int nbytes = zmq_recv (socket, buff, 256, 0); assert (nbytes != -1);

    string message;
    socket.receive(message);

//    ipc::Values response;
//    response.ParseFromString(message);


//    cout << "Received message" << endl;

//    vector<float> result;
//    for(int i=0; i<response.param_size(); ++i) {
//        printf("Index(%d): Value(%f)\n", i, response.param(i));
//        result.push_back(response.param(i));
//    }
//
//    cout << "Finished." << endl;

    return message;
}
