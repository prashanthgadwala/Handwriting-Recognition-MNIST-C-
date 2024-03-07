#pragma once

#include <torch/torch.h>

struct NeuralNet : torch::nn::Module {
    NeuralNet(int64_t input_size, int64_t hidden_size, int64_t num_classes) :
        fc1(register_module("fc1", torch::nn::Linear(input_size, hidden_size))),
        fc2(register_module("fc2", torch::nn::Linear(hidden_size, num_classes))) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.view({x.size(0), -1})));
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    void save_model(const std::string& filename) {
        torch::save(*this, filename);
    }

    void load_model(const std::string& filename) {
        torch::load(*this, filename);
    }

    torch::nn::Linear fc1, fc2;
};