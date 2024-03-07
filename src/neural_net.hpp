#pragma once

// Add the necessary include path for the Torch library
#include <torch/torch.h>

struct NeuralNet : torch::nn::Module {
    NeuralNet(int64_t input_size, int64_t hidden_size, int64_t num_classes) :
        fc1(input_size, hidden_size),
        fc2(hidden_size, num_classes) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.view({x.size(0), -1})));
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    void serialize(torch::serialize::InputArchive& archive) override {
        archive(fc1, fc2);
    }

    void serialize(torch::serialize::OutputArchive& archive) const override {
        archive(fc1, fc2);
    }

    torch::nn::Linear fc1, fc2;
};