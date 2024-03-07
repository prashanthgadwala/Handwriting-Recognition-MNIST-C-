#include "neural_net.hpp"
#include <torch/torch.h>
#include <iostream>

int main() {
    NeuralNet model(784, 500, 10);
    model.load_model("mnist_model.pt");

    auto test_dataset = torch::data::datasets::MNIST("./mnist-data", torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                            std::move(test_dataset));

    model.eval();
    torch::NoGradGuard no_grad;
    int64_t correct = 0;
    int64_t total = 0;

    for (const auto& batch : *test_loader) {
        auto images = batch.data;
        auto labels = batch.target;

        auto outputs = model.forward(images);
        auto predictions = outputs.argmax(1);

        correct += predictions.eq(labels).sum().item<int64_t>();
        total += labels.size(0);
    }

    std::cout << "Accuracy: " << (100.0 * correct / total) << "%" << std::endl;

    return 0;
}