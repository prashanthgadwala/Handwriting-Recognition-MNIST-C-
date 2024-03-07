#include "neural_net.hpp"
#include <torch/torch.h>
#include <iostream>

int main() {
    const int64_t input_size = 784;
    const int64_t hidden_size = 500;
    const int64_t num_classes = 10;
    const int64_t num_epochs = 5;
    const int64_t batch_size = 100;
    const double learning_rate = 0.001;

    auto train_dataset = torch::data::datasets::MNIST("./mnist-data")
                            .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                            std::move(train_dataset), batch_size);

    NeuralNet model(input_size, hidden_size, num_classes);

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto& batch : *train_loader) {
            auto images = batch.data;
            auto labels = batch.target;

            auto outputs = model(images);
            auto loss = torch::nll_loss(outputs, labels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }
    
    model.save_model("mnist_model.pt");
    std::cout << "Training finished." << std::endl;

    return 0;
}