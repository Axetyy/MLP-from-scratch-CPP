#pragma once
#include "Loader.h"
#include "CTorch.h"
#include "Module.h"
#include <vector>
#include <memory>

class Model
{
public:
    std::vector<std::shared_ptr<Linear>> hidden_layers;
    float lr = 1e-4;
    Model(float learning_rate);

    Tensor operator()(const Tensor &X) const;

    void add_layer(int in_features, int out_features);

    void backward(const Tensor &logits, const std::vector<int> &y);
    void step();
    void zero_grad();
    void train(DataLoader &loader, int epochs = 10, int verbose = 1);
    float predict(DataLoader &loader);

private:
    void relu_backward(Tensor &grad, const Tensor &inputs);
    // Softmax + Cross-Entropy -> gradient
    Tensor softmax_crossentropy_backward(const Tensor &logits, const std::vector<int> &y);
};