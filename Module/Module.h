#pragma once
#include "CTorch.h"

float accuracy_score(const Tensor &probs, const std::vector<int> &labels);

class Linear
{
public:
    Tensor Weights;
    std::vector<float> bias;
    Tensor X_cache;
    Tensor dW;
    std::vector<float> db;

    explicit Linear(int in_features, int out_features);
    Tensor forward(const Tensor &X);
    Tensor backward(const Tensor &dY);
};