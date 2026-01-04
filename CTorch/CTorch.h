#pragma once
#include <vector>
class Tensor
{
public:
    int rows{0}, cols{0};
    std::vector<float> data;

    Tensor();
    Tensor(int rows, int cols);

    float &operator()(int row, int col);
    const float &operator()(int row, int col) const;

    static Tensor matmul(const Tensor &A, const Tensor &B);
    static Tensor transpose(const Tensor &A);
    static void add_bias(Tensor &X, const std::vector<float> &b);
    static void ReLU(Tensor &X);
    static void SoftMax(Tensor &X);
    static float CrossEntropyLoss(const Tensor &probabilities,
                                  const std::vector<int> &y);
};

class Torch
{
public:
    static float randf(float a, float b);
    static std::vector<int> argmax(const Tensor &X, int dim = 1);
};