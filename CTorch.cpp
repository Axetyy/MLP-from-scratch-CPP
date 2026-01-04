#include "CTorch.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>

static constexpr float eps = 1e-9f;

Tensor::Tensor() = default;
Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols), data(rows * cols) {};
inline float &Tensor::operator()(int row, int col)
{
    return data[row * cols + col];
}
const float &Tensor::operator()(int row, int col) const
{
    return data[row * cols + col];
}

Tensor Tensor::matmul(const Tensor &A, const Tensor &B)
{
    if (A.cols != B.rows)
        throw std::runtime_error("Incompatible shapes for matmul");

    Tensor C(A.rows, B.cols);

    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
        {
            float a = A(i, k);
            for (int j = 0; j < B.cols; ++j)
                C(i, j) += a * B(k, j);
        }

    return C;
}

Tensor Tensor::transpose(const Tensor &A)
{
    Tensor T(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            T(j, i) = A(i, j);
    return T;
}

void Tensor::add_bias(Tensor &X, const std::vector<float> &b)
{
    for (int i = 0; i < X.rows; ++i)
        for (int j = 0; j < X.cols; ++j)
            X(i, j) += b[j];
}

void Tensor::ReLU(Tensor &X)
{
    for (float &v : X.data)
        v = std::max(0.0f, v);
}

void Tensor::SoftMax(Tensor &X)
{
    for (int i = 0; i < X.rows; ++i)
    {
        float max_val = -1e9f;
        for (int j = 0; j < X.cols; ++j)
            max_val = std::max(max_val, X(i, j));

        float sum = 0.0f;
        for (int j = 0; j < X.cols; ++j)
        {
            X(i, j) = std::exp(X(i, j) - max_val);
            sum += X(i, j);
        }

        for (int j = 0; j < X.cols; ++j)
            X(i, j) /= sum;
    }
}
float Tensor::CrossEntropyLoss(const Tensor &probs,
                               const std::vector<int> &y)
{
    float loss = 0.0f;
    for (int i = 0; i < probs.rows; ++i)
        loss -= std::log(probs(i, y[i]) + eps);

    return loss / probs.rows;
}

float Torch::randf(float a, float b)
{
    static std::mt19937 gen(67);
    std::uniform_real_distribution<float> dist(a, b);
    return dist(gen);
}
std::vector<int> Torch::argmax(const Tensor &X, int dim)
{
    std::vector<int> result;
    if (dim == 1)
    {
        for (int i = 0; i < X.rows; i++)
        {
            int idx = 0;
            float maxi = X(i, 0);
            for (int j = 1; j < X.cols; j++)
            {
                if (X(i, j) > maxi)
                {
                    maxi = X(i, j);
                    idx = j;
                }
            }
            result.push_back(idx);
        }
    }
    else if (dim == 0)
    {
        for (int j = 0; j < X.cols; j++)
        {
            int idx = 0;
            float maxv = X(0, j);
            for (int i = 1; i < X.rows; i++)
            {
                if (X(i, j) > maxv)
                {
                    maxv = X(i, j);
                    idx = i;
                }
            }
            result.push_back(idx);
        }
    }
    return result;
}