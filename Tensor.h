#pragma once
#include <vector>
#include <random>
#include <cmath>
#define eps 1e-9

class Tensor
{
public:
    int rows, cols;

    std::vector<float> data; /// row-major

    Tensor() : rows(0), cols(0) {}
    Tensor(int rows, int cols) : rows(rows), cols(cols), data(rows * cols) {};

    inline float &operator()(int row, int col)
    {
        return data[row * cols + col];
    }

    inline const float &operator()(int row, int col) const
    {
        return data[row * cols + col];
    }

    static Tensor matmul(const Tensor &A, const Tensor &B)
    {
        if (A.cols != B.rows)
        {
            throw std::runtime_error("Can't perform matrix multiplication on matrixes whose shapes aren't compatible.");
        }
        Tensor C(A.rows, B.cols);
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                float a = A(i, j);
                for (int k = 0; k < B.cols; k++)
                {
                    C(i, k) += a * B(j, k);
                }
            }
        }
        return C;
    }
    static Tensor transpose(const Tensor &A)
    {
        Tensor T(A.cols, A.rows);
        for (int i = 0; i < A.rows; i++)
            for (int j = 0; j < A.cols; j++)
                T(j, i) = A(i, j);
        return T;
    }

    static void add_bias(Tensor &X, const std::vector<float> &b)
    {
        for (int i = 0; i < X.rows; i++)
        {
            for (int j = 0; j < X.cols; j++)
            {
                X(i, j) += b[j];
            }
        }
    }
    static void ReLU(Tensor &X)
    {
        for (float &v : X.data)
        {
            v = std::max(0.0f, v);
        }
    }
    static void SoftMax(Tensor &X)
    {
        for (int i = 0; i < X.rows; i++)
        {
            float max_val = -1e9;
            for (int j = 0; j < X.cols; j++)
            {
                max_val = std::max(max_val, X(i, j));
            }
            float sum = 0.0f;
            for (int j = 0; j < X.cols; j++)
            {
                X(i, j) = std::exp(X(i, j) - max_val);
                sum += X(i, j);
            }
            for (int j = 0; j < X.cols; j++)
            {
                X(i, j) /= sum;
            }
        }
    }
    static float CrossEntropyLoss(Tensor &probabilites, const std::vector<int> &y)
    {
        float loss = 0.0f;
        for (int i = 0; i < probabilites.rows; i++)
        {
            loss -= std::log(probabilites(i, y[i]) + eps);
        }
        return loss / probabilites.rows;
    }
};
Tensor softmax_crossentropy_backward(const Tensor &logits, const std::vector<int> y)
{
    Tensor grad(logits.rows, logits.cols);

    for (int i = 0; i < logits.rows; i++)
    {
        float max_val = -1e9;
        for (int j = 0; j < logits.cols; j++)
        {
            max_val = std::max(max_val, logits(i, j));
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < logits.cols; j++)
        {
            sum_exp += std::exp(logits(i, j) - max_val);
        }
        for (int j = 0; j < logits.cols; j++)
        {
            float p = std::exp(logits(i, j) - max_val) / sum_exp;
            grad(i, j) = p;
        }
        grad(i, y[i]) -= 1.0f;
    }
    float inv_bs = 1.0 / logits.rows; // average
    for (float &v : grad.data)
    {
        v *= inv_bs;
    }
    return grad;
}
float randf(float a, float b)
{
    static std::mt19937 gen(67);
    std::uniform_real_distribution<float> dist(a, b);
    return dist(gen);
}

void relu_backward(Tensor &grad, const Tensor &inputs)
{
    for (int i = 0; i < grad.data.size(); i++)
    {
        grad.data[i] = (inputs.data[i] > 0.0f) ? grad.data[i] : 0.0f;
    }
}