#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#define eps 1e-9

class Tensor
{
public:
    int rows, cols;
    std::vector<float> data;

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
Tensor softmax_crossentropy_backward(const Tensor &logits, const std::vector<int> &y)
{
    Tensor grad(logits.rows, logits.cols); // grad tensor

    if ((int)y.size() != logits.rows)
    {
        throw std::runtime_error("softmax_crossentropy_backward: label vector size (" + std::to_string(y.size()) + ") does not match logits.rows (" + std::to_string(logits.rows) + ")");
    }

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
        // softmax
        for (int j = 0; j < logits.cols; j++)
        {
            float probs = std::exp(logits(i, j) - max_val) / sum_exp;
            grad(i, j) = probs;
        }
        int label = y[i];
        if (label < 0 || label >= logits.cols)
        {
            throw std::runtime_error("softmax_crossentropy_backward: label out of range at index " + std::to_string(i) + ": " + std::to_string(label));
        }
        grad(i, label) -= 1.0f; /// cross entropy
    }
    float inv_bs = 1.0 / logits.rows; // average
    for (float &v : grad.data)
    {
        v *= inv_bs; // actualize grads
    }
    return grad;
}

void relu_backward(Tensor &grad, const Tensor &inputs)
{
    for (int i = 0; i < grad.data.size(); i++)
    {
        grad.data[i] = (inputs.data[i] > 0.0f) ? grad.data[i] : 0.0f;
    }
}

class Torch
{
public:
    static float randf(float a, float b)
    {
        static std::mt19937 gen(67);
        std::uniform_real_distribution<float> dist(a, b);
        return dist(gen);
    }
    static std::vector<int> argmax(const Tensor &X, int dim = 1)
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
};