#pragma once
#include "Tensor.h"

class Linear
{
public:
    Tensor Weights;
    std::vector<float> bias;
    Tensor X_cache;
    Tensor dW;
    std::vector<float> db;

    Linear(int in_features, int out_features) : Weights(in_features, out_features), bias(out_features, 0.0f), dW(in_features, out_features), db(out_features, 0.0f)
    {
        float scale = std::sqrt(2.0f / in_features);
        for (float &v : Weights.data)
        {
            v = randf(-scale, scale);
        }
    }
    Tensor forward(const Tensor &X)
    {
        X_cache = X;
        Tensor Y = Tensor::matmul(X, this->Weights);
        Tensor::add_bias(Y, this->bias);
        return Y;
    }
    Tensor backward(const Tensor &dY)
    {
        dW = Tensor::matmul(Tensor::transpose(X_cache), dY);
        std::fill(db.begin(), db.end(), 0.0f);
        for (int i = 0; i < dY.rows; i++)
        {
            for (int j = 0; j < dY.cols; j++)
            {
                db[j] += dY(i, j);
            }
        }
        Tensor dX = Tensor::matmul(dY, Tensor::transpose(Weights));
        return dX;
    }
};
