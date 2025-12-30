#include "Tensor.h"
#include "Module.h"

class SGD
{
public:
    float lr = 1e-4;
    SGD(float learning_rate) : lr(learning_rate) {};

    void step(Linear &layer)
    {
        for (int i = 0; i < layer.Weights.data.size(); i++)
        {
            layer.Weights.data[i] -= lr * layer.dW.data[i];
        }
        for (int i = 0; i < layer.bias.size(); i++)
        {
            layer.bias[i] -= lr * layer.db[i];
        }
    }
    void zero_grad(Linear &layer)
    {
        std::fill(layer.dW.data.begin(), layer.dW.data.end(), 0.0f);
        std::fill(layer.db.begin(), layer.db.end(), 0.0f);
    }
};