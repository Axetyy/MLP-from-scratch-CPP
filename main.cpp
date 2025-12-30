#include <iostream>
#include "Tensor.h"
#include "Loader.h"
#include "Module.h"
#include "Optim.h"
#include <algorithm>

int main()
{
    Tensor X_train;
    std::vector<int> y_train;
    load_mnist_csv("mnist_train_small.csv", X_train, y_train);
    std::cout << " Loaded set into memory " << X_train.rows << "x" << X_train.cols << "\n";

    Linear fc1(784, 128);
    Linear fc2(128, 10);
    SGD optimizer(0.01f);

    int batch_size = 64;
    int num_batches = X_train.rows / batch_size;

    std::vector<int> indices(X_train.rows);
    for (int i = 0; i < X_train.rows; i++)
        indices[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(indices.begin(), indices.end(), g);
    for (int b = 0; b < num_batches; b++)
    {

        Tensor X_batch(batch_size, X_train.cols);
        std::vector<int> y_batch(batch_size);
        for (int i = 0; i < batch_size; i++)
        {
            int idx = indices[b * batch_size + i];
            y_batch[i] = y_train[idx];
            for (int j = 0; j < X_train.cols; j++)
                X_batch(i, j) = X_train(idx, j);
        }

        optimizer.zero_grad(fc1);
        optimizer.zero_grad(fc2);

        Tensor h = fc1.forward(X_batch);
        Tensor::ReLU(h);

        Tensor logits = fc2.forward(h);

        Tensor dlogits = softmax_crossentropy_backward(logits, y_batch);

        Tensor dh = fc2.backward(dlogits);
        relu_backward(dh, h);
        Tensor dX = fc1.backward(dh);

        // Update weights
        optimizer.step(fc1);
        optimizer.step(fc2);

        Tensor probs = logits;
        Tensor::SoftMax(probs);
        float loss = Tensor::CrossEntropyLoss(probs, y_batch);

        int correct = 0;
        for (int i = 0; i < probs.rows; i++)
        {
            int pred = 0;
            float maxv = probs(i, 0);
            for (int j = 1; j < probs.cols; j++)
            {
                if (probs(i, j) > maxv)
                {
                    maxv = probs(i, j);
                    pred = j;
                }
            }
            if (pred == y_batch[i])
                correct++;
        }
        float acc = static_cast<float>(correct) / probs.rows;

        if (b % 200 == 0)
        {
            std::cout << "Batch " << b << "/" << num_batches << " Loss: " << loss << " Acc: " << acc << "\n";
            std::cout << "Loss: " << loss << " Acc: " << acc << "\n";
        }
    }

    return 0;
}