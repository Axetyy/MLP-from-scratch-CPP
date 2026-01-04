#include "Model.h"
#include <iostream>

Model::Model(float learning_rate) : lr(learning_rate) {};
void Model::add_layer(int in_features, int out_features)
{
    auto layer = std::make_shared<Linear>(in_features, out_features);
    hidden_layers.push_back(layer);
}
Tensor Model::operator()(const Tensor &X) const
{
    Tensor out = X;
    for (size_t i = 0; i < hidden_layers.size(); i++)
    {
        out = hidden_layers[i]->forward(out);
        if (i != hidden_layers.size() - 1)
            Tensor::ReLU(out);
    }
    return out;
}
void Model::backward(const Tensor &logits, const std::vector<int> &y)
{
    Tensor grad = softmax_crossentropy_backward(logits, y);

    for (int i = hidden_layers.size() - 1; i >= 0; i--)
    {
        grad = hidden_layers[i]->backward(grad);
        if (i != 0)
            relu_backward(grad, hidden_layers[i - 1]->X_cache);
    }
}
void Model::relu_backward(Tensor &grad, const Tensor &inputs)
{
    for (int i = 0; i < grad.data.size(); i++)
    {
        grad.data[i] = (inputs.data[i] > 0.0f) ? grad.data[i] : 0.0f;
    }
}
void Model::step()
{
    for (auto &layer : this->hidden_layers)
    {

        for (int i = 0; i < layer->Weights.data.size(); i++)
            layer->Weights.data[i] -= lr * layer->dW.data[i];
        for (int i = 0; i < layer->bias.size(); i++)
            layer->bias[i] -= lr * layer->db[i];
    }
}
void Model::zero_grad()
{
    for (auto &layer : this->hidden_layers)
    {
        std::fill(layer->dW.data.begin(), layer->dW.data.end(), 0.0f);
        std::fill(layer->db.begin(), layer->db.end(), 0.0f);
    }
}
void Model::train(DataLoader &loader, int epochs = 10, int verbose = 1)
{
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        loader.reset();
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int epoch_samples = 0;
        int batch_count = 0;

        if (verbose)
            std::cout << "Epoch " << epoch << "/" << epochs << "\n";

        while (loader.has_next())
        {
            auto [X_batch, y_batch] = loader.next();
            zero_grad();
            Tensor logits = (*this)(X_batch);
            backward(logits, y_batch);
            step();

            Tensor probs = logits;
            Tensor::SoftMax(probs);
            float loss = Tensor::CrossEntropyLoss(probs, y_batch);
            epoch_loss += loss * X_batch.rows;

            std::vector<int> preds = Torch::argmax(probs, 1);
            for (size_t k = 0; k < preds.size(); k++)
                if (preds[k] == y_batch[k])
                    epoch_correct++;

            epoch_samples += X_batch.rows;
            batch_count++;
        }

        float epoch_acc = static_cast<float>(epoch_correct) / epoch_samples;
        std::cout << "Epoch " << epoch
                  << " | Loss: " << epoch_loss / epoch_samples
                  << " | Acc: " << epoch_acc << "\n";
    }
}
float Model::predict(DataLoader &loader)
{

    int total_correct = 0;
    int total_samples = 0;

    while (loader.has_next())
    {
        auto [X_batch, y_batch] = loader.next();

        Tensor logits = (*this)(X_batch);
        Tensor::SoftMax(logits);
        std::vector<int> preds = Torch::argmax(logits, 1);

        for (size_t i = 0; i < preds.size(); i++)
            if (preds[i] == y_batch[i])
                total_correct++;

        total_samples += preds.size();
    }

    float accuracy = static_cast<float>(total_correct) / total_samples;
    std::cout << "Final Test Accuracy: " << accuracy << "\n";
    return accuracy;
}

Tensor Model::softmax_crossentropy_backward(const Tensor &logits, const std::vector<int> &y)
{
    Tensor grad(logits.rows, logits.cols);

    if ((int)y.size() != logits.rows)
        throw std::runtime_error("Label vector size does not match logits rows");

    for (int i = 0; i < logits.rows; i++)
    {
        float max_val = -1e9f;
        for (int j = 0; j < logits.cols; j++)
            max_val = std::max(max_val, logits(i, j));

        float sum_exp = 0.0f;
        for (int j = 0; j < logits.cols; j++)
            sum_exp += std::exp(logits(i, j) - max_val);

        for (int j = 0; j < logits.cols; j++)
            grad(i, j) = std::exp(logits(i, j) - max_val) / sum_exp;

        int label = y[i];
        if (label < 0 || label >= logits.cols)
            throw std::runtime_error("Label out of range at index " + std::to_string(i));

        grad(i, label) -= 1.0f;
    }

    float inv_bs = 1.0f / logits.rows;
    for (float &v : grad.data)
        v *= inv_bs;

    return grad;
}