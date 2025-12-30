#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "CTorch.h"

void load_mnist_csv(const std::string &filename, Tensor &X, std::vector<int> &y)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open " + filename);
    }
    std::string line;
    std::vector<float> buffer;
    y.clear();
    int num_cols = 785;
    int num_rows = 0;

    std::getline(file, line); // Skip header row
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        y.push_back(std::stoi(cell));
        for (int i = 0; i < 784; i++)
        {
            std::getline(ss, cell, ',');
            buffer.push_back(std::stof(cell) / 255.0f);
        }
        num_rows++;
    }
    X = Tensor(num_rows, 784);
    X.data = buffer;
}

class Dataset
{
public:
    Tensor data;
    std::vector<int> labels;

    Dataset() = default;

    Dataset(const std::string &filename)
    {
        load_mnist_csv(filename, data, labels);
    }

    int size() const { return data.rows; }
    int num_features() const { return data.cols; }
};
class DataLoader
{
private:
    Dataset &dataset;
    int batch_size;
    bool shuffle;
    std::vector<int> indices;
    size_t current_idx;
    std::mt19937 rng;

public:
    DataLoader(Dataset &dataset, int batch_size, bool shuffle = true)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_idx(0), rng(std::random_device{}())
    {
        indices.resize(dataset.size());
        for (int i = 0; i < dataset.size(); i++)
            indices[i] = i;

        if (shuffle)
            std::shuffle(indices.begin(), indices.end(), rng);
    }

    void reset()
    {
        current_idx = 0;
        if (shuffle)
            std::shuffle(indices.begin(), indices.end(), rng);
    }

    bool has_next() const
    {
        return current_idx < indices.size();
    }

    std::pair<Tensor, std::vector<int>> next()
    {
        int bsize = std::min(batch_size, static_cast<int>(indices.size() - current_idx));
        Tensor X_batch(bsize, dataset.num_features());
        std::vector<int> y_batch(bsize);

        for (int i = 0; i < bsize; i++)
        {
            int idx = indices[current_idx++];
            y_batch[i] = dataset.labels[idx];
            for (int j = 0; j < dataset.num_features(); j++)
                X_batch(i, j) = dataset.data(idx, j);
        }
        return {X_batch, y_batch};
    }
};