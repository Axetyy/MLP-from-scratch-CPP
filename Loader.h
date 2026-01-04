#pragma once

#include <string>
#include <vector>
#include <utility>
#include <random>
#include "CTorch.h"

void load_mnist_csv(const std::string &filename, Tensor &X, std::vector<int> &y);

class Dataset
{
public:
    Tensor data;
    std::vector<int> labels;

    explicit Dataset(const std::string &filename);

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
    DataLoader(Dataset &dataset, int batch_size, bool shuffle = true);
    void reset();
    bool has_next() const;
    std::pair<Tensor, std::vector<int>> next();
};