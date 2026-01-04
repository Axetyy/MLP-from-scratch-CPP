#include "Loader.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <numeric>

void load_mnist_csv(const std::string &filename, Tensor &X, std::vector<int> &y)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Failed to open " + filename);

    std::string line;
    std::vector<float> buffer;
    buffer.reserve(60000 * 784);
    y.reserve(60000);

    std::getline(file, line); // header
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;

        std::getline(ss, cell, ',');
        y.push_back(std::stoi(cell));

        for (int i = 0; i < 784; ++i)
        {
            std::getline(ss, cell, ',');
            buffer.push_back(std::stof(cell) / 255.0f);
        }
    }

    X = Tensor(y.size(), 784);
    X.data = std::move(buffer);
}

Dataset::Dataset(const std::string &filename)
{
    load_mnist_csv(filename, data, labels);
}

DataLoader::DataLoader(Dataset &dataset, int batch_size, bool shuffle)
    : dataset(dataset), batch_size(batch_size), shuffle(shuffle),
      current_idx(0), rng(std::random_device{}())
{
    indices.resize(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle)
        std::shuffle(indices.begin(), indices.end(), rng);
}

void DataLoader::reset()
{
    current_idx = 0;
    if (shuffle)
        std::shuffle(indices.begin(), indices.end(), rng);
}

bool DataLoader::has_next() const
{
    return current_idx < indices.size();
}

std::pair<Tensor, std::vector<int>> DataLoader::next()
{
    int bsize = std::min(batch_size, int(indices.size() - current_idx));
    Tensor X_batch(bsize, dataset.num_features());
    std::vector<int> y_batch(bsize);

    const int cols = dataset.num_features();

    for (int i = 0; i < bsize; ++i)
    {
        int idx = indices[current_idx++];
        y_batch[i] = dataset.labels[idx];
        std::memcpy(
            &X_batch.data[i * cols],
            &dataset.data.data[idx * cols],
            cols * sizeof(float));
    }
    return {X_batch, y_batch};
}
