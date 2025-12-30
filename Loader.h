#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "Tensor.h"

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