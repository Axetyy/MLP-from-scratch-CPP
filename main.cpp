#include <iostream>
#include "Model.h"
#include <algorithm>

int main()
{
    int batch_size = 64;
    int batch_count = 0;
    int verbose = 1;
    int epochs = 15;

    Dataset train_dataset("mnist_train.csv");
    DataLoader train_loader(train_dataset, batch_size);
    Dataset test_dataset("mnist_test.csv");
    DataLoader test_loader(test_dataset, 64, false);

    Model model(0.01f);

    model.add_layer(784, 128);
    model.add_layer(128, 10);

    model.train(train_loader, epochs, verbose);
    model.predict(test_loader);
    return 0;
}