#include "simple_convnet.h"
#include "utils.h"
#include "mnist.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <limits>

int main(int argc, char *argv[])
{
    srand(time(NULL));

    const int batch_size = 100;
    const int channel = 1;
    const int height = 28;
    const int width = 28;
    const int filter_num = 30;
    const int filter_h = 5;
    const int filter_w = 5;
    const int stride = 1;
    const int pad = 0;
    const int hidden_size = 100;
    const int category_num = 10;
    const double weight_init_std = 0.01;
    const int max_epochs = 20;
    const double lr = 0.001;

    SimpleConvNet net(batch_size, channel, height, width, filter_num,
                      filter_h, filter_w, stride, pad, hidden_size,
                      category_num, weight_init_std, lr);

    std::vector<std::vector<std::vector<unsigned char> > > train_images, test_images;
    std::vector<std::vector<unsigned char> > train_labels, test_labels;

    mnist::load_mnist(train_images, train_labels, test_images, test_labels, "data/");

    int train_size = train_images.size();
    int iter_per_epoch = std::max(train_size / batch_size, 1);
    int max_iter = max_epochs * iter_per_epoch;
    int evaluate_sample_num_per_epoch = 1000;

    double ****input = util::alloc<double>(batch_size, channel, height, width);
    double **output = util::alloc<double>(batch_size, category_num);
    double **criterion = util::alloc<double>(batch_size, category_num);

    for (int i = 0; i < max_iter; ++i)
    {
        // バッチサイズ分ランダムチョイス
        for (int j = 0; j < batch_size; ++j) {
            int idx = rand() % train_size;
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    input[j][0][k][l] = train_images[idx][k][l];
                }
            }
            for (int k = 0; k < category_num; ++k) {
                criterion[j][k] = train_labels[idx][k];
            }
        }

        double loss = net.loss(input, criterion);
        std::cout << i << ": " << loss << std::endl;

        net.gradient(input, criterion);

        net.update();

        if (i % iter_per_epoch == 0) {
            double train_acc = 0.0;
            double test_acc = 0.0;
            for (int j = 0; j < evaluate_sample_num_per_epoch / batch_size; ++j) {
                for (int k = 0; k < batch_size; ++k) {
                    for (int l = 0; l < height; ++l) {
                        for (int m = 0; m < width; ++m) {
                            input[k][0][l][m] = train_images[k][l][m];
                        }
                    }
                    for (int l = 0; l < category_num; ++l) {
                        criterion[k][l] = train_labels[k][l];
                    }
                }

                net.predict(input, output);
                for (int k = 0; k < batch_size; ++k) {
                    int idx = -1;
                    double max_val = std::numeric_limits<double>::min();
                    for (int l = 0; l < category_num; ++l) {
                        if (max_val < output[k][l]) {
                            max_val = output[k][l];
                            idx = l;
                        }
                    }
                    if (criterion[k][idx] == 1.0) {
                        train_acc += 1.0;
                    }
                }

                for (int k = 0; k < batch_size; ++k) {
                    for (int l = 0; l < height; ++l) {
                        for (int m = 0; m < width; ++m) {
                            input[k][0][l][m] = test_images[k][l][m];
                        }
                    }
                    for (int l = 0; l < category_num; ++l) {
                        criterion[k][l] = test_labels[k][l];
                    }
                }

                net.predict(input, output);
                for (int k = 0; k < batch_size; ++k) {
                    int idx = -1;
                    double max_val = std::numeric_limits<double>::min();
                    for (int l = 0; l < category_num; ++l) {
                        if (max_val < output[k][l]) {
                            max_val = output[k][l];
                            idx = l;
                        }
                    }
                    if (criterion[k][idx] == 1.0) {
                        test_acc += 1.0;
                    }
                }
            }
            train_acc /= evaluate_sample_num_per_epoch;
            test_acc /= evaluate_sample_num_per_epoch;
            std::cout << "train_acc: " << train_acc << std::endl;
            std::cout << "test_acc: " << test_acc << std::endl;
        }
    }

    util::free(input, batch_size, channel, height);
    util::free(output, batch_size);
    util::free(criterion, batch_size);

    return 0;
}