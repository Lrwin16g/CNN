#include "simple_convnet.h"
#include "utils.h"
#include "mnist.h"

#include <iostream>
#include <ctime>
#include <cstdlib>

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
                      category_num, weight_init_std);

    std::vector<std::vector<std::vector<unsigned char> > > train_images, test_images;
    std::vector<std::vector<unsigned char> > train_labels, test_labels;

    mnist::load_mnist(train_images, train_labels, test_images, test_labels, "data/");

    int train_size = train_images.size();
    int iter_per_epoch = std::max(train_size / batch_size, 1);
    int max_iter = max_epochs * iter_per_epoch;

    double ****input = util::alloc<double>(batch_size, channel, height, width);
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
        std::cout << loss << std::endl;

        net.gradient(input, criterion);

        net.update(lr);

        
    }

    util::free(input, batch_size, channel, height);
    util::free(criterion, batch_size);

    return 0;
}