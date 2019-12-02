#include "simple_convnet.h"
#include "utils.h"
#include "mnist.h"

#include <iostream>

int main(int argc, char *argv[])
{
    int batch_size = 1;
    int channel = 1;
    int height = 28;
    int width = 28;
    int filter_num = 1;
    int filter_h = 3;
    int filter_w = 3;
    int stride = 1;
    int pad = 0;
    int hidden_size = 100;
    int category_num = 10;
    double weight_init_std = 0.01;

    SimpleConvNet net(batch_size, channel, height, width, filter_num,
                      filter_h, filter_w, stride, pad, hidden_size,
                      category_num, weight_init_std);

    std::vector<std::vector<std::vector<unsigned char> > > train_images, test_images;
    std::vector<std::vector<unsigned char> > train_labels, test_labels;

    mnist::load_mnist(train_images, train_labels, test_images, test_labels, "data/");

    double ****input = util::alloc<double>(batch_size, channel, height, width);
    double **output = util::alloc<double>(batch_size, category_num);
    double **criterion = util::alloc<double>(batch_size, category_num);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                input[i][0][j][k] = train_images[i][j][k];
            }
        }
        for (int j = 0; j < category_num; ++j) {
            criterion[i][j] = train_labels[i][j];
        }
    }

    //net.predict(input, output);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < category_num; ++j) {
            //std::cout << output[i][j] << ", ";
        }
        //std::cout << std::endl;
    }

    //double loss = net.loss(input, criterion);
    //std::cout << loss << std::endl;

    net.gradient(input, criterion);

    util::free(input, batch_size, channel, height);
    util::free(output, batch_size);
    util::free(criterion, batch_size);

    return 0;
}