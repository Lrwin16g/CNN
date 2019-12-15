#include "deep_convnet.h"
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
    int input_dim[] = {1, 28, 28};
    const int hidden_size = 50;
    const int category_num = 10;
    const int max_epochs = 20;

    DeepConvNet net(batch_size, input_dim, hidden_size, category_num);

    std::vector<std::vector<std::vector<unsigned char> > > train_images_v, test_images_v;
    std::vector<std::vector<unsigned char> > train_labels_v, test_labels_v;

    mnist::load_mnist(train_images_v, train_labels_v, test_images_v, test_labels_v, "data/");

    int train_size = train_images_v.size();
    int test_size = test_images_v.size();

    double ***train_images = util::alloc<double>(train_size, input_dim[1], input_dim[2]);
    unsigned char**train_labels = util::alloc<unsigned char>(train_size, category_num);
    for (int i = 0; i < train_images_v.size(); ++i) {
        for (int j = 0; j < train_images_v[i].size(); ++j) {
            for (int k = 0; k < train_images_v[i][j].size(); ++k) {
                train_images[i][j][k] = train_images_v[i][j][k] / 255.0;
            }
        }
        for (int j = 0; j < train_labels_v[i].size(); ++j) {
            train_labels[i][j] = train_labels_v[i][j];
        }
    }

    double ***test_images = util::alloc<double>(test_size, input_dim[1], input_dim[2]);
    unsigned char**test_labels = util::alloc<unsigned char>(test_size, category_num);
    for (int i = 0; i < test_images_v.size(); ++i) {
        for (int j = 0; j < test_images_v[i].size(); ++j) {
            for (int k = 0; k < test_images_v[i][j].size(); ++k) {
                test_images[i][j][k] = test_images_v[i][j][k] / 255.0;
            }
        }
        for (int j = 0; j < test_labels_v[i].size(); ++j) {
            test_labels[i][j] = test_labels_v[i][j];
        }
    }

    int iter_per_epoch = std::max(train_size / batch_size, 1);
    int max_iter = max_epochs * iter_per_epoch;
    int evaluate_sample_num_per_epoch = 1000;

    double ****input = util::alloc<double>(batch_size, input_dim[0], input_dim[1], input_dim[2]);
    double **output = util::alloc<double>(batch_size, category_num);
    double **criterion = util::alloc<double>(batch_size, category_num);

    for (int i = 0; i < max_iter; ++i)
    {
        // バッチサイズ分ランダムチョイス
        for (int j = 0; j < batch_size; ++j) {
            int idx = rand() % train_size;
            for (int k = 0; k < input_dim[1]; ++k) {
                for (int l = 0; l < input_dim[2]; ++l) {
                    input[j][0][k][l] = train_images[idx][k][l];
                }
            }
            for (int k = 0; k < category_num; ++k) {
                criterion[j][k] = train_labels[idx][k];
            }
        }

        std::cout << "forward" << std::endl;

        double loss = net.loss(input, criterion);
        std::cout << i << ": " << loss << std::endl;

        std::cout << "backward" << std::endl;

        net.gradient(criterion);

        std::cout << "update" << std::endl;

        net.update();

        if (i % iter_per_epoch == 0) {
            net.setTrainFlg(false);
            double train_acc = 0.0;
            double test_acc = 0.0;
            for (int j = 0; j < evaluate_sample_num_per_epoch / batch_size; ++j) {
                for (int k = 0; k < batch_size; ++k) {
                    for (int l = 0; l < input_dim[1]; ++l) {
                        for (int m = 0; m < input_dim[2]; ++m) {
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
                    for (int l = 0; l < input_dim[1]; ++l) {
                        for (int m = 0; m < input_dim[2]; ++m) {
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
            net.setTrainFlg(true);
        }
    }

    util::free(train_images, train_size, input_dim[1]);
    util::free(train_labels, train_size);
    util::free(test_images, test_size, input_dim[1]);
    util::free(test_labels, test_size);
    util::free(input, batch_size, input_dim[0], input_dim[1]);
    util::free(output, batch_size);
    util::free(criterion, batch_size);

    return 0;
}