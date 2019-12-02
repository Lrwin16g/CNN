#include "softmax.h"
#include "utils.h"

#include <limits>
#include <cmath>

SoftmaxWithLoss::SoftmaxWithLoss(int batch_size, int category_num)
    : batch_size_(batch_size), category_num_(category_num),
      output_(NULL)
{
    output_ = util::alloc<double>(batch_size_, category_num_);
}

SoftmaxWithLoss::~SoftmaxWithLoss()
{
    util::free(output_, batch_size_);
}

double SoftmaxWithLoss::forward(double const * const *input, double const * const *criterion)
{
    double cross_entropy_error = 0.0;

    for (int i = 0; i < batch_size_; ++i) {

        double max_val = std::numeric_limits<double>::min();
        for (int j = 0; j < category_num_; ++j) {
            if (max_val < input[i][j]) {
                max_val = input[i][j];
            }
        }

        double sum = 0.0;
        for (int j = 0; j < category_num_; ++j) {
            output_[i][j] = exp(input[i][j] - max_val);
            sum += output_[i][j];
        }

        for (int j = 0; j < category_num_; ++j) {
            output_[i][j] /= sum;
        }

        for (int j = 0; j < category_num_; ++j) {
            if (criterion[i][j] == 1.0) {
                cross_entropy_error += log(output_[i][j]);
            }
        }
    }

    return -1.0 * cross_entropy_error / batch_size_;
}

void SoftmaxWithLoss::backward(double const * const *criterion, double **dx)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < category_num_; ++j) {
            dx[i][j] = (output_[i][j] - criterion[i][j]) / batch_size_;
        }
    }
}