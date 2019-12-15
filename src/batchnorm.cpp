#include "batchnorm.h"
#include "utils.h"

#include <cmath>

BatchNorm::BatchNorm(int batch_size, int channel, int height, int width, double momentum)
    : batch_size_(batch_size), input_size_(0), channel_(channel), height_(height), width_(width),
      momentum_(momentum), mean2d_(NULL), var2d_(NULL), std2d_(NULL), input_c2d_(NULL),
      input_n2d_(NULL), run_mean2d_(NULL), run_var2d_(NULL), mean_(NULL), var_(NULL), std_(NULL),
      input_c_(NULL), input_n_(NULL), run_mean_(NULL), run_var_(NULL), d_mean2d_(NULL),
      d_var2d_(NULL), d_std2d_(NULL), dx_c2d_(NULL), dx_n2d_(NULL), d_mean_(NULL), d_var_(NULL),
      d_std_(NULL), dx_c_(NULL), dx_n_(NULL)
{
    mean2d_ = util::alloc<double>(channel_, height_, width_);
    var2d_ = util::alloc<double>(channel_, height_, width_);
    std2d_ = util::alloc<double>(channel_, height_, width_);
    run_mean2d_ = util::alloc<double>(channel_, height_, width_);
    run_var2d_ = util::alloc<double>(channel_, height_, width_);

    input_c2d_ = util::alloc<double>(batch_size_, channel_, height_, width_);
    input_n2d_ = util::alloc<double>(batch_size_, channel_, height_, width_);

    d_mean2d_ = util::alloc<double>(channel_, height_, width_);
    d_var2d_ = util::alloc<double>(channel_, height_, width_);
    d_std2d_ = util::alloc<double>(channel_, height_, width_);

    dx_c2d_ = util::alloc<double>(batch_size_, channel_, height_, width_);
    dx_n2d_ = util::alloc<double>(batch_size_, channel_, height_, width_);
}

BatchNorm::BatchNorm(int batch_size, int input_size, double momentum)
    : batch_size_(batch_size), input_size_(input_size), channel_(0), height_(0), width_(0),
      momentum_(momentum), mean2d_(NULL), var2d_(NULL), std2d_(NULL), input_c2d_(NULL),
      input_n2d_(NULL), run_mean2d_(NULL), run_var2d_(NULL), mean_(NULL), var_(NULL), std_(NULL),
      input_c_(NULL), input_n_(NULL), run_mean_(NULL), run_var_(NULL), d_mean2d_(NULL),
      d_var2d_(NULL), d_std2d_(NULL), dx_c2d_(NULL), dx_n2d_(NULL), d_mean_(NULL), d_var_(NULL),
      d_std_(NULL), dx_c_(NULL), dx_n_(NULL)
{
    mean_ = util::alloc<double>(input_size_);
    var_ = util::alloc<double>(input_size_);
    std_ = util::alloc<double>(input_size_);
    run_mean_ = util::alloc<double>(input_size_);
    run_var_ = util::alloc<double>(input_size_);

    input_c_ = util::alloc<double>(batch_size_, input_size_);
    input_n_ = util::alloc<double>(batch_size_, input_size_);

    d_mean_ = util::alloc<double>(input_size_);
    d_var_ = util::alloc<double>(input_size_);
    d_std_ = util::alloc<double>(input_size_);

    dx_c_ = util::alloc<double>(batch_size_, input_size_);
    dx_n_ = util::alloc<double>(batch_size_, input_size_);
}

BatchNorm::~BatchNorm()
{
    util::free(mean2d_, channel_, height_);
    util::free(var2d_, channel_, height_);
    util::free(std2d_, channel_, height_);
    util::free(run_mean2d_, channel_, height_);
    util::free(run_var2d_, channel_, height_);

    util::free(input_c2d_, batch_size_, channel_, height_);
    util::free(input_n2d_, batch_size_, channel_, height_);

    util::free(d_mean2d_, channel_, height_);
    util::free(d_var2d_, channel_, height_);
    util::free(d_std2d_, channel_, height_);

    util::free(dx_c2d_, batch_size_, channel_, height_);
    util::free(dx_n2d_, batch_size_, channel_, height_);

    util::free(mean_);
    util::free(var_);
    util::free(std_);
    util::free(run_mean_);
    util::free(run_var_);

    util::free(input_c_, batch_size_);
    util::free(input_n_, batch_size_);

    util::free(d_mean_);
    util::free(d_var_);
    util::free(d_std_);

    util::free(dx_c_, batch_size_);
    util::free(dx_n_, batch_size_);
}

void BatchNorm::forward(double const * const * const * const *input,
                        double const * const * const *gamma,
                        double const * const * const *beta,
                        double ****output, bool train_flg)
{
    if (train_flg)
    {
        for (int i = 0; i < channel_; ++i) {
            for (int j = 0; j < height_; ++j) {
                for (int k = 0; k < width_; ++k) {
                    mean2d_[i][j][k] = 0.0;
                    var2d_[i][j][k] = 0.0;
                }
            }
        }

        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < channel_; ++j) {
                for (int k = 0; k < height_; ++k) {
                    for (int l = 0; l < width_; ++l) {
                        mean2d_[j][k][l] += input[i][j][k][l] / static_cast<double>(batch_size_);
                    }
                }
            }
        }

        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < channel_; ++j) {
                for (int k = 0; k < height_; ++k) {
                    for (int l = 0; l < width_; ++l) {
                        input_c2d_[i][j][k][l] = input[i][j][k][l] - mean2d_[j][k][l];
                        var2d_[j][k][l] += pow(input_c2d_[i][j][k][l], 2.0)
                                           / static_cast<double>(batch_size_);
                    }
                }
            }
        }

        for (int i = 0; i < channel_; ++i) {
            for (int j = 0; j < height_; ++j) {
                for (int k = 0; k < width_; ++k) {
                    std2d_[i][j][k] = sqrt(var2d_[i][j][k] + 10e-7);
                }
            }
        }

        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < channel_; ++j) {
                for (int k = 0; k < height_; ++k) {
                    for (int l = 0; l < width_; ++l) {
                        input_n2d_[i][j][k][l] = input_c2d_[i][j][k][l] / std2d_[j][k][l];
                    }
                }
            }
        }

        for (int i = 0; i < channel_; ++i) {
            for (int j = 0; j < height_; ++j) {
                for (int k = 0; k < width_; ++k) {
                    run_mean2d_[i][j][k] = momentum_ * run_mean2d_[i][j][k] + (1.0 - momentum_)
                                           * mean2d_[i][j][k];
                    run_var2d_[i][j][k] = momentum_ * run_var2d_[i][j][k] + (1.0 - momentum_)
                                          * var2d_[i][j][k];
                }
            }
        }
    } else
    {
        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < channel_; ++j) {
                for (int k = 0; k < height_; ++k) {
                    for (int l = 0; l < width_; ++l) {
                        input_c2d_[i][j][k][l] = input[i][j][k][l] - run_mean2d_[j][k][l];
                        input_n2d_[i][j][k][l] = input_c2d_[i][j][k][l] / sqrt(run_var2d_[j][k][l]
                                                 + 10e-7);
                    }
                }
            }
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    output[i][j][k][l] = gamma[j][k][l] * input_n2d_[i][j][k][l] + beta[j][k][l];
                }
            }
        }
    }
}

void BatchNorm::forward(double const * const *input, const double *gamma,
                        const double *beta, double **output, bool train_flg)
{
    if (train_flg)
    {
        for (int i = 0; i < input_size_; ++i) {
            mean_[i] = 0.0;
            var_[i] = 0.0;
        }

        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                mean_[j] += input[i][j] / static_cast<double>(batch_size_);
            }
        }

        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                input_c_[i][j] = input[i][j] - mean_[j];
                var_[j] += pow(input_c_[i][j], 2.0) / static_cast<double>(batch_size_);
            }
        }

        for (int i = 0; i < input_size_; ++i) {
            std_[i] = sqrt(var_[i] + 10e-7);
        }

        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                input_n_[i][j] = input_c_[i][j] / std_[j];
            }
        }

        for (int i = 0; i < input_size_; ++i) {
            run_mean_[i] = momentum_ * run_mean_[i] + (1.0 - momentum_) * mean_[i];
            run_var_[i] = momentum_ * run_var_[i] + (1.0 - momentum_) * var_[i];
        }
    } else
    {
        for (int i = 0; i < batch_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                input_c_[i][j] = input[i][j] - run_mean_[j];
                input_n_[i][j] = input_c_[i][j] / sqrt(run_var_[j] + 10e-7);
            }
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            output[i][j] = gamma[j] * input_n_[i][j] + beta[j];
        }
    }
}

void BatchNorm::backward(double const * const * const * const *dout,
                         double const * const * const *gamma,
                         double ***d_gamma, double ***d_beta, double ****dx)
{
    for (int i = 0; i < channel_; ++i) {
        for (int j = 0; j < height_; ++j) {
            for (int k = 0; k < width_; ++k) {
                d_beta[i][j][k] = 0.0;
                d_gamma[i][j][k] = 0.0;
                d_std2d_[i][j][k] = 0.0;
                d_mean2d_[i][j][k] = 0.0;
            }
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    d_beta[j][k][l] += dout[i][j][k][l];
                    d_gamma[j][k][l] += input_n2d_[i][j][k][l] * dout[i][j][k][l];
                }
            }
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    dx_n2d_[i][j][k][l] = gamma[j][k][l] * dout[i][j][k][l];
                    dx_c2d_[i][j][k][l] = dx_n2d_[i][j][k][l] / std2d_[j][k][l];
                    d_std2d_[j][k][l] -= (dx_n2d_[i][j][k][l] * input_c2d_[i][j][k][l])
                                         / (std2d_[j][k][l] * std2d_[j][k][l]);
                }
            }
        }
    }

    for (int i = 0; i < channel_; ++i) {
        for (int j = 0; j < height_; ++j) {
            for (int k = 0; k < width_; ++k) {
                d_var2d_[i][j][k] = 0.5 * d_std2d_[i][j][k] / std2d_[i][j][k];
            }
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    dx_c2d_[i][j][k][l] += (2.0 / static_cast<double>(batch_size_))
                                           * input_c2d_[i][j][k][l] * d_var2d_[j][k][l];
                    d_mean2d_[j][k][l] += dx_c2d_[i][j][k][l] / static_cast<double>(batch_size_);
                }
            }
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    dx[i][j][k][l] = dx_c2d_[i][j][k][l] - d_mean2d_[j][k][l];
                }
            }
        }
    }
}

void BatchNorm::backward(double const * const *dout, const double *gamma, double *d_gamma,
                         double *d_beta, double **dx)
{
    for (int i = 0; i < input_size_; ++i) {
        d_beta[i] = 0.0;
        d_gamma[i] = 0.0;
        d_std_[i] = 0.0;
        d_mean_[i] = 0.0;
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            d_beta[j] += dout[i][j];
            d_gamma[j] += input_n_[i][j] * dout[i][j];
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            dx_n_[i][j] = gamma[j] * dout[i][j];
            dx_c_[i][j] = dx_n_[i][j] / std_[j];
            d_std_[j] -= (dx_n_[i][j] * input_c_[i][j]) / (std_[j] * std_[j]);
        }
    }

    for (int i = 0; i < input_size_; ++i) {
        d_var_[i] = 0.5 * d_std_[i] / std_[i];
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            dx_c_[i][j] += (2.0 / static_cast<double>(batch_size_)) * input_c_[i][j] * d_var_[j];
            d_mean_[j] += dx_c_[i][j] / static_cast<double>(batch_size_);
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            dx[i][j] = dx_c_[i][j] - d_mean_[j];
        }
    }
}