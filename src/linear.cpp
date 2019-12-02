#include "linear.h"
#include "utils.h"

Linear::Linear(int batch_size, int input_size, int output_size,
               int channel, int height, int width)
    : batch_size_(batch_size), input_size_(input_size), output_size_(output_size),
      channel_(channel), height_(height), width_(width),
      input_T_(NULL), weight_T_(NULL), d_weight_(NULL), d_bias_(NULL),
      input_col_(NULL), dx_col_(NULL)
{
    input_T_ = util::alloc<double>(input_size_, batch_size_);
    weight_T_ = util::alloc<double>(output_size_, input_size_);
    d_weight_ = util::alloc<double>(input_size_, output_size_);
    d_bias_ = util::alloc<double>(output_size_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        input_col_ = util::alloc<double>(batch_size_, channel_ * height_ * width_);
        dx_col_ = util::alloc<double>(batch_size_, channel_ * height_ * width_);
    }
}

Linear::~Linear()
{
    util::free(input_T_, input_size_);
    util::free(weight_T_, output_size_);
    util::free(d_weight_, input_size_);
    util::free(d_bias_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        util::free(input_col_, batch_size_);
        util::free(dx_col_, batch_size_);
    }
}

void Linear::forward(double const * const * const * const *input,
                     double const * const *weight, const double *bias,
                     double **output)
{
    // テンソルを行列に展開
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    int idx_x = c * (height_ * width_) + y * width_ + x;
                    input_col_[n][idx_x] = input[n][c][y][x];
                }
            }
        }
    }

    forward(input_col_, weight, bias, output);
}

void Linear::forward(double const * const *input, double const * const *weight,
                     const double *bias, double **output)
{
    // 入力をコピー
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            input_T_[j][i] = input[i][j];
        }
    }

    // 重みをコピー
    for (int i = 0; i < input_size_; ++i) {
        for (int j = 0; j < output_size_; ++j) {
            weight_T_[j][i] = weight[i][j];
        }
    }

    // 出力の計算
    util::dot(input, weight, output, batch_size_, input_size_,
              input_size_, output_size_);

    // バイアスの加算  
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < output_size_; ++j) {
            output[i][j] += bias[j];
        }
    }
}

void Linear::backward(double const * const *dout, double ****dx)
{
    backward(dout, dx_col_);

    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    int idx_x = c * (height_ * width_) + y * width_ + x;
                    dx[n][c][y][x] = dx_col_[n][idx_x];
                }
            }
        }
    }
}

void Linear::backward(double const * const *dout, double **dx)
{
    // 誤差逆伝播
    util::dot(dout, weight_T_, dx, batch_size_, output_size_,
              output_size_, input_size_);

    // 重みの勾配の計算
    util::dot(input_T_, dout, d_weight_, input_size_, batch_size_,
              batch_size_, output_size_);

    // バイアスの勾配の計算
    for (int i = 0; i < output_size_; ++i) {
        d_bias_[i] = 0.0;
        for (int j = 0; j < batch_size_; ++j) {
            d_bias_[i] += dout[j][i];
        }
    }
}