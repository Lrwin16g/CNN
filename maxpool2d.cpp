#include "maxpool2d.h"
#include "utils.h"

#include <limits>

MaxPool2d::MaxPool2d(int batch_size, int channel, int height, int width,
                     int pool_h, int pool_w, int stride)
    : batch_size_(batch_size), channel_(channel), height_(height), width_(width),
      pool_h_(pool_h), pool_w_(pool_w), stride_(stride),
      output_h_(0), output_w_(0), input_col_h_(0), input_col_w_(0),
      input_col_(NULL), dmax_col_(NULL), output_col_(NULL), arg_max_(NULL),
      dout_col_(NULL)
{
    // 出力画像の縦横サイズ
    output_h_ = (height_ - pool_h_) / stride_ + 1;
    output_w_ = (width_ - pool_w_) / stride_ + 1;

    // 入力画像を2次元配列に展開した時のサイズ
    input_col_h_ = batch_size_ * channel_ * output_h_ * output_w_;
    input_col_w_ = pool_h_ * pool_w_;
    input_col_ = util::alloc<double>(input_col_h_, input_col_w_);
    dmax_col_ = util::alloc<double>(input_col_h_, input_col_w_);

    // プーリング結果格納用配列
    output_col_ = util::alloc<double>(input_col_h_);
    arg_max_ = util::alloc<int>(input_col_h_);
    dout_col_ = util::alloc<double>(input_col_h_);
}

MaxPool2d::~MaxPool2d()
{
    util::free(input_col_, input_col_h_);
    util::free(dmax_col_, input_col_h_);
    util::free(output_col_);
    util::free(arg_max_);
    util::free(dout_col_);
}

void MaxPool2d::forward(double const * const * const * const *input, double ****output)
{
    // 入力を2次元配列に展開
    im2col(input, input_col_);

    // 最大値を取得
    for (int i = 0; i < input_col_h_; ++i) {
        int idx = -1;
        int max_val = std::numeric_limits<double>::min();
        for (int j = 0; j < input_col_w_; ++j) {
            if (input_col_[i][j] > max_val) {
                idx = j;
                max_val = input_col_[i][j];
            }
        }
        output_col_[i] = max_val;
        arg_max_[i] = idx;
    }

    // 出力の形式を戻す
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < output_h_; ++y) {
                for (int x = 0; x < output_w_; ++x) {
                    int idx = n * (channel_ * output_h_ * output_w_)
                              + c * (output_h_ * output_w_)
                              + y * output_w_ + x;
                    output[n][c][y][x] = output_col_[idx];
                }
            }
        }
    }
}

void MaxPool2d::backward(double const * const * const * const *dout, double ****dx)
{
    // 入力を行列形式に変換
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < output_h_; ++y) {
                for (int x = 0; x < output_w_; ++x) {
                    int idx = n * (channel_ * output_h_ * output_w_)
                              + c * (output_h_ * output_w_)
                              + y * output_w_ + x;
                    dout_col_[idx] = dout[n][c][y][x];
                }
            }
        }
    }

    // 誤差逆伝播
    for (int i = 0; i < input_col_h_; ++i) {
        for (int j = 0; j < input_col_w_; ++j) {
            dmax_col_[i][j] = 0.0;
        }
        int idx = arg_max_[i];
        dmax_col_[i][idx] = dout_col_[i];
    }

    col2im(dmax_col_, dx);
}

void MaxPool2d::im2col(double const * const * const * const *src, double **dst)
{
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y <= height_ - pool_h_; y += stride_) {
                for (int x = 0; x <= width_ - pool_w_; x += stride_) {
                    int dst_y = n * (channel_ * output_h_ * output_w_)
                                + c * (output_h_ * output_w_)
                                + (y / stride_) * output_w_ + (x / stride_);
                    for (int v = 0; v < pool_h_; ++v) {
                        for (int u = 0; u < pool_w_; ++u) {
                            int src_y = y + v;
                            int src_x = x + u;
                            int dst_x = v * pool_w_ + u;

                            dst[dst_y][dst_x] = src[n][c][src_y][src_x];
                        }
                    }
                }
            }
        }
    }
}

void MaxPool2d::col2im(double const * const *src, double ****dst)
{
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    dst[n][c][y][x] = 0.0;
                }
            }
            for (int y = 0; y <= height_ - pool_h_; y += stride_) {
                for (int x = 0; x <= width_ - pool_w_; x += stride_) {
                    int src_y = n * (channel_ * output_h_ * output_w_)
                                + c * (output_h_ * output_w_)
                                + (y / stride_) * output_w_ + (x / stride_);
                    for (int v = 0; v < pool_h_; ++v) {
                        for (int u = 0; u < pool_w_; ++u) {
                            int dst_y = y + v;
                            int dst_x = x + u;
                            int src_x = v * pool_w_ + u;

                            dst[n][c][dst_y][dst_x] += src[src_y][src_x];
                        }
                    }
                }
            }
        }
    }
}