#include "maxpool2d.h"
#include "utils.h"

#include <limits>

MaxPool2d::MaxPool2d()
{
    // 出力画像の縦横サイズ
    output_h_ = (height_ - pool_h_) / stride_ + 1;
    output_w_ = (width_ - pool_w_) / stride_ + 1;
}

MaxPool2d::~MaxPool2d()
{

}

void MaxPool2d::forward(double const * const * const * const *input, double ****output)
{
    // 入力を2次元配列に展開
    im2col(input, input_col_, batch_size_, channel_, height_, width_,
           pool_h_, pool_w_, stride_);

    // 最大値を取得
    for (int i = 0; i < output_h_; ++i) {
        int idx = -1;
        int max_val = std::numeric_limits<double>::min();
        for (int j = 0; j < output_w_; ++j) {
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

}

void MaxPool2d::im2col(double const * const * const * const *src, double **dst)
{
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < height_ - pool_h_; y += stride_) {
                for (int x = 0; x < width_ - pool_w_; x += stride_) {
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