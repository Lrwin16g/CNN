#include "conv2d.h"
#include "utils.h"

Conv2d::Conv2d(int batch_size, int channel, int height, int width,
               int filter_num, int filter_h, int filter_w, int stride, int pad)
    : batch_size_(batch_size), channel_(channel), height_(height), width_(width),
      filter_num_(filter_num), filter_h_(filter_h), filter_w_(filter_w),
      stride_(stride), pad_(pad), output_h_(0), output_w_(0),
      input_col_h_(0), input_col_w_(0), weight_col_h_(0), weight_col_w_(0),
      tmp_col_h_(0), tmp_col_w_(0), output_col_h_(0), output_col_w_(0),
      input_col_(NULL), input_col_T_(NULL), dx_col_(NULL), weight_col_(NULL),
      weight_col_T_(NULL), tmp_col_(NULL), output_col_(NULL), dout_col_(NULL),
      d_weight_(NULL), d_bias_(NULL), d_weight_col_(NULL)
{
    // 出力画像の縦横サイズ
    output_h_ = (height_ + 2 * pad_ - filter_h_) / stride_ + 1;
    output_w_ = (width_ + 2 * pad_ - filter_w_) / stride_ + 1;

    // 入力画像を2次元配列に展開した時のサイズ
    input_col_h_ = batch_size_ * output_h_ * output_w_;
    input_col_w_ = channel_ * filter_h_ * filter_w_;
    input_col_ = util::alloc<double>(input_col_h_, input_col_w_);
    input_col_T_ = util::alloc<double>(input_col_w_, input_col_h_);
    dx_col_ = util::alloc<double>(input_col_h_, input_col_w_);

    // 重みを2次元配列に展開した時のサイズ
    weight_col_h_ = channel_ * filter_h_ * filter_w_;
    weight_col_w_ = filter_num_;
    weight_col_ = util::alloc<double>(weight_col_h_, weight_col_w_);
    weight_col_T_ = util::alloc<double>(weight_col_w_, weight_col_h_);

    // 入力と重みの積を格納する行列
    tmp_col_h_ = input_col_h_;
    tmp_col_w_ = weight_col_w_;
    tmp_col_ = util::alloc<double>(tmp_col_h_, tmp_col_w_);

    // 出力の行列形式
    output_col_h_ = input_col_h_;
    output_col_w_ = weight_col_w_;
    output_col_ = util::alloc<double>(output_col_h_, output_col_w_);
    dout_col_ = util::alloc<double>(output_col_h_, output_col_w_);

    // 重み・バイアスの勾配
    d_weight_ = util::alloc<double>(filter_num_, channel_, filter_h_, filter_w_);
    d_bias_ = util::alloc<double>(filter_num_);

    // 重みの勾配の行列形式
    d_weight_col_ = util::alloc<double>(weight_col_h_, weight_col_w_);
}

Conv2d::~Conv2d()
{
    util::free(input_col_, input_col_h_);
    util::free(input_col_T_, input_col_w_);
    util::free(dx_col_, input_col_h_);
    util::free(weight_col_, weight_col_h_);
    util::free(weight_col_T_, weight_col_w_);
    util::free(tmp_col_, tmp_col_h_);
    util::free(output_col_, output_col_h_);
    util::free(dout_col_, output_col_h_);
    util::free(d_weight_, filter_num_, channel_, filter_h_);
    util::free(d_bias_);
    util::free(d_weight_col_, weight_col_h_);
}

void Conv2d::forward(double const * const * const * const *input,
                     double const * const * const * const *weight,
                     const double *bias,
                     double ****output)
{
    // 入力を2次元配列に展開
    im2col(input, input_col_);

    // 重みを2次元配列に展開
    for (int n = 0; n < filter_num_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < filter_h_; ++y) {
                for (int x = 0; x < filter_w_; ++x) {
                    int idx_y = c * (filter_h_ * filter_w_) + y * filter_w_ + x;
                    weight_col_[idx_y][n] = weight[n][c][y][x];
                }
            }
        }
    }

    // 入力と重みの積を計算
    util::dot(input_col_, weight_col_, tmp_col_, input_col_h_, input_col_w_,
              weight_col_h_, weight_col_w_);

    // バイアスを加算
    for (int y = 0; y < output_col_h_; ++y) {
        for (int x = 0; x < output_col_w_; ++x) {
            output_col_[y][x] = tmp_col_[y][x] + bias[x];
        }
    }

    // 出力の形式を戻す
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < filter_num_; ++c) {
            for (int y = 0; y < output_h_; ++y) {
                for (int x = 0; x < output_w_; ++x) {
                    int idx_y = n * (output_h_ * output_w_) + y * output_w_ + x;
                    output[n][c][y][x] = output_col_[idx_y][c];
                }
            }
        }
    }
}

void Conv2d::backward(double const * const * const * const *dout, double ****dx)
{
    // 入力を2次元配列に展開
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < filter_num_; ++c) {
            for (int y = 0; y < output_h_; ++y) {
                for (int x = 0; x < output_w_; ++x) {
                    int idx_y = n * (output_h_ * output_w_) + y * output_w_ + x;
                    dout_col_[idx_y][c] = dout[n][c][y][x];
                }
            }
        }
    }

    // バイアスの勾配を計算
    for (int c = 0; c < filter_num_; ++c) {
        d_bias_[c] = 0.0;
        for (int n = 0; n < batch_size_; ++n) {
            for (int y = 0; y < output_h_; ++y) {
                for (int x = 0; x < output_w_; ++x) {
                    int idx_y = n * (output_h_ * output_w_) + y * output_w_ + x;
                    d_bias_[c] += dout_col_[idx_y][c];
                }
            }
        }
    }

    // 入力の2次元配列を転置
    util::transpose(input_col_, input_col_T_, input_col_h_, input_col_w_);

    // 重みの勾配を計算
    util::dot(input_col_T_, dout_col_, d_weight_col_, input_col_w_, input_col_h_,
              output_col_h_, output_col_w_);

    // 行列をテンソルに戻す
    for (int n = 0; n < filter_num_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < filter_h_; ++y) {
                for (int x = 0; x < filter_w_; ++x) {
                    int idx_y = c * (filter_h_ * filter_w_) + y * filter_w_ + x;
                    d_weight_[n][c][y][x] = d_weight_col_[idx_y][n];
                }
            }
        }
    }

    // 重みを転置
    util::transpose(weight_col_, weight_col_T_, weight_col_h_, weight_col_w_);

    // 誤差逆伝播
    util::dot(dout_col_, weight_col_T_, dx_col_, output_col_h_, output_col_w_,
              weight_col_w_, weight_col_h_);

    // 行列をテンソルに戻す
    col2im(dx_col_, dx);
}

void Conv2d::im2col(double const * const * const * const *src, double **dst)
{
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = -pad_; y <= height_ + pad_ - filter_h_; y += stride_) {
                for (int x = -pad_; x <= width_ + pad_ - filter_w_; x += stride_) {
                    int dst_y = n * (output_h_ * output_w_) + ((y + pad_) / stride_) * output_w_
                                + ((x + pad_) / stride_);
                    for (int v = 0; v < filter_h_; ++v) {
                        for (int u = 0; u < filter_w_; ++u) {
                            int src_y = y + v;
                            int src_x = x + u;
                            int dst_x = c * (filter_h_ * filter_w_) + v * filter_w_ + u;

                            if (0 <= src_y && src_y < height_ && 0 <= src_x && src_x < width_) {
                                dst[dst_y][dst_x] = src[n][c][src_y][src_x];
                            } else {
                                dst[dst_y][dst_x] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

void Conv2d::col2im(double const * const *src, double ****dst)
{
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    dst[n][c][y][x] = 0.0;
                }
            }
            for (int y = -pad_; y <= height_ + pad_ - filter_h_; y += stride_) {
                for (int x = -pad_; x <= width_ + pad_ - filter_w_; x += stride_) {
                    int src_y = n * (output_h_ * output_w_) + ((y + pad_) / stride_) * output_w_
                                + ((x + pad_) / stride_);
                    for (int v = 0; v < filter_h_; ++v) {
                        for (int u = 0; u < filter_w_; ++u) {
                            int src_x = c * (filter_h_ * filter_w_) + v * filter_w_ + u;
                            int dst_y = y + v;
                            int dst_x = x + u;

                            if (0 <= dst_y && dst_y < height_ && 0 <= dst_x && dst_x < width_) {
                                dst[n][c][dst_y][dst_x] += src[src_y][src_x];
                            }
                        }
                    }
                }
            }
        }
    }
}