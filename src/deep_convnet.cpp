#include "deep_convnet.h"

#include <iostream>

DeepConvNet::DeepConvNet(int batch_size, int input_dim[], int hidden_size, int output_size)
    : batch_size_(batch_size), hidden_size_(hidden_size),
      output_size_(output_size), pool_h_(2), pool_w_(2), pool_stride_(2), pool_output_size_(0),
      layer1_(NULL), layer2_(NULL), layer3_(NULL), layer4_(NULL), layer5_(NULL),
      layer6_(NULL), layer7_(NULL), layer8_(NULL), layer9_(NULL), layer10_(NULL),
      layer11_(NULL), layer12_(NULL), layer13_(NULL), layer14_(NULL), layer15_(NULL),
      layer16_(NULL), layer17_(NULL), layer18_(NULL), layer19_(NULL), layer20_(NULL),
      last_layer_(NULL), optim_(NULL), train_flg_(true),
      weight2d_(NULL), weight_(NULL), bias_(NULL), x2d_(NULL), x_(NULL), output_(NULL),
      d_weight2d_(NULL), d_weight_(NULL), d_bias_(NULL), dout2d_(NULL), dout_(NULL),
      conv_param_(NULL), conv_output_h_(NULL), conv_output_w_(NULL)
{
    input_dim_[0] = input_dim[0];
    input_dim_[1] = input_dim[1];
    input_dim_[2] = input_dim[2];

    conv_param_ = new util::ConvParam*[6];
    conv_param_[0] = new util::ConvParam(16, 3, 3, 1, 1);
    conv_param_[1] = new util::ConvParam(16, 3, 3, 1, 1);
    conv_param_[2] = new util::ConvParam(32, 3, 3, 1, 1);
    conv_param_[3] = new util::ConvParam(32, 3, 3, 1, 2);
    conv_param_[4] = new util::ConvParam(64, 3, 3, 1, 1);
    conv_param_[5] = new util::ConvParam(64, 3, 3, 1, 1);

    conv_output_h_ = new int[9];
    conv_output_w_ = new int[9];

    // 1: Conv2d
    layer1_ = new Conv2d(batch_size_, input_dim_[0], input_dim_[1], input_dim_[2],
                         conv_param_[0]->filter_num, conv_param_[0]->filter_h,
                         conv_param_[0]->filter_w, conv_param_[0]->stride,
                         conv_param_[0]->pad);
    // 2: ReLU
    conv_output_h_[0] = (input_dim_[1] + 2 * conv_param_[0]->pad - conv_param_[0]->filter_h)
                        / conv_param_[0]->stride + 1;
    conv_output_w_[0] = (input_dim_[2] + 2 * conv_param_[0]->pad - conv_param_[0]->filter_w)
                        / conv_param_[0]->stride + 1;
    layer2_ = new ReLU(batch_size_, 0, conv_param_[0]->filter_num, conv_output_h_[0],
                       conv_output_w_[0]);
    // 3: Conv2d
    layer3_ = new Conv2d(batch_size_, conv_param_[0]->filter_num, conv_output_h_[0],
                         conv_output_w_[0], conv_param_[1]->filter_num, conv_param_[1]->filter_h,
                         conv_param_[1]->filter_w, conv_param_[1]->stride, conv_param_[1]->pad);
    // 4: ReLU
    conv_output_h_[1] = (conv_output_h_[0] + 2 * conv_param_[1]->pad - conv_param_[1]->filter_h)
                        / conv_param_[1]->stride + 1;
    conv_output_w_[1] = (conv_output_w_[0] + 2 * conv_param_[1]->pad - conv_param_[1]->filter_w)
                        / conv_param_[1]->stride + 1;
    layer4_ = new ReLU(batch_size_, 0, conv_param_[1]->filter_num, conv_output_h_[1],
                       conv_output_w_[1]);
    // 5: MaxPool2d
    layer5_ = new MaxPool2d(batch_size_, conv_param_[1]->filter_num, conv_output_h_[1],
                            conv_output_w_[1], pool_h_, pool_w_, pool_stride_);
    // 6: Conv2d
    conv_output_h_[2] = (conv_output_h_[1] - pool_h_) / pool_stride_ + 1;
    conv_output_w_[2] = (conv_output_w_[1] - pool_w_) / pool_stride_ + 1;
    layer6_ = new Conv2d(batch_size_, conv_param_[1]->filter_num, conv_output_h_[2],
                         conv_output_w_[2], conv_param_[2]->filter_num, conv_param_[2]->filter_h,
                         conv_param_[2]->filter_w, conv_param_[2]->stride, conv_param_[2]->pad);
    // 7: ReLU
    conv_output_h_[3] = (conv_output_h_[2] + 2 * conv_param_[2]->pad - conv_param_[2]->filter_h)
                        / conv_param_[2]->stride + 1;
    conv_output_w_[3] = (conv_output_w_[2] + 2 * conv_param_[2]->pad - conv_param_[2]->filter_w)
                        / conv_param_[2]->stride + 1;
    layer7_ = new ReLU(batch_size_, 0, conv_param_[2]->filter_num, conv_output_h_[3],
                       conv_output_w_[3]);
    // 8: Conv2d
    layer8_ = new Conv2d(batch_size_, conv_param_[2]->filter_num, conv_output_h_[3],
                         conv_output_w_[3], conv_param_[3]->filter_num, conv_param_[3]->filter_h,
                         conv_param_[3]->filter_w, conv_param_[3]->stride, conv_param_[3]->pad);
    // 9: ReLU
    conv_output_h_[4] = (conv_output_h_[3] + 2 * conv_param_[3]->pad - conv_param_[3]->filter_h)
                        / conv_param_[3]->stride + 1;
    conv_output_w_[4] = (conv_output_w_[3] + 2 * conv_param_[3]->pad - conv_param_[3]->filter_w)
                        / conv_param_[3]->stride + 1;
    layer9_ = new ReLU(batch_size_, 0, conv_param_[3]->filter_num, conv_output_h_[4],
                       conv_output_w_[4]);
    // 10: MaxPool2d
    layer10_ = new MaxPool2d(batch_size_, conv_param_[3]->filter_num, conv_output_h_[4],
                             conv_output_w_[4], pool_h_, pool_w_, pool_stride_);
    // 11: Conv2d
    conv_output_h_[5] = (conv_output_h_[4] - pool_h_) / pool_stride_ + 1;
    conv_output_w_[5] = (conv_output_w_[4] - pool_w_) / pool_stride_ + 1;
    layer11_ = new Conv2d(batch_size_, conv_param_[3]->filter_num, conv_output_h_[5],
                          conv_output_w_[5], conv_param_[4]->filter_num, conv_param_[4]->filter_h,
                          conv_param_[4]->filter_w, conv_param_[4]->stride, conv_param_[4]->pad);
    // 12: ReLU
    conv_output_h_[6] = (conv_output_h_[5] + 2 * conv_param_[4]->pad - conv_param_[4]->filter_h)
                        / conv_param_[4]->stride + 1;
    conv_output_w_[6] = (conv_output_w_[5] + 2 * conv_param_[4]->pad - conv_param_[4]->filter_w)
                        / conv_param_[4]->stride + 1;
    layer12_ = new ReLU(batch_size_, 0, conv_param_[4]->filter_num, conv_output_h_[6],
                        conv_output_w_[6]);
    // 13: Conv2d
    layer13_ = new Conv2d(batch_size_, conv_param_[4]->filter_num, conv_output_h_[6],
                          conv_output_w_[6], conv_param_[5]->filter_num, conv_param_[5]->filter_h,
                          conv_param_[5]->filter_w, conv_param_[5]->stride, conv_param_[5]->pad);
    // 14: ReLU
    conv_output_h_[7] = (conv_output_h_[6] + 2 * conv_param_[5]->pad - conv_param_[5]->filter_h)
                        / conv_param_[5]->stride + 1;
    conv_output_w_[7] = (conv_output_w_[6] + 2 * conv_param_[5]->pad - conv_param_[5]->filter_w)
                        / conv_param_[5]->stride + 1;
    layer14_ = new ReLU(batch_size_, 0, conv_param_[5]->filter_num, conv_output_h_[7],
                        conv_output_w_[7]);
    // 15: MaxPool2d
    layer15_ = new MaxPool2d(batch_size_, conv_param_[5]->filter_num, conv_output_h_[7],
                             conv_output_w_[7], pool_h_, pool_w_, pool_stride_);
    // 16: Linear
    conv_output_h_[8] = (conv_output_h_[7] - pool_h_) / pool_stride_ + 1;
    conv_output_w_[8] = (conv_output_w_[7] - pool_w_) / pool_stride_ + 1;
    pool_output_size_ = conv_param_[5]->filter_num * conv_output_h_[8] * conv_output_w_[8];
    layer16_ = new Linear(batch_size_, pool_output_size_, hidden_size_,
                          conv_param_[5]->filter_num, conv_output_h_[8], conv_output_w_[8]);
    // 17: ReLU
    layer17_ = new ReLU(batch_size_, hidden_size_);
    // 18: Dropout
    layer18_ = new Dropout(batch_size_, hidden_size_);
    // 19: Linear
    layer19_ = new Linear(batch_size_, hidden_size_, output_size_);
    // 20: Dropout
    layer20_ = new Dropout(batch_size_, output_size_);
    // 21: Softmax with Cross Entropy
    last_layer_ = new SoftmaxWithLoss(batch_size_, output_size_);

    double lr = 0.001;
    optim_ = new Adam*[16];
    optim_[0] = new Adam(lr, 0.9, 0.999, conv_param_[0]->filter_num, input_dim_[0],
                         conv_param_[0]->filter_h, conv_param_[0]->filter_w);
    optim_[1] = new Adam(lr, 0.9, 0.999, conv_param_[0]->filter_num);
    for (int i = 1; i < 6; ++i) {
        optim_[i * 2] = new Adam(lr, 0.9, 0.999, conv_param_[i]->filter_num, conv_param_[i - 1]->filter_num,
                             conv_param_[i]->filter_h, conv_param_[i]->filter_w);
        optim_[i * 2 + 1] = new Adam(lr, 0.9, 0.999, conv_param_[i]->filter_num);
    }
    optim_[12] = new Adam(lr, 0.9, 0.999, pool_output_size_, hidden_size_);
    optim_[13] = new Adam(lr, 0.9, 0.999, hidden_size_);
    optim_[14] = new Adam(lr, 0.9, 0.999, hidden_size_, output_size_);
    optim_[15] = new Adam(lr, 0.9, 0.999, output_size_);

    weight2d_ = new double****[6];
    d_weight2d_ = new double****[6];
    weight2d_[0] = util::alloc<double>(conv_param_[0]->filter_num, input_dim_[0],
                                       conv_param_[0]->filter_h, conv_param_[0]->filter_w);
    d_weight2d_[0] = util::alloc<double>(conv_param_[0]->filter_num, input_dim_[0],
                                         conv_param_[0]->filter_h, conv_param_[0]->filter_w);
    for (int i = 0; i < conv_param_[0]->filter_num; ++i) {
        for (int j = 0; j < input_dim_[0]; ++j) {
            for (int k = 0; k < conv_param_[0]->filter_h; ++k) {
                for (int l = 0; l < conv_param_[0]->filter_w; ++l) {
                    double stddev = input_dim_[0] * conv_param_[0]->filter_h * conv_param_[0]->filter_w;
                    stddev = sqrt(2.0 / stddev);
                    weight2d_[0][i][j][k][l] = util::randn(0.0, stddev);
                }
            }
        }
    }
    for (int i = 1; i < 6; ++i) {
        weight2d_[i] = util::alloc<double>(conv_param_[i]->filter_num, conv_param_[i - 1]->filter_num,
                                           conv_param_[i]->filter_h, conv_param_[i]->filter_w);
        d_weight2d_[i] = util::alloc<double>(conv_param_[i]->filter_num, conv_param_[i - 1]->filter_num,
                                             conv_param_[i]->filter_h, conv_param_[i]->filter_w);
        for (int j = 0; j < conv_param_[i]->filter_num; ++j) {
            for (int k = 0; k < conv_param_[i - 1]->filter_num; ++k) {
                for (int l = 0; l < conv_param_[i]->filter_h; ++l) {
                    for (int m = 0; m < conv_param_[i]->filter_w; ++m) {
                        double stddev = conv_param_[i - 1]->filter_num * conv_param_[i]->filter_h
                                        * conv_param_[i]->filter_w;
                        stddev = sqrt(2.0 / stddev);
                        weight2d_[i][j][k][l][m] = util::randn(0.0, stddev);
                    }
                }
            }
        }
    }

    weight_ = new double**[2];
    d_weight_ = new double**[2];
    weight_[0] = util::alloc<double>(pool_output_size_, hidden_size_);
    d_weight_[0] = util::alloc<double>(pool_output_size_, hidden_size_);
    for (int i = 0; i < pool_output_size_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            double stddev = sqrt(2.0 / pool_output_size_);
            weight_[0][i][j] = util::randn(0.0, stddev);
        }
    }
    weight_[1] = util::alloc<double>(hidden_size_, output_size_);
    d_weight_[1] = util::alloc<double>(hidden_size_, output_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < output_size_; ++j) {
            double stddev = sqrt(2.0 / hidden_size_);
            weight_[1][i][j] = util::randn(0.0, stddev);
        }
    }

    bias_ = new double*[8];
    d_bias_ = new double*[8];
    for (int i = 0; i < 6; ++i) {
        bias_[i] = util::alloc<double>(conv_param_[i]->filter_num);
        d_bias_[i] = util::alloc<double>(conv_param_[i]->filter_num);
    }
    bias_[6] = util::alloc<double>(hidden_size_);
    d_bias_[6] = util::alloc<double>(hidden_size_);
    bias_[7] = util::alloc<double>(output_size_);
    d_bias_[7] = util::alloc<double>(output_size_);

    x2d_ = new double****[15];
    x2d_[0] = util::alloc<double>(batch_size_, conv_param_[0]->filter_num, conv_output_h_[0], conv_output_w_[0]);
    x2d_[1] = util::alloc<double>(batch_size_, conv_param_[0]->filter_num, conv_output_h_[0], conv_output_w_[0]);
    x2d_[2] = util::alloc<double>(batch_size_, conv_param_[1]->filter_num, conv_output_h_[1], conv_output_w_[1]);
    x2d_[3] = util::alloc<double>(batch_size_, conv_param_[1]->filter_num, conv_output_h_[1], conv_output_w_[1]);
    x2d_[4] = util::alloc<double>(batch_size_, conv_param_[1]->filter_num, conv_output_h_[2], conv_output_w_[2]);
    x2d_[5] = util::alloc<double>(batch_size_, conv_param_[2]->filter_num, conv_output_h_[3], conv_output_w_[3]);
    x2d_[6] = util::alloc<double>(batch_size_, conv_param_[2]->filter_num, conv_output_h_[3], conv_output_w_[3]);
    x2d_[7] = util::alloc<double>(batch_size_, conv_param_[3]->filter_num, conv_output_h_[4], conv_output_w_[4]);
    x2d_[8] = util::alloc<double>(batch_size_, conv_param_[3]->filter_num, conv_output_h_[4], conv_output_w_[4]);
    x2d_[9] = util::alloc<double>(batch_size_, conv_param_[3]->filter_num, conv_output_h_[5], conv_output_w_[5]);
    x2d_[10] = util::alloc<double>(batch_size_, conv_param_[4]->filter_num, conv_output_h_[6], conv_output_w_[6]);
    x2d_[11] = util::alloc<double>(batch_size_, conv_param_[4]->filter_num, conv_output_h_[6], conv_output_w_[6]);
    x2d_[12] = util::alloc<double>(batch_size_, conv_param_[5]->filter_num, conv_output_h_[7], conv_output_w_[7]);
    x2d_[13] = util::alloc<double>(batch_size_, conv_param_[5]->filter_num, conv_output_h_[7], conv_output_w_[7]);
    x2d_[14] = util::alloc<double>(batch_size_, conv_param_[5]->filter_num, conv_output_h_[8], conv_output_w_[8]);

    dout2d_ = new double****[16];
    dout2d_[0] = util::alloc<double>(batch_size_, conv_param_[5]->filter_num, conv_output_h_[8], conv_output_w_[8]);
    dout2d_[1] = util::alloc<double>(batch_size_, conv_param_[5]->filter_num, conv_output_h_[7], conv_output_w_[7]);
    dout2d_[2] = util::alloc<double>(batch_size_, conv_param_[5]->filter_num, conv_output_h_[7], conv_output_w_[7]);
    dout2d_[3] = util::alloc<double>(batch_size_, conv_param_[4]->filter_num, conv_output_h_[6], conv_output_w_[6]);
    dout2d_[4] = util::alloc<double>(batch_size_, conv_param_[4]->filter_num, conv_output_h_[6], conv_output_w_[6]);
    dout2d_[5] = util::alloc<double>(batch_size_, conv_param_[3]->filter_num, conv_output_h_[5], conv_output_w_[5]);
    dout2d_[6] = util::alloc<double>(batch_size_, conv_param_[3]->filter_num, conv_output_h_[4], conv_output_w_[4]);
    dout2d_[7] = util::alloc<double>(batch_size_, conv_param_[3]->filter_num, conv_output_h_[4], conv_output_w_[4]);
    dout2d_[8] = util::alloc<double>(batch_size_, conv_param_[2]->filter_num, conv_output_h_[3], conv_output_w_[3]);
    dout2d_[9] = util::alloc<double>(batch_size_, conv_param_[2]->filter_num, conv_output_h_[3], conv_output_w_[3]);
    dout2d_[10] = util::alloc<double>(batch_size_, conv_param_[1]->filter_num, conv_output_h_[2], conv_output_w_[2]);
    dout2d_[11] = util::alloc<double>(batch_size_, conv_param_[1]->filter_num, conv_output_h_[1], conv_output_w_[1]);
    dout2d_[12] = util::alloc<double>(batch_size_, conv_param_[1]->filter_num, conv_output_h_[1], conv_output_w_[1]);
    dout2d_[13] = util::alloc<double>(batch_size_, conv_param_[0]->filter_num, conv_output_h_[0], conv_output_w_[0]);
    dout2d_[14] = util::alloc<double>(batch_size_, conv_param_[0]->filter_num, conv_output_h_[0], conv_output_w_[0]);
    dout2d_[15] = util::alloc<double>(batch_size_, input_dim_[0], input_dim_[1], input_dim_[2]);

    x_ = new double**[4];
    x_[0] = util::alloc<double>(batch_size_, hidden_size_);
    x_[1] = util::alloc<double>(batch_size_, hidden_size_);
    x_[2] = util::alloc<double>(batch_size_, hidden_size_);
    x_[3] = util::alloc<double>(batch_size_, output_size_);

    dout_ = new double**[5];
    dout_[0] = util::alloc<double>(batch_size_, output_size_);
    dout_[1] = util::alloc<double>(batch_size_, output_size_);
    dout_[2] = util::alloc<double>(batch_size_, hidden_size_);
    dout_[3] = util::alloc<double>(batch_size_, hidden_size_);
    dout_[4] = util::alloc<double>(batch_size_, hidden_size_);

    output_ = util::alloc<double>(batch_size_, output_size_);
}

DeepConvNet::~DeepConvNet()
{
    util::free(weight2d_[0], conv_param_[0]->filter_num, input_dim_[0],
               conv_param_[0]->filter_h);
    util::free(d_weight2d_[0], conv_param_[0]->filter_num, input_dim_[0],
               conv_param_[0]->filter_h);
    for (int i = 1; i < 6; ++i) {
        util::free(weight2d_[i], conv_param_[i]->filter_num, conv_param_[i - 1]->filter_num,
                   conv_param_[i]->filter_h);
        util::free(d_weight2d_[i], conv_param_[i]->filter_num, conv_param_[i - 1]->filter_num,
                   conv_param_[i]->filter_h);
    }
    delete[] weight2d_;
    delete[] d_weight2d_;

    util::free(weight_[0], pool_output_size_);
    util::free(d_weight_[0], pool_output_size_);
    util::free(weight_[1], hidden_size_);
    util::free(d_weight_[1], hidden_size_);
    delete[] weight_;
    delete[] d_weight_;

    util::free(bias_, 8);
    util::free(d_bias_, 8);

    util::free(x2d_[0], batch_size_, conv_param_[0]->filter_num, conv_output_h_[0]);
    util::free(x2d_[1], batch_size_, conv_param_[0]->filter_num, conv_output_h_[0]);
    util::free(x2d_[2], batch_size_, conv_param_[1]->filter_num, conv_output_h_[1]);
    util::free(x2d_[3], batch_size_, conv_param_[1]->filter_num, conv_output_h_[1]);
    util::free(x2d_[4], batch_size_, conv_param_[1]->filter_num, conv_output_h_[2]);
    util::free(x2d_[5], batch_size_, conv_param_[2]->filter_num, conv_output_h_[3]);
    util::free(x2d_[6], batch_size_, conv_param_[2]->filter_num, conv_output_h_[3]);
    util::free(x2d_[7], batch_size_, conv_param_[3]->filter_num, conv_output_h_[4]);
    util::free(x2d_[8], batch_size_, conv_param_[3]->filter_num, conv_output_h_[4]);
    util::free(x2d_[9], batch_size_, conv_param_[3]->filter_num, conv_output_h_[5]);
    util::free(x2d_[10], batch_size_, conv_param_[4]->filter_num, conv_output_h_[6]);
    util::free(x2d_[11], batch_size_, conv_param_[4]->filter_num, conv_output_h_[6]);
    util::free(x2d_[12], batch_size_, conv_param_[5]->filter_num, conv_output_h_[7]);
    util::free(x2d_[13], batch_size_, conv_param_[5]->filter_num, conv_output_h_[7]);
    util::free(x2d_[14], batch_size_, conv_param_[5]->filter_num, conv_output_h_[8]);
    delete[] x2d_;

    util::free(dout2d_[0], batch_size_, conv_param_[5]->filter_num, conv_output_h_[8]);
    util::free(dout2d_[1], batch_size_, conv_param_[5]->filter_num, conv_output_h_[7]);
    util::free(dout2d_[2], batch_size_, conv_param_[5]->filter_num, conv_output_h_[7]);
    util::free(dout2d_[3], batch_size_, conv_param_[4]->filter_num, conv_output_h_[6]);
    util::free(dout2d_[4], batch_size_, conv_param_[4]->filter_num, conv_output_h_[6]);
    util::free(dout2d_[5], batch_size_, conv_param_[3]->filter_num, conv_output_h_[5]);
    util::free(dout2d_[6], batch_size_, conv_param_[3]->filter_num, conv_output_h_[4]);
    util::free(dout2d_[7], batch_size_, conv_param_[3]->filter_num, conv_output_h_[4]);
    util::free(dout2d_[8], batch_size_, conv_param_[2]->filter_num, conv_output_h_[3]);
    util::free(dout2d_[9], batch_size_, conv_param_[2]->filter_num, conv_output_h_[3]);
    util::free(dout2d_[10], batch_size_, conv_param_[1]->filter_num, conv_output_h_[2]);
    util::free(dout2d_[11], batch_size_, conv_param_[1]->filter_num, conv_output_h_[1]);
    util::free(dout2d_[12], batch_size_, conv_param_[1]->filter_num, conv_output_h_[1]);
    util::free(dout2d_[13], batch_size_, conv_param_[0]->filter_num, conv_output_h_[0]);
    util::free(dout2d_[14], batch_size_, conv_param_[0]->filter_num, conv_output_h_[0]);
    util::free(dout2d_[15], batch_size_, input_dim_[0], input_dim_[1]);
    delete[] dout2d_;

    util::free(x_, 4, batch_size_);
    util::free(dout_, 5, batch_size_);
    util::free(output_, batch_size_);

    for (int i = 0; i < 6; ++i) {
        delete conv_param_[i];
    }
    delete[] conv_param_;

    delete[] conv_output_h_;
    delete[] conv_output_w_;

    delete layer1_;
    delete layer2_;
    delete layer3_;
    delete layer4_;
    delete layer5_;
    delete layer6_;
    delete layer7_;
    delete layer8_;
    delete layer9_;
    delete layer10_;
    delete layer11_;
    delete layer12_;
    delete layer13_;
    delete layer14_;
    delete layer15_;
    delete layer16_;
    delete layer17_;
    delete layer18_;
    delete layer19_;
    delete layer20_;
    delete last_layer_;

    for (int i = 0; i < 16; ++i) {
        delete optim_[i];
    }
    delete[] optim_;
}

void DeepConvNet::predict(double const * const * const * const *input,
                          double **output)
{
    // 1: Conv2d
    std::cout << "layer 1: Conv2d" << std::endl;
    layer1_->forward(input, weight2d_[0], bias_[0], x2d_[0]);
    // 2: ReLU
    std::cout << "layer 2: ReLU" << std::endl;
    layer2_->forward(x2d_[0], x2d_[1]);
    // 3: Conv2d
    std::cout << "layer 3: Conv2d" << std::endl;
    layer3_->forward(x2d_[1], weight2d_[1], bias_[1], x2d_[2]);
    // 4: ReLU
    std::cout << "layer 4: ReLU" << std::endl;
    layer4_->forward(x2d_[2], x2d_[3]);
    // 5: MaxPool2d
    std::cout << "layer 5: MaxPool2d" << std::endl;
    layer5_->forward(x2d_[3], x2d_[4]);
    // 6: Conv2d
    std::cout << "layer 6: Conv2d" << std::endl;
    layer6_->forward(x2d_[4], weight2d_[2], bias_[2], x2d_[5]);
    // 7: ReLU
    std::cout << "layer 7: ReLU" << std::endl;
    layer7_->forward(x2d_[5], x2d_[6]);
    // 8: Conv2d
    std::cout << "layer 8: Conv2d" << std::endl;
    layer8_->forward(x2d_[6], weight2d_[3], bias_[3], x2d_[7]);
    // 9: ReLU
    std::cout << "layer 9: ReLU" << std::endl;
    layer9_->forward(x2d_[7], x2d_[8]);
    // 10: MaxPool2d
    std::cout << "layer 10: MaxPool2d" << std::endl;
    layer10_->forward(x2d_[8], x2d_[9]);
    // 11: Conv2d
    std::cout << "layer 11: Conv2d" << std::endl;
    layer11_->forward(x2d_[9], weight2d_[4], bias_[4], x2d_[10]);
    // 12: ReLU
    std::cout << "layer 12: ReLU" << std::endl;
    layer12_->forward(x2d_[10], x2d_[11]);
    // 13: Conv2d
    std::cout << "layer 13: Conv2d" << std::endl;
    layer13_->forward(x2d_[11], weight2d_[5], bias_[5], x2d_[12]);
    // 14: ReLU
    std::cout << "layer 14: ReLU" << std::endl;
    layer14_->forward(x2d_[12], x2d_[13]);
    // 15: MaxPool2d
    std::cout << "layer 15: MaxPool2d" << std::endl;
    layer15_->forward(x2d_[13], x2d_[14]);
    // 16: Linear
    std::cout << "layer 16: Linear" << std::endl;
    layer16_->forward(x2d_[14], weight_[0], bias_[6], x_[0]);
    // 17: ReLU
    std::cout << "layer 17: ReLU" << std::endl;
    layer17_->forward(x_[0], x_[1]);
    // 18: Dropout
    std::cout << "layer 18: Dropout" << std::endl;
    layer18_->forward(x_[1], x_[2], train_flg_);
    // 19: Linear
    std::cout << "layer 19: Linear" << std::endl;
    layer19_->forward(x_[2], weight_[1], bias_[7], x_[3]);
    // 20: Dropout
    std::cout << "layer 20: Dropout" << std::endl;
    layer20_->forward(x_[3], output, train_flg_);
}

double DeepConvNet::loss(double const * const * const * const *input,
                         double const * const *criterion)
{
    predict(input, output_);

    // 21: Softmax with Cross Entropy
    std::cout << "last layer: SoftmaxWithLoss" << std::endl;
    return last_layer_->forward(output_, criterion);
}

void DeepConvNet::gradient(double const * const *criterion)
{
    // 21: Softmax with Cross Entropy
    std::cout << "last layer: SoftmaxWithLoss" << std::endl;
    last_layer_->backward(criterion, dout_[0]);
    // 20: Dropout
    std::cout << "layer 20: Dropout" << std::endl;
    layer20_->backward(dout_[0], dout_[1]);
    // 19: Linear
    std::cout << "layer 19: Linear" << std::endl;
    layer19_->backward(dout_[1], d_weight_[1], d_bias_[7], dout_[2]);
    // 18: Dropout
    std::cout << "layer 18: Dropout" << std::endl;
    layer18_->backward(dout_[2], dout_[3]);
    // 17: ReLU
    std::cout << "layer 17: ReLU" << std::endl;
    layer17_->backward(dout_[3], dout_[4]);
    // 16: Linear
    std::cout << "layer 16: Linear" << std::endl;
    layer16_->backward(dout_[4], d_weight_[0], d_bias_[6], dout2d_[0]);
    // 15: MaxPool2d
    std::cout << "layer 15: MaxPool2d" << std::endl;
    layer15_->backward(dout2d_[0], dout2d_[1]);
    // 14: ReLU
    std::cout << "layer 14: ReLU" << std::endl;
    layer14_->backward(dout2d_[1], dout2d_[2]);
    // 13: Conv2d
    std::cout << "layer 13: Conv2d" << std::endl;
    layer13_->backward(dout2d_[2], d_weight2d_[5], d_bias_[5], dout2d_[3]);
    // 12: ReLU
    std::cout << "layer 12: ReLU" << std::endl;
    layer12_->backward(dout2d_[3], dout2d_[4]);
    // 11: Conv2d
    std::cout << "layer 11: Conv2d" << std::endl;
    layer11_->backward(dout2d_[4], d_weight2d_[4], d_bias_[4], dout2d_[5]);
    // 10: MaxPool2d
    std::cout << "layer 10: MaxPool2d" << std::endl;
    layer10_->backward(dout2d_[5], dout2d_[6]);
    // 9: ReLU
    std::cout << "layer 9: ReLU" << std::endl;
    layer9_->backward(dout2d_[6], dout2d_[7]);
    // 8: Conv2d
    std::cout << "layer 8: Conv2d" << std::endl;
    layer8_->backward(dout2d_[7], d_weight2d_[3], d_bias_[3], dout2d_[8]);
    // 7: ReLU
    std::cout << "layer 7: ReLU" << std::endl;
    layer7_->backward(dout2d_[8], dout2d_[9]);
    // 6: Conv2d
    std::cout << "layer 6: Conv2d" << std::endl;
    layer6_->backward(dout2d_[9], d_weight2d_[2], d_bias_[2], dout2d_[10]);
    // 5: MaxPool2d
    std::cout << "layer 5: MaxPool2d" << std::endl;
    layer5_->backward(dout2d_[10], dout2d_[11]);
    // 4: ReLU
    std::cout << "layer 4: ReLU" << std::endl;
    layer4_->backward(dout2d_[11], dout2d_[12]);
    // 3: Conv2d
    std::cout << "layer 3: Conv2d" << std::endl;
    layer3_->backward(dout2d_[12], d_weight2d_[1], d_bias_[1], dout2d_[13]);
    // 2: ReLU
    std::cout << "layer 2: ReLU" << std::endl;
    layer2_->backward(dout2d_[13], dout2d_[14]);
    // 1: Conv2d
    std::cout << "layer 1: Conv2d" << std::endl;
    layer1_->backward(dout2d_[14], d_weight2d_[0], d_bias_[0], dout2d_[15]);
}

void DeepConvNet::update()
{
    optim_[0]->update(weight2d_[0], d_weight2d_[0]);
    optim_[1]->update(bias_[0], d_bias_[0]);
    optim_[2]->update(weight2d_[1], d_weight2d_[1]);
    optim_[3]->update(bias_[1], d_bias_[1]);
    optim_[4]->update(weight2d_[2], d_weight2d_[2]);
    optim_[5]->update(bias_[2], d_bias_[2]);
    optim_[6]->update(weight2d_[3], d_weight2d_[3]);
    optim_[7]->update(bias_[3], d_bias_[3]);
    optim_[8]->update(weight2d_[4], d_weight2d_[4]);
    optim_[9]->update(bias_[4], d_bias_[4]);
    optim_[10]->update(weight2d_[5], d_weight2d_[5]);
    optim_[11]->update(bias_[5], d_bias_[5]);
    optim_[12]->update(weight_[0], d_weight_[0]);
    optim_[13]->update(bias_[6], d_bias_[6]);
    optim_[14]->update(weight_[1], d_weight_[1]);
    optim_[15]->update(bias_[7], d_bias_[7]);
}
