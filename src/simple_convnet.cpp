#include "simple_convnet.h"
#include "utils.h"

#include <iostream>

SimpleConvNet::SimpleConvNet(int batch_size, int channel, int height, int width,
                             int filter_num, int filter_h, int filter_w, int stride, int pad,
                             int hidden_size, int category_num, double weight_init_std,
                             double lr)
    : batch_size_(batch_size), channel_(channel), height_(height), width_(width),
      filter_num_(filter_num), filter_h_(filter_h), filter_w_(filter_w), stride_(stride),
      pad_(pad), hidden_size_(hidden_size), category_num_(category_num), weight_init_std_(weight_init_std),
      conv_output_h_(0), conv_output_w_(0), pool_h_(0), pool_w_(0), pool_stride_(0),
      pool_output_h_(0), pool_output_w_(0), pool_output_size_(0),
      layer1_(NULL), layer2_(NULL), layer3_(NULL), layer4_(NULL), layer5_(NULL), layer6_(NULL),
      last_layer_(NULL), weight2d_(NULL), weight_(NULL), bias_(NULL), x2d_(NULL), x_(NULL),
      output_(NULL), dout_(NULL), dout2d_(NULL), d_weight2d_(NULL), d_weight_(NULL), d_bias_(NULL),
      optim_(NULL)
{
    // ネットワークの初期化
    // Conv2d
    layer1_ = new Conv2d(batch_size_, channel_, height_, width_,
                         filter_num_, filter_h_, filter_w_, stride_, pad_);
    // ReLU
    conv_output_h_ = (height_ + 2 * pad_ - filter_h_) / stride_ + 1;
    conv_output_w_ = (width_ + 2 * pad_ - filter_w_) / stride_ + 1;
    layer2_ = new ReLU(batch_size_, 0, filter_num_, conv_output_h_, conv_output_w_);
    // MaxPool2d
    pool_h_ = 2;
    pool_w_ = 2;
    pool_stride_ = 2;
    layer3_ = new MaxPool2d(batch_size_, filter_num_, conv_output_h_, conv_output_w_,
                            pool_h_, pool_w_, pool_stride_);
    // Linear
    pool_output_h_ = (conv_output_h_ - pool_h_) / pool_stride_ + 1;
    pool_output_w_ = (conv_output_w_ - pool_w_) / pool_stride_ + 1;
    pool_output_size_ = filter_num_ * pool_output_h_ * pool_output_w_;
    layer4_ = new Linear(batch_size_, pool_output_size_, hidden_size_,
                         filter_num_, pool_output_h_, pool_output_w_);
    // ReLU
    layer5_ = new ReLU(batch_size_, hidden_size_);
    // Linear
    layer6_ = new Linear(batch_size_, hidden_size_, category_num_);
    // Softmax
    last_layer_ = new SoftmaxWithLoss(batch_size_, category_num_);

    //optim_ = new SGD();
    optim_ = new Adam*[6];
    optim_[0] = new Adam(lr, 0.9, 0.999, filter_num_, channel_, filter_h_, filter_w_);
    optim_[1] = new Adam(lr, 0.9, 0.999, filter_num_, 0, 0, 0);
    optim_[2] = new Adam(lr, 0.9, 0.999, pool_output_size_, hidden_size_, 0, 0);
    optim_[3] = new Adam(lr, 0.9, 0.999, hidden_size_, 0, 0, 0);
    optim_[4] = new Adam(lr, 0.9, 0.999, hidden_size_, category_num_, 0, 0);
    optim_[5] = new Adam(lr, 0.9, 0.999, category_num_, 0, 0, 0);

    // 重みの初期化
    weight2d_ = util::alloc<double>(filter_num_, channel_, filter_h_, filter_w_);
    weight_ = new double**[2];
    weight_[0] = util::alloc<double>(pool_output_size_, hidden_size_);
    weight_[1] = util::alloc<double>(hidden_size_, category_num_);

    for (int i = 0; i < filter_num_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < filter_h_; ++k) {
                for (int l = 0; l < filter_w_; ++l) {
                    weight2d_[i][j][k][l] = weight_init_std_ * util::randn<double>();
                }
            }
        }
    }

    for (int i = 0; i < pool_output_size_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            weight_[0][i][j] = weight_init_std_ * util::randn<double>();
        }
    }

    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < category_num_; ++j) {
            weight_[1][i][j] = weight_init_std_ * util::randn<double>();
        }
    }

    // バイアスの初期化
    bias_ = new double*[3];
    bias_[0] = util::alloc<double>(filter_num_);
    bias_[1] = util::alloc<double>(hidden_size_);
    bias_[2] = util::alloc<double>(category_num_);

    // 重みの勾配
    d_weight2d_ = util::alloc<double>(filter_num_, channel_, filter_h_, filter_w_);
    d_weight_ = new double**[2];
    d_weight_[0] = util::alloc<double>(hidden_size_, category_num_);
    d_weight_[1] = util::alloc<double>(pool_output_size_, hidden_size_);

    // バイアスの勾配
    d_bias_ = new double*[3];
    d_bias_[0] = util::alloc<double>(category_num_);
    d_bias_[1] = util::alloc<double>(hidden_size_);
    d_bias_[2] = util::alloc<double>(filter_num_);

    // 中間ユニットの初期化
    x2d_ = new double****[3];
    x2d_[0] = util::alloc<double>(batch_size_, filter_num_, conv_output_h_, conv_output_w_);
    x2d_[1] = util::alloc<double>(batch_size_, filter_num_, conv_output_h_, conv_output_w_);
    x2d_[2] = util::alloc<double>(batch_size_, filter_num_, pool_output_h_, pool_output_w_);

    x_ = new double**[2];
    x_[0] = util::alloc<double>(batch_size_, hidden_size_);
    x_[1] = util::alloc<double>(batch_size_, hidden_size_);

    output_ = util::alloc<double>(batch_size_, category_num_);

    dout2d_ = new double****[4];
    dout2d_[0] = util::alloc<double>(batch_size_, filter_num_, pool_output_h_, pool_output_w_);
    dout2d_[1] = util::alloc<double>(batch_size_, filter_num_, conv_output_h_, conv_output_w_);
    dout2d_[2] = util::alloc<double>(batch_size_, filter_num_, conv_output_h_, conv_output_w_);
    dout2d_[3] = util::alloc<double>(batch_size_, channel_, height_, width_);

    dout_ = new double**[3];
    dout_[0] = util::alloc<double>(batch_size_, category_num_);
    dout_[1] = util::alloc<double>(batch_size_, hidden_size_);
    dout_[2] = util::alloc<double>(batch_size_, hidden_size_);
}

SimpleConvNet::~SimpleConvNet()
{
    delete layer1_;
    delete layer2_;
    delete layer3_;
    delete layer4_;
    delete layer5_;
    delete layer6_;
    delete last_layer_;

    for (int i = 0; i < 6; ++i) {
        delete optim_[i];
    }
    delete[] optim_;

    util::free(weight2d_, filter_num_, channel_, filter_h_);
    util::free(weight_[0], pool_output_size_);
    util::free(weight_[1], hidden_size_);
    delete[] weight_;

    util::free(bias_, 3);

    util::free(d_weight2d_, filter_num_, channel_, filter_h_);
    util::free(d_weight_[0], hidden_size_);
    util::free(d_weight_[1], pool_output_size_);
    delete[] d_weight_;

    util::free(d_bias_, 3);

    util::free(x2d_[0], batch_size_, filter_num_, conv_output_h_);
    util::free(x2d_[1], batch_size_, filter_num_, conv_output_h_);
    util::free(x2d_[2], batch_size_, filter_num_, pool_output_h_);
    delete[] x2d_;

    util::free(x_[0], batch_size_);
    util::free(x_[1], batch_size_);
    delete[] x_;

    util::free(output_, batch_size_);

    util::free(dout2d_[0], batch_size_, filter_num_, pool_output_h_);
    util::free(dout2d_[1], batch_size_, filter_num_, conv_output_h_);
    util::free(dout2d_[2], batch_size_, filter_num_, conv_output_h_);
    util::free(dout2d_[3], batch_size_, channel_, height_);
    delete[] dout2d_;

    util::free(dout_[0], batch_size_);
    util::free(dout_[1], batch_size_);
    util::free(dout_[2], batch_size_);
    delete[] dout_;
}

void SimpleConvNet::predict(double const * const * const * const *input,
                            double **output)
{
    // Conv2d
    layer1_->forward(input, weight2d_, bias_[0], x2d_[0]);
    // ReLU
    layer2_->forward(x2d_[0], x2d_[1]);
    // MaxPool2d
    layer3_->forward(x2d_[1], x2d_[2]);
    // Linear
    layer4_->forward(x2d_[2], weight_[0], bias_[1], x_[0]);
    // ReLU
    layer5_->forward(x_[0], x_[1]);
    // Linear
    layer6_->forward(x_[1], weight_[1], bias_[2], output);
}

double SimpleConvNet::loss(double const * const * const * const *input,
                           double const * const *criterion)
{
    predict(input, output_);

    // Softmax With Cross Entropy
    return last_layer_->forward(output_, criterion);
}

void SimpleConvNet::gradient(double const * const * const * const *input,
                             double const * const *criterion)
{
    // forward
    //loss(input, criterion);

    // backward
    last_layer_->backward(criterion, dout_[0]);
    // Linear
    layer6_->backward(dout_[0], d_weight_[0], d_bias_[0], dout_[1]);
    // ReLU
    layer5_->backward(dout_[1], dout_[2]);
    // Linear
    layer4_->backward(dout_[2], d_weight_[1], d_bias_[1], dout2d_[0]);
    // MaxPool2d
    layer3_->backward(dout2d_[0], dout2d_[1]);
    // ReLU
    layer2_->backward(dout2d_[1], dout2d_[2]);
    // Conv2d
    layer1_->backward(dout2d_[2], d_weight2d_, d_bias_[2], dout2d_[3]);
}

void SimpleConvNet::update()
{
    /*optim_->update(weight2d_, d_weight2d_, lr, filter_num_, channel_, filter_h_, filter_w_);
    optim_->update(bias_[0], d_bias_[2], lr, filter_num_);

    optim_->update(weight_[0], d_weight_[1], lr, pool_output_size_, hidden_size_);
    optim_->update(bias_[1], d_bias_[1], lr, hidden_size_);

    optim_->update(weight_[1], d_weight_[0], lr, hidden_size_, category_num_);
    optim_->update(bias_[2], d_bias_[0], lr, category_num_);*/

    optim_[0]->update(weight2d_, d_weight2d_);
    optim_[1]->update(bias_[0], d_bias_[2]);
    optim_[2]->update(weight_[0], d_weight_[1]);
    optim_[3]->update(bias_[1], d_bias_[1]);
    optim_[4]->update(weight_[1], d_weight_[0]);
    optim_[5]->update(bias_[2], d_bias_[0]);
}
