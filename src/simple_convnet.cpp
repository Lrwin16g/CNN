#include "simple_convnet.h"
#include "utils.h"

SimpleConvNet::SimpleConvNet(int batch_size, int channel, int height, int width,
                             int filter_num, int filter_h, int filter_w, int stride, int pad,
                             int hidden_size, int category_num, double weight_init_std)
    : batch_size_(batch_size), channel_(channel), height_(height), width_(width),
      filter_num_(filter_num), filter_h_(filter_h), filter_w_(filter_w), stride_(stride),
      pad_(pad), hidden_size_(hidden_size), category_num_(category_num), weight_init_std_(weight_init_std),
      conv_output_h_(0), conv_output_w_(0), pool_h_(0), pool_w_(0), pool_stride_(0),
      pool_output_h_(0), pool_output_w_(0), pool_output_size_(0),
      layer1_(NULL), layer2_(NULL), layer3_(NULL), layer4_(NULL), layer5_(NULL), layer6_(NULL),
      last_layer_(NULL), weight2d_(NULL), weight_(NULL), bias_(NULL), x2d_(NULL), x_(NULL),
      output_(NULL)
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

    // 重みの初期化
    weight2d_ = util::alloc<double>(filter_num_, channel_, filter_h_, filter_w_);
    weight_ = new double**[2];
    weight_[0] = util::alloc<double>(pool_output_size_, hidden_size_);
    weight_[1] = util::alloc<double>(hidden_size_, category_num_);

    // バイアスの初期化
    bias_ = new double*[3];
    bias_[0] = util::alloc<double>(filter_num_);
    bias_[1] = util::alloc<double>(hidden_size_);
    bias_[2] = util::alloc<double>(category_num_);

    // 中間ユニットの初期化
    x2d_ = new double****[3];
    x2d_[0] = util::alloc<double>(batch_size_, filter_num_, conv_output_h_, conv_output_w_);
    x2d_[1] = util::alloc<double>(batch_size_, filter_num_, conv_output_h_, conv_output_w_);
    x2d_[2] = util::alloc<double>(batch_size_, filter_num_, pool_output_h_, pool_output_w_);

    x_ = new double**[2];
    x_[0] = util::alloc<double>(batch_size_, hidden_size_);
    x_[1] = util::alloc<double>(batch_size_, hidden_size_);

    output_ = util::alloc<double>(batch_size_, category_num_);
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

    util::free(weight2d_, filter_num_, channel_, filter_h_);
    util::free(weight_[0], pool_output_size_);
    util::free(weight_[1], hidden_size_);
    delete[] weight_;

    util::free(bias_, 3);

    util::free(x2d_[0], batch_size_, filter_num_, conv_output_h_);
    util::free(x2d_[1], batch_size_, filter_num_, conv_output_h_);
    util::free(x2d_[2], batch_size_, filter_num_, pool_output_h_);
    delete[] x2d_;

    util::free(x_[0], batch_size_);
    util::free(x_[1], batch_size_);
    delete[] x_;

    util::free(output_, batch_size_);
}

void SimpleConvNet::forward(double const * const * const * const *input,
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
    forward(input, output_);

    // Softmax With Cross Entropy
    return last_layer_->forward(output_, criterion);
}

