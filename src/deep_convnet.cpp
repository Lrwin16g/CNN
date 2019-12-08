#include "deep_convnet.h"
#include "utils.h"

DeepConvNet::DeepConvNet(int batch_size, int input_dim[3], int hidden_size, int output_size)
    : batch_size_(batch_size), input_dim_(input_dim), hidden_size_(hidden_size),
      output_size_(output_size), pool_h_(2), pool_w_(2), pool_stride_(2),
      layer1_(NULL), layer2_(NULL), layer3_(NULL), layer4_(NULL), layer5_(NULL),
      layer6_(NULL), layer7_(NULL), layer8_(NULL), layer9_(NULL), layer10_(NULL),
      layer11_(NULL), layer12_(NULL), layer13_(NULL), layer14_(NULL), layer15_(NULL),
      layer16_(NULL), layer17_(NULL), layer18_(NULL), layer19_(NULL), layer20_(NULL),
      last_layer_(NULL), optim_(NULL),
      weight2d_(NULL), weight_(NULL), bias_(NULL), x2d_(NULL), x_(NULL), output_(NULL),
      d_weight2d_(NULL), d_weight_(NULL), d_bias_(NULL), dout2d_(NULL), dout_(NULL),
      conv_param_(NULL), conv_output_h_(NULL), conv_output_w_(NULL)
{
    conv_param_ = new ConvParam[6];
    conv_param_[0] = new ConvParam(16, 3, 3, 1, 1);
    conv_param_[1] = new ConvParam(16, 3, 3, 1, 1);
    conv_param_[2] = new ConvParam(32, 3, 3, 1, 1);
    conv_param_[3] = new ConvParam(32, 3, 3, 1, 2);
    conv_param_[4] = new ConvParam(64, 3, 3, 1, 1);
    conv_param_[5] = new ConvParam(64, 3, 3, 1, 1);

    // 1: Conv2d
    layer1_ = new Conv2d(batch_size_, input_dim_[0], input_dim_[1], input_dim_[2],
                         conv_param_[0].filter_num, conv_param_[0].filter_h,
                         conv_param_[0].filter_w, conv_param_[0].stride,
                         conv_param_[0].pad);
    // 2: ReLU
    conv_output_h_[0] = (input_dim_[1] + 2 * conv_param_[0].pad - conv_param_[0].filter_h)
                        / conv_param_[0].stride + 1;
    conv_output_w_[0] = (input_dim_[2] + 2 * conv_param_[0].pad - conv_param_[0].filter_w)
                        / conv_param_[0].stride + 1;
    layer2_ = new ReLU(batch_size_, 0, conv_param_[0].filter_num, conv_output_h_[0],
                       conv_output_w_[0]);
    // 3: Conv2d
    layer3_ = new Conv2d(batch_size_, conv_param_[0].filter_num, conv_output_h_[0],
                         conv_output_w_[0], conv_param_[1].filter_num, conv_param_[1].filter_h,
                         conv_param_[1].filter_w, conv_param_[1].stride, conv_param_[1].pad);
    // 4: ReLU
    conv_output_h_[1] = (conv_output_h_[0] + 2 * conv_param_[1].pad - conv_param_[1].filter_h)
                        / conv_param_[1].stride + 1;
    conv_output_w_[1] = (conv_output_w_[0] + 2 * conv_param_[1].pad - conv_param_[1].filter_w)
                        / conv_param_[1].stride + 1;
    layer4_ = new ReLU(batch_size_, 0, conv_param_[1].filter_num, conv_output_h_[1],
                       conv_output_w_[1]);
    // 5: MaxPool2d
    layer5_ = new MaxPool2d(batch_size_, conv_param_[1].filter_num, conv_output_h_[1],
                            conv_output_w_[1], pool_h_, pool_w_, pool_stride_);
    // 6: Conv2d
    conv_output_h_[2] = (conv_output_h_[1] - pool_h_) / pool_stride_ + 1;
    conv_output_w_[2] = (conv_output_w_[1] - pool_w_) / pool_stride_ + 1;
    layer6_ = new Conv2d(batch_size_, conv_param_[1].filter_num, conv_output_h_[2],
                         conv_output_w_[2], conv_param_[2].filter_num, conv_param_[2].filter_h,
                         conv_param_[2].filter_w, conv_param_[2].stride, conv_param_[2].pad);
    // 7: ReLU
    conv_output_h_[3] = (conv_output_h_[2] + 2 * conv_param_[2].pad - conv_param_[2].filter_h)
                        / conv_param_[2].stride + 1;
    conv_output_w_[3] = (conv_output_w_[2] + 2 * conv_param_[2].pad - conv_param_[2].filter_w)
                        / conv_param_[2].stride + 1;
    layer7_ = new ReLU(batch_size_, 0, conv_param_[2].filter_num, conv_output_h_[3],
                       conv_output_w_[3]);
    // 8: Conv2d
    layer8_ = new Conv2d(batch_size_, conv_param_[2].filter_num, conv_output_h_[3],
                         conv_output_w_[3], conv_param_[3].filter_num, conv_param_[3].filter_h,
                         conv_param_[3].filter_w, conv_param_[3].stride, conv_param_[3].pad);
    // 9: ReLU
    conv_output_h_[4] = (conv_output_h_[3] + 2 * conv_param_[3].pad - conv_param_[3].filter_h)
                        / conv_param_[3].stride + 1;
    conv_output_w_[4] = (conv_output_w_[3] + 2 * conv_param_[3].pad - conv_param_[3].filter_w)
                        / conv_param_[3].stride + 1;
    layer9_ = new ReLU(batch_size_, 0, conv_param_[3].filter_num, conv_output_h_[4],
                       conv_output_w_[4]);
    // 10: MaxPool2d
    layer10_ = new MaxPool2d(batch_size_, conv_param_[3].filter_num, conv_output_h_[4],
                             conv_output_w_[4], pool_h_, pool_w_, pool_stride_);
    // 11: Conv2d
    conv_output_h_[5] = (conv_output_h_[4] - pool_h_) / pool_stride_ + 1;
    conv_output_w_[5] = (conv_output_w_[4] - pool_w_) / pool_stride_ + 1;
    
    // 12: ReLU
    // 13: Conv2d
    // 14: ReLU
    // 15: MaxPool2d
    // 16: Linear
    // 17: ReLU
    // 18: Dropout
    // 19: Linear
    // 20: Dropout
}

DeepConvNet::~DeepConvNet()
{
    for (int i = 0; i < 6; ++i) {
        delete conv_param_[i];
    }
    delete[] conv_param_;
}

void DeepConvNet::predict(double const * const * const * const *input,
                          double **output)
{
    // 1: Conv2d
    layer1_->forward(input, weight2d_[0], bias_[0], x2d_[0]);
    // 2: ReLU
    layer2_->forward(x2d_[0], x2d_[1]);
    // 3: Conv2d
    layer3_->forward(x2d_[1], weight2d_[1], bias_[1], x2d_[2]);
    // 4: ReLU
    layer4_->forward(x2d_[2], x2d_[3]);
    // 5: MaxPool2d
    layer5_->forward(x2d_[3], x2d_[4]);
    // 6: Conv2d
    layer6_->forward(x2d_[4], weight2d_[2], bias_[2], x2d_[5]);
    // 7: ReLU
    layer7_->forward(x2d_[5], x2d_[6]);
    // 8: Conv2d
    layer8_->forward(x2d_[6], weight2d_[3], bias_[3], x2d_[7]);
    // 9: ReLU
    layer9_->forward(x2d_[7], x2d_[8]);
    // 10: MaxPool2d
    layer10_->forward(x2d_[8], x2d_[9]);
    // 11: Conv2d
    layer11_->forward(x2d_[9], weight2d_[4], bias_[4], x2d_[10]);
    // 12: ReLU
    layer12_->forward(x2d_[10], x2d_[11]);
    // 13: Conv2d
    layer13_->forward(x2d_[11], weight2d_[5], bias_[5], x2d_[12]);
    // 14: ReLU
    layer14_->forward(x2d_[12], x2d_[13]);
    // 15: MaxPool2d
    layer15_->forward(x2d_[13], x2d_[14]);
    // 16: Linear
    layer16_->forward(x2d_[14], weight_[0], bias_[6], x_[0]);
    // 17: ReLU
    layer17_->forward(x_[0], x_[1]);
    // 18: Dropout
    layer18_->forward(x_[1], x_[2], train_flg_);
    // 19: Linear
    layer19_->forward(x_[2], weight_[1], bias_[7], x_[3]);
    // 20: Dropout
    layer20_->forward(x_[3], output, train_flg_);
}

double DeepConvNet::loss(double const * const * const * const *input,
                         double const * const *criterion)
{
    predict(input, output_);

    // 21: Softmax with Cross Entropy
    return last_layer_->forward(output_, criterion);
}

void DeepConvNet::gradient(double const * const *criterion)
{
    // 21: Softmax with Cross Entropy
    last_layer_->backward(criterion, dout_[0]);
    // 20: Dropout
    layer20_->backward(dout_[0], dout_[1]);
    // 19: Linear
    layer19_->backward(dout_[1], d_weight_[1], d_bias_[7], dout_[2]);
    // 18: Dropout
    layer18_->backward(dout_[2], dout_[3]);
    // 17: ReLU
    layer17_->backward(dout_[3], dout_[4]);
    // 16: Linear
    layer16_->backward(dout_[4], d_weight_[0], d_bias_[6], dout2d_[0]);
    // 15: MaxPool2d
    layer15_->backward(dout2d_[0], dout2d_[1]);
    // 14: ReLU
    layer14_->backward(dout2d_[1], dout2d_[2]);
    // 13: Conv2d
    layer13_->backward(dout2d_[2], d_weight2d_[5], d_bias_[5], dout2d_[3]);
    // 12: ReLU
    layer12_->backward(dout2d_[3], dout2d_[4]);
    // 11: Conv2d
    layer11_->backward(dout2d_[4], d_weight2d_[4], d_bias_[4], dout2d_[5]);
    // 10: MaxPool2d
    layer10_->backward(dout2d_[5], dout2d_[6]);
    // 9: ReLU
    layer9_->backward(dout2d_[6], dout2d_[7]);
    // 8: Conv2d
    layer8_->backward(dout2d_[7], d_weight2d_[3], d_bias_[3], dout2d_[8]);
    // 7: ReLU
    layer7_->backward(dout2d_[8], dout2d_[9]);
    // 6: Conv2d
    layer6_->backward(dout2d_[9], d_weight2d_[2], d_bias_[2], dout2d_[10]);
    // 5: MaxPool2d
    layer5_->backward(dout2d_[10], dout2d_[11]);
    // 4: ReLU
    layer4_->backward(dout2d_[11], dout2d_[12]);
    // 3: Conv2d
    layer3_->backward(dout2d_[12], d_weight2d_[1], d_bias_[1], dout2d_[13]);
    // 2: ReLU
    layer2_->backward(dout2d_[13], dout2d_[14]);
    // 1: Conv2d
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
