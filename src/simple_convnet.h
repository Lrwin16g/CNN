#ifndef _SIMPLE_CONVNET_H_
#define _SIMPLE_CONVNET_H_

#include "conv2d.h"
#include "relu.h"
#include "maxpool2d.h"
#include "linear.h"
#include "softmax.h"

class SimpleConvNet
{
public:
    SimpleConvNet(int batch_size, int channel, int height, int width,
                  int filter_num, int filter_h, int filter_w, int stride, int pad,
                  int hidden_size, int category_num, double weight_init_std);
    ~SimpleConvNet();

    void predict(double const * const * const * const *input, double **output);
    double loss(double const * const * const * const *input, double const * const *criterion);
    void gradient(double const * const * const * const *input, double const * const *criterion);

private:
    int batch_size_;
    int channel_;
    int height_;
    int width_;
    int filter_num_;
    int filter_h_;
    int filter_w_;
    int stride_;
    int pad_;
    int hidden_size_;
    int category_num_;
    double weight_init_std_;

    int conv_output_h_;
    int conv_output_w_;
    int pool_h_;
    int pool_w_;
    int pool_stride_;
    int pool_output_h_;
    int pool_output_w_;
    int pool_output_size_;

    Conv2d *layer1_;
    ReLU *layer2_;
    MaxPool2d *layer3_;
    Linear *layer4_;
    ReLU *layer5_;
    Linear *layer6_;
    SoftmaxWithLoss *last_layer_;

    double ****weight2d_;
    double ***weight_;
    double **bias_;

    double ****d_weight2d_;
    double ***d_weight_;
    double **d_bias_;

    double *****x2d_;
    double ***x_;
    double **output_;

    double *****dout2d_;
    double ***dout_;
};

#endif