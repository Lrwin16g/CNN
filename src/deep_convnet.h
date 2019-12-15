#ifndef _DEEP_CONVNET_H_
#define _DEEP_CONVNET_H_

#include "utils.h"
#include "conv2d.h"
#include "relu.h"
#include "maxpool2d.h"
#include "linear.h"
#include "dropout.h"
#include "softmax.h"
#include "adam.h"

class DeepConvNet
{
public:
    DeepConvNet(int batch_size, int input_dim[], int hidden_size, int output_size);
    ~DeepConvNet();

    void predict(double const * const * const * const *input, double **output);
    double loss(double const * const * const * const *input,
                double const * const *criterion);
    void gradient(double const * const *criterion);
    void update();
    void setTrainFlg(bool flg) {train_flg_ = flg;}

private:
    int batch_size_;
    int input_dim_[3];
    int hidden_size_;
    int output_size_;
    int pool_h_;
    int pool_w_;
    int pool_stride_;
    int pool_output_size_;
    bool train_flg_;

    util::ConvParam **conv_param_;
    int *conv_output_h_;
    int *conv_output_w_;

    Conv2d *layer1_, *layer3_, *layer6_, *layer8_, *layer11_, *layer13_;
    ReLU *layer2_, *layer4_, *layer7_, *layer9_, *layer12_, *layer14_,
         *layer17_;
    MaxPool2d *layer5_, *layer10_, *layer15_;
    Linear *layer16_, *layer19_;
    Dropout *layer18_, *layer20_;
    SoftmaxWithLoss *last_layer_;

    Adam **optim_;

    double *****weight2d_, *****d_weight2d_;
    double ***weight_, ***d_weight_;
    double **bias_, **d_bias_;
    double *****x2d_, *****dout2d_;
    double ***x_, ***dout_;
    double **output_;
};

#endif