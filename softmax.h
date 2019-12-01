#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

class SoftmaxWithLoss
{
public:
    SoftmaxWithLoss(int batch_size, int category_num);
    ~SoftmaxWithLoss();

    double forward(double const * const *input, double const * const *criterion);
    void backward(double const * const *criterion, double **dx);

private:
    int batch_size_;
    int category_num_;

    double **output_;
};

#endif