#ifndef _MAXPOOL2D_H_
#define _MAXPOOL2D_H_

class MaxPool2d
{
public:
    MaxPool2d(int batch_size, int channel, int height, int width,
              int pool_h, int pool_w, int stride);
    ~MaxPool2d();

    void forward(double const * const * const * const *input, double ****output);
    void backward(double const * const * const * const *dout, double ****dx);

private:
    void im2col(double const * const * const * const *src, double **dst);
    void col2im(double const * const *src, double ****dst);

    int batch_size_;
    int channel_;
    int height_;
    int width_;
    int pool_h_;
    int pool_w_;
    int stride_;
    int output_h_;
    int output_w_;
    int input_col_h_;
    int input_col_w_;

    double **input_col_;
    double **dmax_col_;
    double *output_col_;
    double *dout_col_;
    int *arg_max_;
};

#endif