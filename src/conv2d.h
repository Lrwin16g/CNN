#ifndef _CONV2D_H_
#define _CONV2D_H_

class Conv2d
{
public:
    Conv2d(int batch_size, int channel, int height, int width,
           int filter_num, int filter_h, int filter_w, int stride, int pad);
    ~Conv2d();

    void forward(double const * const * const * const *input,
                 double const * const * const * const *weight,
                 const double *bias,
                 double ****output);
    void backward(double const * const * const * const *dout, double ****dx);

private:
    void im2col(double const * const * const * const *src, double **dst);
    void col2im(double const * const *src, double ****dst);
    
    int batch_size_;
    int channel_;
    int height_;
    int width_;
    int filter_num_;
    int filter_h_;
    int filter_w_;
    int stride_;
    int pad_;
    int output_h_;
    int output_w_;
    int input_col_h_;
    int input_col_w_;
    int weight_col_h_;
    int weight_col_w_;
    int tmp_col_h_;
    int tmp_col_w_;
    int output_col_h_;
    int output_col_w_;

    double **input_col_;
    double **input_col_T_;
    double **dx_col_;
    double **weight_col_;
    double **weight_col_T_;
    double **tmp_col_;
    double **output_col_;
    double **dout_col_;
    double ****d_weight_;
    double *d_bias_;
    double **d_weight_col_;
};

#endif