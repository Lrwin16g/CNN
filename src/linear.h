#ifndef _LINEAR_H_
#define _LINEAR_H_

class Linear
{
public:
    Linear(int batch_size, int input_size, int output_size,
           int channel = 0, int height = 0, int width = 0);
    ~Linear();

    void forward(double const * const * const * const *input,
                 double const * const *weight, const double *bias,
                 double **output);
    void forward(double const * const *input, double const * const *weight,
                 const double *bias, double **output);
    void backward(double const * const *dout, double **d_weight,
                  double *d_bias, double ****dx);
    void backward(double const * const *dout, double **d_weight,
                  double *d_bias, double **dx);

private:
    int batch_size_;
    int input_size_;
    int output_size_;
    int channel_;
    int height_;
    int width_;

    double **input_T_;
    double **weight_T_;
    double **input_col_;
    double **dx_col_;
};

#endif