#ifndef _BATCHNORM_H_
#define _BATCHNORM_H_

class BatchNorm
{
public:
    BatchNorm(int batch_size, int channel, int height, int width, double momentum);
    BatchNorm(int batch_size, int input_size, double momentum);
    ~BatchNorm();

    void forward(double const * const * const * const *input,
                 double const * const * const *gamma,
                 double const * const * const *beta,
                 double ****output, bool train_flg);
    void forward(double const * const *input, const double *gamma,
                 const double *beta, double **output, bool train_flg);
    void backward(double const * const * const * const *dout,
                  double const * const * const *gamma,
                  double ***d_gamma, double ***d_beta, double ****dx);
    void backward(double const * const *dout, const double *gamma, double *d_gamma,
                  double *d_beta, double **dx);

private:
    int batch_size_;
    int input_size_;
    int channel_;
    int height_;
    int width_;
    double momentum_;

    double ***mean2d_, ***var2d_, ***std2d_, ***run_mean2d_, ***run_var2d_,
           ***d_mean2d_, ***d_var2d_, ***d_std2d_;
    double ****input_c2d_, ****input_n2d_, ****dx_c2d_, ****dx_n2d_;
    double *mean_, *var_, *std_, *run_mean_, *run_var_,
           *d_mean_, *d_var_, *d_std_;
    double **input_c_, **input_n_, **dx_c_, **dx_n_;
};

#endif