#ifndef _RELU_H_
#define _RELU_H_

class ReLU
{
public:
    ReLU(int batch_size, int input_size, int channel = 0, int height = 0, int width = 0);
    ~ReLU();

    void forward(double const * const * const * const *input,
                 double ****output);
    void forward(double const * const *input, double **output);
    void backward(double const * const * const * const *dout,
                  double ****dx);
    void backward(double const * const *dout, double **dx);

private:
    int batch_size_;
    int input_size_;
    int channel_;
    int height_;
    int width_;

    double **mask_;
    double ****mask_2d_;
};

#endif