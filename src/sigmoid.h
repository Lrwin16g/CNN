#ifndef _SIGMOID_H_
#define _SIGMOID_H_

class Sigmoid
{
public:
    Sigmoid(int batch_size, int input_size, int channel, int height, int width);
    ~Sigmoid();

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

    double ****out2d_;
    double **out_;
};

#endif