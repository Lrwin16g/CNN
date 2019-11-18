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
}

#endif