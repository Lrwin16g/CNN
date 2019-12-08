#ifndef _DROPOUT_H_
#define _DROPOUT_H_

class Dropout
{
public:
    Dropout(int batch_size, int input_size, int channel, int height, int width,
            double ratio);
    ~Dropout();

    void forward(double const * const * const * const *input,
                 double ****output, bool train_flg);
    void forward(double const * const *input, double **output,
                 bool train_flg);
    void backward(double const * const * const * const *dout,
                  double ****dx);
    void backward(double const * const *dout, double **dx);

private:
    int batch_size_;
    int input_size_;
    int channel_;
    int height_;
    int width_;
    double ratio_;

    bool ****mask2d_;
    bool **mask_;
};

#endif