#ifndef _UTILS_H_
#define _UTILS_H_

namespace util
{
    double** alloc(int dim_1, int dim_2);
    double**** alloc(int dim_1, int dim_2, int dim_3, int dim_4);

    void free(double **data, int dim_1);
    void free(double ****data, int dim_1, int dim_2, int dim_3);

    void im2col(double const * const * const * const *src, double **dst,
                int batch_size, int channel, int height, int width,
                int filter_h, int filter_w, int stride, int pad);

    void col2im(double const * const *src, double ****dst,
                int batch_size, int channel, int height, int width,
                int filter_h, int filter_w, int stride, int pad);
}

#endif