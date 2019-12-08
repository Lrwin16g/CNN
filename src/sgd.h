#ifndef _SGD_H_
#define _SGD_H_

class SGD
{
public:
    SGD();
    ~SGD();
    void update(double ****params, double const * const * const * const *grads,
                double lr, int dim_1, int dim_2, int dim_3, int dim_4);
    void update(double **params, double const * const *grads,
                double lr, int dim_1, int dim_2);
    void update(double *params, const double *grads, double lr, int dim);
};

#endif