#include "sgd.h"

SGD::SGD()
{
}

SGD::~SGD()
{
}

void SGD::update(double ****params, double const * const * const * const *grads,
                 double lr, int dim_1, int dim_2, int dim_3, int dim_4)
{
    for (int i = 0; i < dim_1; ++i) {
        for (int j = 0; j < dim_2; ++j) {
            for (int k = 0; k < dim_3; ++k) {
                for (int l = 0; l < dim_4; ++l) {
                    params[i][j][k][l] -= lr * grads[i][j][k][l];
                }
            }
        }
    }
}

void SGD::update(double **params, double const * const *grads,
                 double lr, int dim_1, int dim_2)
{
    for (int i = 0; i < dim_1; ++i) {
        for (int j = 0; j < dim_2; ++j) {
            params[i][j] -= lr * grads[i][j];
        }
    }
}

void SGD::update(double *params, const double *grads, double lr, int dim)
{
    for (int i = 0; i < dim; ++i) {
        params[i] -= lr * grads[i];
    }
}
