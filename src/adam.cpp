#include "adam.h"
#include "utils.h"

#include <cmath>

Adam::Adam(double lr, double beta_1, double beta_2, int dim_1, int dim_2, int dim_3, int dim_4)
    : lr_(lr), beta_1_(beta_1), beta_2_(beta_2), iter_(0), dim_1_(dim_1), dim_2_(dim_2),
      dim_3_(dim_3), dim_4_(dim_4), momentum4d_(NULL), velocity4d_(NULL), momentum2d_(NULL),
      velocity2d_(NULL), momentum_(NULL), velocity_(NULL)
{
    if (dim_2_ == 0 && dim_3_ == 0 && dim_4_ == 0) {
        momentum_ = util::alloc<double>(dim_1_);
        velocity_ = util::alloc<double>(dim_1_);
    } else if (dim_3_ == 0 && dim_4_ == 0) {
        momentum2d_ = util::alloc<double>(dim_1_, dim_2_);
        velocity2d_ = util::alloc<double>(dim_1_, dim_2_);
    } else {
        momentum4d_ = util::alloc<double>(dim_1_, dim_2_, dim_3_, dim_4_);
        velocity4d_ = util::alloc<double>(dim_1_, dim_2_, dim_3_, dim_4_);
    }
}

Adam::~Adam()
{
    if (dim_2_ == 0 && dim_3_ == 0 && dim_4_ == 0) {
        util::free(momentum_);
        util::free(velocity_);
    } else if (dim_3_ == 0 && dim_4_ == 0) {
        util::free(momentum2d_, dim_1_);
        util::free(velocity2d_, dim_1_);
    } else {
        util::free(momentum4d_, dim_1_, dim_2_, dim_3_);
        util::free(velocity4d_, dim_1_, dim_2_, dim_3_);
    }
}

void Adam::update(double ****params, double const * const * const * const *grads)
{
    iter_++;
    double lr_t = lr_ * sqrt(1.0 - pow(beta_2_, iter_)) / (1.0 - pow(beta_1_, iter_));

    for (int i = 0; i < dim_1_; ++i) {
        for (int j = 0; j < dim_2_; ++j) {
            for (int k = 0; k < dim_3_; ++k) {
                for (int l = 0; l < dim_4_; ++l) {
                    momentum4d_[i][j][k][l] += (1.0 - beta_1_) * (grads[i][j][k][l] - momentum4d_[i][j][k][l]);
                    velocity4d_[i][j][k][l] += (1.0 - beta_2_) * (pow(grads[i][j][k][l], 2.0) - velocity4d_[i][j][k][l]);

                    params[i][j][k][l] -= lr_t * momentum4d_[i][j][k][l] / (sqrt(velocity4d_[i][j][k][l]) + 1e-7);
                }
            }
        }
    }
}

void Adam::update(double **params, double const * const *grads)
{
    iter_++;
    double lr_t = lr_ * sqrt(1.0 - pow(beta_2_, iter_)) / (1.0 - pow(beta_1_, iter_));

    for (int i = 0; i < dim_1_; ++i) {
        for (int j = 0; j < dim_2_; ++j) {
            momentum2d_[i][j] += (1.0 - beta_1_) * (grads[i][j] - momentum2d_[i][j]);
            velocity2d_[i][j] += (1.0 - beta_2_) * (pow(grads[i][j], 2.0) - velocity2d_[i][j]);

            params[i][j] -= lr_t * momentum2d_[i][j] / (sqrt(velocity2d_[i][j]) + 1e-7);
        }
    }
}

void Adam::update(double *params, const double *grads)
{
    iter_++;
    double lr_t = lr_ * sqrt(1.0 - pow(beta_2_, iter_)) / (1.0 - pow(beta_1_, iter_));

    for (int i = 0; i < dim_1_; ++i) {
        momentum_[i] += (1.0 - beta_1_) * (grads[i] - momentum_[i]);
        velocity_[i] += (1.0 - beta_2_) * (pow(grads[i], 2.0) - velocity_[i]);

        params[i] -= lr_t * momentum_[i] / (sqrt(velocity_[i]) + 1e-7);
    }
}