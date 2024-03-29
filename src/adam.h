#ifndef _ADAM_H_
#define _ADAM_H_

class Adam
{
public:
    Adam(double lr, double beta_1, double beta_2, int dim_1, int dim_2 = 0,
         int dim_3 = 0, int dim_4 = 0);
    ~Adam();
    void update(double ****params, double const * const * const * const *grads);
    void update(double **params, double const * const *grads);
    void update(double *params, const double *grads);

private:
    double lr_;
    double beta_1_;
    double beta_2_;
    int iter_;
    int dim_1_;
    int dim_2_;
    int dim_3_;
    int dim_4_;

    double ****momentum4d_;
    double ****velocity4d_;
    double **momentum2d_;
    double **velocity2d_;
    double *momentum_;
    double *velocity_;
};

#endif