#ifndef _UTILS_H_
#define _UTILS_H_

#include <cmath>

namespace util
{
    /*double* alloc(int dim_1);
    double** alloc(int dim_1, int dim_2);
    double**** alloc(int dim_1, int dim_2, int dim_3, int dim_4);

    void free(double *data);
    void free(double **data, int dim_1);
    void free(double ****data, int dim_1, int dim_2, int dim_3);*/

    template<typename Type>
    Type* alloc(int dim_1)
    {
        Type *data = new Type[dim_1];
        for (int i = 0; i < dim_1; ++i) {
            data[i] = 0.0;
        }
        return data;
    }

    template<typename Type>
    Type** alloc(int dim_1, int dim_2)
    {
        Type **data = new Type*[dim_1];
        for (int i = 0; i < dim_1; ++i) {
            data[i] = new Type[dim_2];
            for (int j = 0; j < dim_2; ++j) {
                data[i][j] = 0.0;
            }
        }
        return data;
    }

    template<typename Type>
    Type**** alloc(int dim_1, int dim_2, int dim_3, int dim_4)
    {
        Type ****data = new Type***[dim_1];
        for (int i = 0; i < dim_1; ++i) {
            data[i] = new Type**[dim_2];
            for (int j = 0; j < dim_2; ++j) {
                data[i][j] = new Type*[dim_3];
                for (int k = 0; k < dim_3; ++k) {
                    data[i][j][k] = new Type[dim_4];
                    for (int l = 0; l < dim_4; ++l) {
                        data[i][j][k][l] = 0.0;
                    }
                }
            }
        }
        return data;
    }

    template<typename Type>
    void free(Type *data)
    {
        delete[] data;
        data = NULL;
    }

    template<typename Type>
    void free(Type **data, int dim_1)
    {
        for (int i = 0; i < dim_1; ++i) {
            delete[] data[i];
        }
        delete[] data;
        data = NULL;
    }

    template<typename Type>
    void free(Type ****data, int dim_1, int dim_2, int dim_3)
    {
        for (int i = 0; i < dim_1; ++i) {
            for (int j = 0; j < dim_2; ++j) {
                for (int k = 0; k < dim_3; ++k) {
                    delete[] data[i][j][k];
                }
                delete[] data[i][j];
            }
            delete[] data[i];
        }
        delete[] data;
        data = NULL;
    }

    void transpose(double const * const *src, double **dst, int rows, int cols);

    void dot(double const * const *lhs, double const * const *rhs, double **dst,
             int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols);
}

#endif