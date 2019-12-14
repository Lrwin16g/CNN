#ifndef _UTILS_H_
#define _UTILS_H_

#include <cmath>
#include <cstdlib>

namespace util
{
    struct ConvParam
    {
        int filter_num;
        int filter_h;
        int filter_w;
        int stride;
        int pad;

        ConvParam(int fn, int fh, int fw, int st, int pd)
            : filter_num(fn), filter_h(fh), filter_w(fw),
              stride(st), pad(pd)
        {
        }
    };

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
    Type*** alloc(int dim_1, int dim_2, int dim_3)
    {
        Type ***data = new Type**[dim_1];
        for (int i = 0; i < dim_1; ++i) {
            data[i] = new Type*[dim_2];
            for (int j = 0; j < dim_2; ++j) {
                data[i][j] = new Type[dim_3];
                for (int k = 0; k < dim_3; ++k) {
                    data[i][j][k] = 0.0;
                }
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
    void free(Type ***data, int dim_1, int dim_2)
    {
        for (int i = 0; i < dim_1; ++i) {
            for (int j = 0; j < dim_2; ++j) {
                delete[] data[i][j];
            }
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

    template<typename Type>
    Type randu()
    {
        return (static_cast<Type>(rand()) + 1.0) / (static_cast<Type>(RAND_MAX) + 2.0);
    }

    template<typename Type>
    Type randu(Type min, Type max)
    {
        Type z = static_cast<Type>(rand()) / static_cast<Type>(RAND_MAX) * (max - min);
        return z + min;
    }

    template<typename Type>
    Type randn(Type mean = 0.0, Type stddev = 1.0)
    {
        Type z = sqrt(-2.0 * log(randu<Type>())) * sin(2.0 * M_PI * randu<Type>());
        return mean + stddev * z;
    }

    template<typename Type>
    Type sigmoid(Type x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    void transpose(double const * const *src, double **dst, int rows, int cols);

    void dot(double const * const *lhs, double const * const *rhs, double **dst,
             int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols);
}

#endif