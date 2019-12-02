#include "utils.h"

void util::transpose(double const * const *src, double **dst, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j][i] = src[i][j];
        }
    }
}

void util::dot(double const * const *lhs, double const * const *rhs, double **dst,
               int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols)
{
    for (int i = 0; i < lhs_rows; ++i) {
        for (int j = 0; j < rhs_cols; ++j) {
            dst[i][j] = 0.0;
            for (int k = 0; k < lhs_cols; ++k) {
                dst[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }
}
