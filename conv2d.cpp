

Conv2d::Conv2d()
{

}

Conv2d::~Conv2d()
{

}

void Conv2d::forward(double const * const *input, double **output, double **weight, double *bias,
                     int rows, int cols)
{
    util::dot(input, weight, tmp_, input_row_, input_col_, weight_row_, weight_col_);
    util::add(tmp_, bias, output, output_row_, output_col_);
}