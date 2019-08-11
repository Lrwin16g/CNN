#include "layer.h"

#include <iostream>

Convolution::Convolution(int filter_num, int input_channel, int filter_height, int filter_width,
						 int stride, int pad)
	: filter_(filter_num, input_channel, filter_height, filter_width),
	  bias_(filter_num, 1, 1, 1),
	  grad_filter_(filter_num, input_channel, filter_height, filter_width),
	  grad_bias_(filter_num, 1, 1, 1),
	  input_(), input_col_(), filter_col_(),
	  filter_num_(filter_num), input_channel_(input_channel),
	  filter_height_(filter_height), filter_width_(filter_width),
	  stride_(stride), pad_(pad)
{
	int cnt = 0;
	for (int s = 0; s < filter_.size; ++s) {
		for (int c = 0; c < filter_.channel; ++c) {
			for (int y = 0; y < filter_.height; ++y) {
				for (int x = 0; x < filter_.width; ++x) {
					filter_.data[s][c][y][x] = 1;
					cnt++;
				}
			}
		}
		bias_.data[s][0][0][0] = 0;
	}
}

Convolution::~Convolution()
{
}

multmat Convolution::forward(const multmat &input)
{
	input_ = input;
	int output_height = 1 + (input.height + 2 * pad_ - filter_.height) / stride_;
	int output_width = 1 + (input.width + 2 * pad_ - filter_.width) / stride_;

	// image to column
	input_col_ = im2col(input, filter_.height, filter_.width, stride_, pad_);

	// filter to column
	filter_col_ = filt2col(filter_);

	// bias to column
	multmat bias_col = bias2col(bias_, input.size * output_height * output_width);

	// convolution
	multmat output_col = input_col_ * filter_col_ + bias_col;

	// column to image
	return col2im(output_col, input.size, output_height, output_width);
}

multmat Convolution::backward(const multmat &dout)
{
	// dout to column
	multmat dout_col = dout2col(dout);

	// calculate bias gradient
	for (int s = 0; s < filter_num_; ++s) {
		grad_bias_.data[s][0][0][0] = 0.0;
		for (int i = 0; i < dout.height; ++i) {
			grad_bias_.data[s][0][0][0] += dout.data[0][0][i][s];
		}
	}

	// input column transpose
	multmat input_col_trans = transpose(input_col_);

	// calculate filter gradient
	multmat grad_filter_col = input_col_trans * dout_col;
	grad_filter_ = col2filt(grad_filter_col, input_channel_, filter_height_, filter_width_);

	// filter column transpose
	multmat filter_col_trans = transpose(filter_col_);

	// calculate back propagation
	multmat dx_col = dout_col * filter_col_trans;

	// column to image
	return col2im(dx_col, input_.size, input_.channel, input_.height,
				  input_.width, filter_height_, filter_width_, stride_, pad_);
}
