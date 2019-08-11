#ifndef _LAYER_H_
#define _LAYER_H_

#include "multmat.h"

class Convolution
{
public:
	Convolution(int filter_num, int input_channel, int filter_height,
				int filter_width, int stride, int pad);
	~Convolution();

	multmat forward(const multmat &input);
	multmat backward(const multmat &dout);

private:
	multmat filter_;
	multmat bias_;
	multmat grad_filter_;
	multmat grad_bias_;
	multmat input_;
	multmat input_col_;
	multmat filter_col_;

	int filter_num_;
	int input_channel_;
	int filter_height_;
	int filter_width_;
	int stride_;
	int pad_;
};

#endif
