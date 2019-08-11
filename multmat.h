#ifndef _multmat_H_
#define _multmat_H_

#include <vector>

struct multmat
{
	multmat();
	multmat(int s, int c, int h, int w);
	multmat operator+(const multmat &cmp);
	multmat operator*(const multmat &cmp);
	multmat& operator=(const multmat &cmp);

	int size;
	int channel;
	int height;
	int width;

	std::vector<std::vector<std::vector<std::vector<double> > > > data;
};

multmat transpose(const multmat &src);

multmat padding(const multmat &src, int pad);

multmat im2col(const multmat &src, const multmat &filter, int stride);
multmat im2col(const multmat &src, int filter_height, int filter_width, int stride, int pad);

multmat col2im(const multmat &src, int size, int height, int width);
multmat col2im(const multmat &src, int size, int channel, int height, int width,
			   int filter_height, int filter_width, int stride, int pad);

//multmat filt2col(const multmat &src, int size, int channel);
//multmat filt2col(const multmat &src, int channel);
multmat filt2col(const multmat &src);
multmat col2filt(const multmat &src, int channel, int height, int width);

multmat bias2col(const multmat &src, int height);

multmat dout2col(const multmat &src);

#endif
