#include "multmat.h"
#include <stdexcept>

multmat::multmat()
	: size(0), channel(0), height(0), width(0)
{
}

multmat::multmat(int s, int c, int h, int w)
	: size(s), channel(c), height(h), width(w)
{
	data.resize(size);
	for (int i = 0; i < size; ++i) {
		data[i].resize(channel);
		for (int j = 0; j < channel; ++j) {
			data[i][j].resize(height);
			for (int y = 0; y < height; ++y) {
				data[i][j][y].resize(width);
				for (int x = 0; x < width; ++x) {
					data[i][j][y][x] = 0.0;
				}
			}
		}
	}
}

multmat multmat::operator+(const multmat &cmp)
{
	if (size != cmp.size) {
		throw std::logic_error("multmat.size is not equal");
	}
	if (channel != cmp.channel) {
		throw std::logic_error("multmat.channel is not equal");
	}
	if (height != cmp.height) {
		throw std::logic_error("multmat.height is not equal");
	}
	if (width != cmp.width) {
		throw std::logic_error("multmat.width is not equal");
	}

	multmat dst(size, channel, height, width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					dst.data[s][c][y][x] = data[s][c][y][x] + cmp.data[s][c][y][x];
				}
			}
		}
	}

	return dst;
}

multmat multmat::operator*(const multmat &cmp)
{
	if (size != cmp.size) {
		throw std::logic_error("multmat.size is not equal");
	}
	if (channel != cmp.channel) {
		throw std::logic_error("multmat.channel is not equal");
	}
	if (width != cmp.height) {
		throw std::logic_error("multmat.width is not equal");
	}

	multmat dst(size, channel, height, cmp.width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < cmp.width; ++x) {
					for (int i = 0; i < width; ++i) {
						dst.data[s][c][y][x] += data[s][c][y][i] * cmp.data[s][c][i][x];
					}
				}
			}
		}
	}

	return dst;
}

multmat& multmat::operator=(const multmat &cmp)
{
	size = cmp.size;
	channel = cmp.channel;
	height = cmp.height;
	width = cmp.width;

	data.resize(cmp.size);
	for (int s = 0; s < cmp.size; ++s) {
		data[s].resize(cmp.channel);
		for (int c = 0; c < cmp.channel; ++c) {
			data[s][c].resize(cmp.height);
			for (int y = 0; y < cmp.height; ++y) {
				data[s][c][y].resize(cmp.width);
				for (int x = 0; x < cmp.width; ++x) {
					data[s][c][y][x] = cmp.data[s][c][y][x];
				}
			}
		}
	}

	return *this;
}

multmat transpose(const multmat &src)
{
	multmat dst(src.size, src.channel, src.width, src.height);

	for (int s = 0; s < src.size; ++s) {
		for (int c = 0; c < src.channel; ++c) {
			for (int y = 0; y < src.height; ++y) {
				for (int x = 0; x < src.width; ++x) {
					dst.data[s][c][x][y] = src.data[s][c][y][x];
				}
			}
		}
	}

	return dst;
}

/*multmat multmat::pad(int pad)
{
	int pad_height = height + pad * 2;
	int pad_width = width + pad * 2;

	multmat dst(size, channel, pad_height, pad_width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					dst.data[s][c][y + pad][x + pad] = data[s][c][y][x];
				}
			}
		}
	}

	return dst;
}*/

multmat padding(const multmat &src, int pad)
{
	int height = src.height + pad * 2;
	int width = src.width + pad * 2;

	multmat dst(src.size, src.channel, height, width);

	for (int s = 0; s < src.size; ++s) {
		for (int c = 0; c < src.channel; ++c) {
			for (int y = 0; y < src.height; ++y) {
				for (int x = 0; x < src.width; ++x) {
					dst.data[s][c][y + pad][x + pad] = src.data[s][c][y][x];
				}
			}
		}
	}

	return dst;
}

multmat im2col(const multmat &src, const multmat &filter, int stride)
{
	int output_height = 1 + (src.height - filter.height) / stride;
	int output_width = 1 + (src.width - filter.height) / stride;
	int col_height = src.size * src.channel * filter.height * filter.width;
	int col_width = output_height * output_width;

	multmat dst(1, 1, col_height, col_width);

	for (int s = 0; s < src.size; ++s) {
		for (int c = 0; c < src.channel; ++c) {
			for (int y = 0; y <= src.height - filter.height; y += stride) {
				for (int x = 0; x <= src.width - filter.width; x += stride) {
					for (int v = 0; v < filter.height; ++v) {
						for (int u = 0; u < filter.width; ++u) {
							int idx_h = s * (src.channel * filter.height * filter.width)
										+ c * (filter.height * filter.width) + v * filter.width + u;
							int idx_w = (y / stride) * output_width + (x / stride);
							dst.data[0][0][idx_h][idx_w] = src.data[s][c][y + v][x + u];
						}
					}
				}
			}
		}
	}

	return dst;
}

multmat im2col(const multmat &src, int filter_height, int filter_width, int stride, int pad)
{
	int output_height = (src.height + 2 * pad - filter_height) / stride + 1;
	int output_width = (src.width + 2 * pad - filter_width) / stride + 1;

	multmat img = padding(src, pad);

	int col_height = img.size * output_height * output_width;
	int col_width = img.channel * filter_height * filter_width;

	multmat dst(1, 1, col_height, col_width);

	for (int s = 0; s < img.size; ++s) {
		for (int c = 0; c < img.channel; ++c) {
			for (int y = 0; y <= img.height - filter_height; y += stride) {
				for (int x = 0; x <= img.width - filter_width; x += stride) {
					for (int v = 0; v < filter_height; ++v) {
						for (int u = 0; u < filter_width; ++u) {
							int idx_h = s * (output_height * output_width)
										+ (y / stride) * output_width + (x / stride);
							int idx_w = c * (filter_height * filter_width) + v * filter_width + u;
							dst.data[0][0][idx_h][idx_w] = img.data[s][c][y + v][x + u];
						}
					}
				}
			}
		}
	}

	return dst;
}

multmat col2im(const multmat &src, int size, int height, int width)
{
	/*int channel = src.height / size;
	multmat dst(size, channel, height, width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					int idx_h = s * channel + c;
					int idx_w = y * width + x;
					dst.data[s][c][y][x] = src.data[0][0][idx_h][idx_w];
				}
			}
		}
	}*/

	int channel = src.width;
	multmat dst(size, channel, height, width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					int idx_h = s * (height * width) + y * width + x;
					dst.data[s][c][y][x] = src.data[0][0][idx_h][c];
				}
			}
		}
	}

	return dst;
}

multmat col2im(const multmat &src, int size, int channel, int height, int width,
			   int filter_height, int filter_width, int stride, int pad)
{
	int output_height = (height + 2 * pad - filter_height) / stride + 1;
	int output_width = (width + 2 * pad - filter_width) / stride + 1;

	int pad_height = height + 2 * pad;
	int pad_width = width + 2 * pad;

	multmat img(size, channel, pad_height, pad_width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y <= pad_height - filter_height; y += stride) {
				for (int x = 0; x <= pad_width - filter_width; x += stride) {
					for (int v = 0; v < filter_height; ++v) {
						for (int u = 0; u < filter_width; ++u) {
							int idx_h = s * (output_height * output_width)
										+ (y / stride) * output_width + (x / stride);
							int idx_w = c * (filter_height * filter_width) + v * filter_width + u;
							img.data[s][c][y + v][x + u] = src.data[0][0][idx_h][idx_w];
						}
					}
				}
			}
		}
	}

	multmat dst(size, channel, height, width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					dst.data[s][c][y][x] = img.data[s][c][y + pad][x + pad];
				}
			}
		}
	}

	return dst;
}

/*multmat filt2col(const multmat &src, int size, int channel)
{
	int col_width = size * channel * src.height * src.width;

	multmat dst(1, 1, src.channel, col_width);

	for (int i = 0; i < src.channel; ++i) {
		for (int y = 0; y < src.height; ++y) {
			for (int x = 0; x < src.width; ++x) {
				for (int j = 0; j < channel; ++j) {
					for (int k = 0; k < size; ++k) {
						int idx_w = k * (channel * src.height * src.width)
									+ j * (src.height * src.width) + y * src.width + x;
						dst.data[0][0][i][idx_w] = src.data[0][i][y][x];
					}
				}
			}
		}
	}

	return dst;
}

multmat filt2col(const multmat &src, int channel)
{
	int col_height = channel * src.height * src.width;

	multmat dst(1, 1, col_height, src.channel);

	for (int c = 0; c < src.channel; ++c) {
		for (int y = 0; y < src.height; ++y) {
			for (int x = 0; x < src.width; ++x) {
				for (int i = 0; i < channel; ++i) {
					int idx_h = i * (src.height * src.width) + y * src.width + x;
					dst.data[0][0][idx_h][c] = src.data[0][c][y][x];
				}
			}
		}
	}

	return dst;
}*/

multmat filt2col(const multmat &src)
{
	int col_height = src.channel * src.height * src.width;

	multmat dst(1, 1, col_height, src.size);

	for (int s = 0; s < src.size; ++s) {
		for (int c = 0; c < src.channel; ++c) {
			for (int y = 0; y < src.height; ++y) {
				for (int x = 0; x < src.width; ++x) {
					int idx_h = c * (src.height * src.width) + y * src.width + x;
					dst.data[0][0][idx_h][s] = src.data[s][c][y][x];
				}
			}
		}
	}

	return dst;
}

multmat col2filt(const multmat &src, int channel, int height, int width)
{
	int size = src.width;

	multmat dst(size, channel, height, width);

	for (int s = 0; s < size; ++s) {
		for (int c = 0; c < channel; ++c) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					int idx_h = c * (height * width) + y * width + x;
					dst.data[s][c][y][x] = src.data[0][0][idx_h][s];
				}
			}
		}
	}

	return dst;
}

multmat bias2col(const multmat &src, int height)
{
	/*multmat dst(1, 1, src.channel, height);

	for (int c = 0; c < src.channel; ++c) {
		for (int x = 0; x < height; ++x) {
			dst.data[0][0][c][x] = src.data[0][c][0][0];
		}
	}*/

	multmat dst(1, 1, height, src.size);

	for (int y = 0; y < height; ++y) {
		for (int s = 0; s < src.size; ++s) {
			dst.data[0][0][y][s] = src.data[s][0][0][0];
		}
	}

	return dst;
}

multmat dout2col(const multmat &src)
{
	multmat dst(1, 1, src.size * src.height * src.width, src.channel);

	for (int s = 0; s < src.size; ++s) {
		for (int c = 0; c < src.channel; ++c) {
			for (int y = 0; y < src.height; ++y) {
				for (int x = 0; x < src.width; ++x) {
					int idx_h = s * (src.height * src.width) + y * src.width + x;
					dst.data[0][0][idx_h][c] = src.data[s][c][y][x];
				}
			}
		}
	}

	return dst;
}
