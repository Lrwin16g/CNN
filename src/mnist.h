#ifndef _MNIST_H_
#define _MNIST_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

namespace mnist
{
	// エンディアンを変換
    int reverseEndian(int i)
	{
		unsigned char c1 = i & 0xff;
		unsigned char c2 = (i >> 8) & 0xff;
		unsigned char c3 = (i >> 16) & 0xff;
		unsigned char c4 = (i >> 24) & 0xff;

		return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) + (static_cast<int>(c3) << 8) + static_cast<int>(c4);
	}

	// 画像データを読み出し
	template<typename Type>
    void loadImages(std::string filename, std::vector<std::vector<std::vector<Type> > > &images)
	{
		std::ifstream ifs(filename, std::ios::in | std::ios::binary);

		if (!ifs.is_open()) {
			throw std::runtime_error("failed to open " + filename);
		}

		int magic_number = 0;
		int number_of_images = 0;
		int rows = 0;
		int cols = 0;

		ifs.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
		magic_number = mnist::reverseEndian(magic_number);
		ifs.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
		number_of_images = mnist::reverseEndian(number_of_images);
		ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
		rows = mnist::reverseEndian(rows);
		ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
		cols = mnist::reverseEndian(cols);

		images.resize(number_of_images);

#ifdef _DEBUG
		std::cout << "Loading: " << filename << std::endl;
		std::cout << "magic number: " << magic_number << std::endl;
		std::cout << "number of images: " << number_of_images << std::endl;
		std::cout << "rows: " << rows << std::endl;
		std::cout << "cols: " << cols << std::endl << std::endl;
#endif

		for (int i = 0; i < number_of_images; ++i)
		{
			images[i].resize(rows);

			for (int y = 0; y < rows; ++y)
			{
				images[i][y].resize(cols);

				for (int x = 0; x < cols; ++x)
				{
					unsigned char temp = 0;
					ifs.read(reinterpret_cast<char*>(&temp), sizeof(temp));
					images[i][y][x] = static_cast<Type>(temp);
				}
			}
		}
	}

	// ラベルデータを読み出し
	template<typename Type>
    void loadLabels(std::string filename, std::vector<std::vector<Type> > &labels)
	{
		std::ifstream ifs(filename, std::ios::in | std::ios::binary);

		if (!ifs.is_open()) {
			throw std::runtime_error("failed to open " + filename);
		}

		int magic_number = 0;
		int number_of_labels = 0;

		ifs.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
		magic_number = mnist::reverseEndian(magic_number);
		ifs.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
		number_of_labels = mnist::reverseEndian(number_of_labels);

		labels.resize(number_of_labels);

#ifdef _DEBUG
		std::cout << "Loading: " << filename << std::endl;
		std::cout << "magic number: " << magic_number << std::endl;
		std::cout << "number of labels: " << number_of_labels << std::endl << std::endl;
#endif

		for (int i = 0; i < number_of_labels; ++i)
		{
			labels[i].resize(10);
			for (int j = 0; j < 10; ++j)
			{
				labels[i][j] = 0;
			}

			unsigned char temp = 0;
			ifs.read(reinterpret_cast<char*>(&temp), sizeof(temp));
			labels[i][static_cast<int>(temp)] = 1;
		}
	}

	// mnistデータをまとめて読み出し
	template<typename Type1, typename Type2>
    void load_mnist(std::vector<std::vector<std::vector<Type1> > > &train_images, std::vector<std::vector<Type2> > &train_labels,
                    std::vector<std::vector<std::vector<Type1> > > &test_images, std::vector<std::vector<Type2> > &test_labels,
                    std::string dir_path)
	{
		mnist::loadImages<Type1>(dir_path + "/train-images.idx3-ubyte", train_images);
		mnist::loadLabels<Type2>(dir_path + "/train-labels.idx1-ubyte", train_labels);
		mnist::loadImages<Type1>(dir_path + "/t10k-images.idx3-ubyte", test_images);
		mnist::loadLabels<Type2>(dir_path + "/t10k-labels.idx1-ubyte", test_labels);
	}
};

#endif