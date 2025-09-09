#pragma once
#include <fstream>
#include <vector>
#include <filesystem>
#include <cassert>
#include <iostream>
#include <Eigen/Dense>


inline uint32_t readBigEndianUint32(std::ifstream& f) {
	uint32_t result = 0;
	for (int i = 0; i < 4; ++i) {
		unsigned char byte;
		f.read(reinterpret_cast<char*>(&byte), 1);
		result = (result << 8) | byte;
	}
	return result;
}

//MNIST reader
inline std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> readLabelledData(const std::filesystem::path& imagesPath, const std::filesystem::path& labelsPath) {
	std::ifstream labels(labelsPath, std::ios::binary);
	assert(labels);

	std::ifstream images(imagesPath, std::ios::binary);
	assert(images);

	const uint32_t magic_labels = readBigEndianUint32(labels);
	const uint32_t magic_images = readBigEndianUint32(images);

	assert(magic_labels == 2049); //Indicates file is labels
	assert(magic_images == 2051); //Indicates file is imgaes;

	const uint32_t count_labels = readBigEndianUint32(labels);
	const uint32_t count_images = readBigEndianUint32(images);

	assert(count_images == count_labels);

	std::vector<std::uint8_t> label_ints(count_labels);
	labels.read(reinterpret_cast<char*>(label_ints.data()), count_labels);

	//Map to vector type

	std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> out(count_labels);

	std::vector<Eigen::VectorXf> labelOut;
	for (int i = 0; i < label_ints.size(); i++) {
		const int label = label_ints[i];
		Eigen::VectorXf temp(10);
		for (int i = 0; i < 10; i++) {
			temp(i) = 0;
		}
		temp(label) = 1;

		out[i].second = temp;
	}

	labels.close();


	const uint32_t rows = readBigEndianUint32(images);
	const uint32_t columns = readBigEndianUint32(images);

	const size_t imageSize = rows * columns;
	assert(rows * columns == 784);

	std::vector<Eigen::VectorXf> imagesOut;

	std::vector<std::uint8_t> image(imageSize);

	for (int i = 0; i < count_images; i++) {

		images.read(reinterpret_cast<char*>(image.data()), imageSize);
		Eigen::VectorXf vec(784);
		for (int j = 0; j < 784; j++) {
			vec(j) = (float)image[j] / (float)255.0f;
		}
		out[i].first = vec;
	}

	return out;
}

inline int vectorToLabel(const Eigen::VectorXf& vec) {
	Eigen::Index idx;
	vec.maxCoeff(&idx);
	return idx;
}

inline void print(const std::pair<Eigen::VectorXf, Eigen::VectorXf>& pair) {
	const int label = vectorToLabel(pair.second);

	std::cout << "Label: " << label << std::endl;
	for (uint32_t r = 0; r < 28; ++r) {
		for (uint32_t c = 0; c < 28; ++c) {
			std::cout << (pair.first(r * 28 + c) > 0.5f ? '#' : '.');
		}
		std::cout << "\n";
	}


}
