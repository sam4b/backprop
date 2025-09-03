#include <iostream>
#include <Eigen/Dense> 
#include <vector>
#include <random>
#include <format>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <filesystem>
#include <nlohmann/json.hpp>

struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf bias;
};


//Xavier weight initialisation cuz sigmoid
Layer createLayer(int neuronsIn, int neuronsOut) {
	std::random_device rd;
	std::mt19937 gen(rd());
	float scale = std::sqrt(1.0f / neuronsIn);
	std::normal_distribution<float> dist(0.0f, scale);

	Eigen::MatrixXf weights = Eigen::MatrixXf::NullaryExpr(
		neuronsOut, neuronsIn, [&](int, int) { return dist(gen); });
	Eigen::VectorXf bias = Eigen::VectorXf::Zero(neuronsOut);
	return Layer{ weights, bias };
}


const auto sigmoid = [](const float x) -> float {
	return 1.0f / (1.0f + exp(-1.0f * x));
	};

const auto sigmoidDerivative = [](const float x) -> float {
	return sigmoid(x) * (1.0f - sigmoid(x));
	};






Eigen::VectorXf FeedForward(Eigen::VectorXf x, const std::vector<Layer>& network) {

	for (const auto& layer : network) {
		x = layer.weights * x + layer.bias;
		x = x.unaryExpr(sigmoid);
	}

	return x;
}

struct TrainingOptions {
	int batchSize;
	float learningRate;
	int iterations;
	int progressSampleSize;
};


void backprop(const std::pair<Eigen::VectorXf, Eigen::VectorXf>& data, const std::vector<Layer>& layers, std::vector<Eigen::VectorXf>& biasErrors, std::vector<Eigen::MatrixXf>& weightErrors) { 
	std::vector<Eigen::VectorXf> weightedSums; 
	std::vector<Eigen::VectorXf> activations; 
	const Eigen::VectorXf xcopy = data.first; 
	Eigen::VectorXf x = data.first; 
	activations.push_back(x); 

	for (const auto& layer : layers) {
		weightedSums.push_back(layer.weights * x + layer.bias);
		activations.push_back(weightedSums.back());
		activations.back() = activations.back().unaryExpr(sigmoid);
		x = activations.back(); 
	} 
	const auto& label = data.second; //y
	const auto& predicted_label = x; //y hat 

	Eigen::VectorXf delta = (predicted_label - label).cwiseProduct(weightedSums.back().unaryExpr(sigmoidDerivative));

    assert(weightErrors.size() == biasErrors.size());
	assert(weightedSums.size() == biasErrors.size()); 
	assert(biasErrors.size() == layers.size());

	biasErrors[biasErrors.size() - 1] = delta;
	weightErrors[weightErrors.size() - 1] = delta * activations[activations.size() - 2].transpose(); 

	for (int i = biasErrors.size() - 2; i >= 0; i--) {
		const Eigen::VectorXf activationDerivative = weightedSums[i].unaryExpr(sigmoidDerivative);
		assert(i + 1 < layers.size()); //loop invariant
		delta = layers[i + 1].weights.transpose() * delta;
		delta = delta.cwiseProduct(activationDerivative);
		biasErrors[i] = delta;
		weightErrors[i] = delta * activations[i].transpose();
	}
} 

using LabelledSet = std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>;



void SGD(std::vector<Layer>& layers, const LabelledSet& trainingSet, const TrainingOptions options) {

	std::random_device device;
	std::mt19937 mersenne(device());
	std::uniform_int_distribution<> sampler(0, trainingSet.size() - 1);

	for (int batch = 0; batch < options.iterations; batch++) {
		std::vector<Eigen::VectorXf> updateBias;
		std::vector<Eigen::MatrixXf> updateWeights;

		for (const auto& layer : layers) {
			updateBias.emplace_back(Eigen::VectorXf::Zero(layer.bias.size()));
			updateWeights.emplace_back(Eigen::MatrixXf::Zero(layer.weights.rows(), layer.weights.cols()));

		}

		for (int sample = 0; sample < options.batchSize; sample++) {
			const auto idx = sampler(mersenne);

			std::vector<Eigen::VectorXf> deltaBias(layers.size());
			std::vector<Eigen::MatrixXf> deltaWeight(layers.size());

			backprop(trainingSet[idx], layers, deltaBias, deltaWeight);

			for (int i = 0; i < updateBias.size(); i++) {
				updateBias[i] += deltaBias[i];
				updateWeights[i] += deltaWeight[i];
			}
		}

		for (int i = 0; i < layers.size(); i++) {
			layers[i].bias -= (options.learningRate / (float)options.batchSize) * updateBias[i];
			layers[i].weights -= (options.learningRate / (float)options.batchSize) * updateWeights[i];
		}

		int correct = 0;

		for (int i = 0; i < options.progressSampleSize; i++) {
			const int idx = sampler(mersenne);

			const auto yhat = FeedForward(trainingSet[idx].first, layers);

			const auto& y = trainingSet[idx].second;

			Eigen::Index maxYHat, maxY;
			yhat.maxCoeff(&maxYHat);
			y.maxCoeff(&maxY);

			if (maxYHat == maxY) {
				correct++;
			}


		}

		std::cout << std::format("Batch {}/{}. {}/{} correct ({}%).\n", batch + 1 /*offset 0 start*/, options.iterations, correct, options.progressSampleSize, static_cast<float>(correct) * 100.0f / static_cast<float>(options.progressSampleSize));

	}

	}


uint32_t readBigEndianUint32(std::ifstream& f) {
	uint32_t result = 0;
	for (int i = 0; i < 4; ++i) {
		unsigned char byte;
		f.read(reinterpret_cast<char*>(&byte), 1);
		result = (result << 8) | byte;
	}
	return result;
}

//MNIST reader
std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> readLabelledData(const std::filesystem::path& imagesPath, const std::filesystem::path& labelsPath) {
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

int vectorToLabel(const Eigen::VectorXf& vec) {
	Eigen::Index idx;
	vec.maxCoeff(&idx);
	return idx;
}

void print(const std::pair<Eigen::VectorXf, Eigen::VectorXf>& pair) {
	const int label = vectorToLabel(pair.second);

	std::cout << "Label: " << label << std::endl;
	for (uint32_t r = 0; r < 28; ++r) {
		for (uint32_t c = 0; c < 28; ++c) {
			std::cout << (pair.first(r * 28 + c) > 0.5f ? '#' : '.');
		}
		std::cout << "\n";
	}


}

nlohmann::json VectorToJson(const Eigen::VectorXf& vector) {
	nlohmann::json out;

	out["elements"] = vector.size();

	for (int i = 0; i < vector.size(); i++) {
		out["data"].push_back(vector(i));
	}

	return out;
}

nlohmann::json MatrixToJson(const Eigen::MatrixXf& matrix) {
	nlohmann::json out;
	out["rows"] = matrix.rows();
	out["cols"] = matrix.cols();

	for (int i = 0; i < matrix.rows(); i++) {
		for (int j = 0; j < matrix.cols(); j++) {
			out["data"].push_back(matrix(i, j));
		}
	}

	return out;
}

Eigen::MatrixXf JsonToMatrix(const nlohmann::json& json) {
	const int rows = json["rows"];
	const int cols = json["cols"];

	assert(rows * cols == json["data"].size());
	assert(json["data"].is_array());
	assert(json["data"][0].is_number_float()); //assumes non-empty

	Eigen::MatrixXf matrix(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix(i, j) = json["data"][i * cols + j];
		}
	}
	return matrix;
}

Eigen::VectorXf JsonToVector(const nlohmann::json& json) {
	const int elements = json["elements"];

	assert(json["data"].is_array());
	assert(json["data"][0].is_number_float()); //again non-empty assertion, deal with it later.

	Eigen::VectorXf vector(elements);

	for (int i = 0; i < elements; i++) {
		vector(i) = json["data"][i];
	}

	return vector;
}
int main() {
	Eigen::MatrixXf matrix{ {32,12,0},
		{4,5,6} ,
		{1,2,3} };
	
	Eigen::MatrixXf identity = JsonToMatrix(MatrixToJson(matrix));
	std::cout << matrix << std::endl << identity << std::endl;
	assert(identity == matrix); //inverse of a function composed with function = identity function

	Eigen::VectorXf vec{ { 1,2,3,4564,5,6,7} };
	Eigen::VectorXf loaded = JsonToVector(VectorToJson(vec));
	std::cout << vec << std::endl << loaded << std::endl;
	assert(vec == loaded);

	return 0;
	const auto train = readLabelledData("C:\\Users\\Sam\\Downloads\\train-images.idx3-ubyte", "C:\\Users\\Sam\\Downloads\\train-labels.idx1-ubyte");
	const auto test = readLabelledData("C:\\Users\\Sam\\Downloads\\t10k-images.idx3-ubyte", "C:\\Users\\Sam\\Downloads\\t10k-labels.idx1-ubyte");

	for (int i = 0; i < 10; i++) {
		print(train[i]);
	}

	std::vector<Layer> network; //784 neurons in, 10 neurons out (prediction: [0-9].)
	network.push_back(createLayer(784, 60));
	network.push_back(createLayer(60, 10));


	const TrainingOptions options = {
		.batchSize = 100,
		.learningRate = 3.0f,
		.iterations = 200,
		.progressSampleSize = 50
	};

	SGD(network, train, options);

	return 0;
}


