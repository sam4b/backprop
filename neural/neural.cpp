#include <iostream>
#include <Eigen/Dense> 
#include <vector>
#include <random>
#include <format>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <functional>
#include <future>
#include <thread>

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

enum class ActivationFunctionType {
	Sigmoid,
	Tanh,
	LeakyReLU,
	Identity
};


struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf bias;
	ActivationFunctionType type;

	nlohmann::json tojson() const {
		nlohmann::json out;

		out["weights"] = MatrixToJson(weights);
		out["bias"] = VectorToJson(bias);
		out["activation"] = magic_enum::enum_name(type);

		return out;
	}

	static Layer fromJson(const nlohmann::json& json) {
		Layer out;
		out.weights = JsonToMatrix(json["weights"]);
		out.bias = JsonToVector(json["bias"]);
		if (json["activation"] == "Sigmoid") {
			out.type = ActivationFunctionType::Sigmoid;
		}
		else if (json["activation"] == "Tanh") {
			out.type = ActivationFunctionType::Tanh;
		}
		else if (json["activation"] == "LeakyReLU") {
			out.type = ActivationFunctionType::LeakyReLU;
		}
		else if (json["activation"] == "Identity") {
			out.type = ActivationFunctionType::Identity;
		}
		else {
			assert(false);
		}
		return out;
	}
};

//Xavier weight initialisation cuz sigmoid
Layer createLayer(int neuronsIn, int neuronsOut, ActivationFunctionType type, const int randomState) {
	std::mt19937 gen(randomState);

	Eigen::MatrixXf weights;
	Eigen::VectorXf bias = Eigen::VectorXf::Zero(neuronsOut);

	if (type == ActivationFunctionType::Sigmoid || type == ActivationFunctionType::Tanh || type == ActivationFunctionType::Identity) {
		//Xavier

		float scale = std::sqrt(2.0f / (neuronsIn + neuronsOut));
		std::normal_distribution<float> dist(0.0f, scale);

		weights = Eigen::MatrixXf::NullaryExpr(
			neuronsOut, neuronsIn, [&](int, int) { return dist(gen); });
	}
	else if (type == ActivationFunctionType::LeakyReLU) { //He 
		float scale = 2.0f / (float)neuronsIn;
		std::normal_distribution<float> dist(0.0f, scale);

		weights = Eigen::MatrixXf::NullaryExpr(
			neuronsOut, neuronsIn, [&](int, int) { return dist(gen); }); 
	}
	else {
		assert(false);
	}
	return Layer{ weights, bias, type };
}

struct TrainingOptions {
	int batchSize;
	float learningRate;
	int epochs;
};

const std::unordered_map<ActivationFunctionType, std::function<float(float)>> activationFunctions = {
	{ActivationFunctionType::Sigmoid, [](const float x) -> float { return  1.0f / (1.0f + exp(-1.0f * x));}},
	{ActivationFunctionType::Tanh, [](const float x) -> float { return sinh(x) / cosh(x);}}
	,{ActivationFunctionType::LeakyReLU, [](const float x) -> float { return (x >= 0) ? x : 0.01f * x;}},
	{ActivationFunctionType::Identity, [](const float x) { return x;}}
};

const std::unordered_map<ActivationFunctionType, std::function<float(float)>> derivativeMap = {
	{ActivationFunctionType::Sigmoid, [](const float x) -> float {
	const float sigmoid = 1.0f / (1.0f + exp(-1.0f * x));
	return sigmoid * (1.0f - sigmoid);
	}},
	{ActivationFunctionType::Tanh, [](const float x) -> float {
		return (1.0f / cosh(x)) * (1.0f / cosh(x));
	}},

	{ActivationFunctionType::LeakyReLU, [](const float x) -> float {
		return (x > 0) ? 1.0f : 0.01f;
}},
	{ActivationFunctionType::Identity,[](const float x) { return 1.0f;}} 


};

//copy xs as we need to reuse a matrix of its size anyway!
void matrix_based_backprop(Eigen::MatrixXf xs, const Eigen::MatrixXf& ys, const std::vector<Layer>& layers, std::vector<Eigen::VectorXf>& biasErrors, std::vector<Eigen::MatrixXf>& weightErrors,
	const std::vector<Eigen::MatrixXf>& biasMatricesPerLayer) {
	std::vector<Eigen::MatrixXf> zs; //weighted sums
	std::vector<Eigen::MatrixXf> as; //activations
	//Reserve ! (re-use matrices ?)

	as.push_back(xs);

	//Feed-forward the batch 
	int biasIdx = 0;
	assert(biasMatricesPerLayer.size() == layers.size());
	for (const auto& layer : layers) {
		const Eigen::MatrixXf& biases = biasMatricesPerLayer[biasIdx++];

		Eigen::MatrixXf z = layer.weights * xs + biases;
		Eigen::MatrixXf a = z.unaryExpr(activationFunctions.at(layer.type));

		zs.push_back(z);
		as.push_back(a);

		xs = a;
	}

	//Compute error at Lth layer
	const Eigen::MatrixXf cost_derivative = (xs - ys);
	const Eigen::MatrixXf zs_derivative = zs.back().unaryExpr(derivativeMap.at(layers.back().type));

	Eigen::MatrixXf delta = cost_derivative.cwiseProduct(zs_derivative);

	biasErrors.back() = delta.rowwise().sum(); //Reduce from matrix of errors down to a vector of errors for the bias for this layer
	weightErrors.back() = delta * as[as.size() - 2].transpose();

	//Normal backprop recurrence for lth layer

	for (int layer = layers.size() - 2; layer >= 0; layer--) {
		const Eigen::MatrixXf zs_derivative = zs[layer].unaryExpr(derivativeMap.at(layers[layer].type));
		const Eigen::MatrixXf weight_error_product = layers[layer + 1].weights.transpose() * delta;

		delta = zs_derivative.cwiseProduct(weight_error_product); 

		biasErrors[layer] = delta.rowwise().sum(); //Reduce from matrix of errors down to a vector of errors for the bia sfor this layer


		//Collapse weightrows x (weightcols * batchsize) mat -> weightrows x weightcols mat by product of transpose.

		weightErrors[layer] = delta * as[layer].transpose();


	}

}



using LabelledSet = std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>;




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


class Network {
public:
	//TBA: Add weight initialization type (enum)?
	Network(const std::vector<std::pair<int, ActivationFunctionType>>& neurons, const int randomState) {
		m_randomState = randomState;
		for (int i = 0; i < neurons.size() - 1; i++) {
			const int out = neurons[i + 1].first;
			const int in = neurons[i].first;

			layers.emplace_back(createLayer(in, out, neurons[i].second, this->rand()));
		}

	}

	Network(const std::vector<std::pair<int, ActivationFunctionType>>& neurons) {

		m_randomState = std::random_device()();
		for (int i = 0; i < neurons.size() - 1; i++) {
			const int out = neurons[i + 1].first;
			const int in = neurons[i].first;

			layers.emplace_back(createLayer(in, out, neurons[i].second, this->rand()));
		}

	}

	Network(const std::filesystem::path& path) {
		std::ifstream in(path);

		const nlohmann::json json = nlohmann::json::parse(in);

		m_randomState = std::random_device()(); //tba hacky
		fromjson(json);
	}

	void train(LabelledSet& trainingSet, const TrainingOptions options) {
		const auto batchUpdate = [&]() -> void {
			static int epochs = 1;
			std::cout << std::format("Epoch {}/{} completed.\n", epochs++, options.batchSize);
			};
		MatrixSGD(trainingSet, options, batchUpdate);
	}

	void train(LabelledSet& trainingSet, const TrainingOptions options, const std::function<void()>& progressUpdater) {
		MatrixSGD(trainingSet, options, progressUpdater);
	}
	
	void WriteNetworkAsJson(const std::filesystem::path& dest) const {
		std::ofstream out(dest);
		assert(out);

		const nlohmann::json json = tojson();

		out << json;
	}

	Eigen::VectorXf predict(Eigen::VectorXf x) const {
		for (const auto& layer : layers) {
			x = layer.weights * x + layer.bias;
			x = x.unaryExpr(activationFunctions.at(layer.type));
		}

		return x;
	}
private:
	void MatrixSGD(LabelledSet& trainingSet, const TrainingOptions options, const std::function<void()>& progressUpdater) {
		std::mt19937 mersenne(this->rand());
		std::uniform_int_distribution<> sampler(0, trainingSet.size() - 1);

		std::vector<Eigen::MatrixXf> biasMatrices; //create a bias matrix for n examples (just the bias vector augmented with itself n times) per layer.
		biasMatrices.reserve(layers.size());

		for (const auto& layer : layers) {
			biasMatrices.emplace_back(layer.bias.replicate(1, options.batchSize));
		}

		std::vector<Eigen::VectorXf> updateBias;
		std::vector<Eigen::MatrixXf> updateWeights;

		updateBias.reserve(layers.size());
		updateWeights.reserve(layers.size());

		for (const auto& layer : layers) {
			updateBias.push_back(Eigen::VectorXf::Zero(layer.bias.size()));
			updateWeights.push_back(Eigen::MatrixXf::Zero(layer.weights.rows(), layer.weights.cols() * options.batchSize)); //I think this is right?
		}

		for (int epoch = 0; epoch < options.epochs; epoch++) {
			std::shuffle(trainingSet.begin(), trainingSet.end(), mersenne);
			int idx = 0;
			for (int batch = 0; batch < trainingSet.size() / options.batchSize; batch++) {

				for (int i = 0; i < layers.size(); i++) {
					updateBias[i].setZero();
					updateWeights[i].setZero();
				}

				const int x_rows = trainingSet[0].first.size(); //Number of rows of the input vector
				const int y_rows = trainingSet[0].second.size();

				Eigen::MatrixXf xs = Eigen::MatrixXf::Zero(x_rows, options.batchSize); //For MNIST, should be 784 x N matrix.
				Eigen::MatrixXf ys = Eigen::MatrixXf::Zero(y_rows, options.batchSize); //For MNIST, should be 10 X N matrix.

				//Stick them all together


				for (int i = 0; i < options.batchSize; i++) {
					assert(idx < trainingSet.size()); 
					const auto& [x, y] = trainingSet[idx++];

					for (int j = 0; j < x_rows; j++) {
						xs(j, i) = x(j);
					}

					for (int j = 0; j < y_rows; j++) {
						ys(j, i) = y(j);
					}
				}

				matrix_based_backprop(xs, ys, layers, updateBias, updateWeights, biasMatrices);

				assert(layers.size() == updateBias.size());
				assert(updateWeights.size() == updateBias.size());

				for (int i = 0; i < layers.size(); i++) {
					layers[i].bias -= (options.learningRate / (float)options.batchSize) * updateBias[i];
					layers[i].weights -= (options.learningRate / (float)options.batchSize) * updateWeights[i];
				}

			}
			progressUpdater();
		}
	
	}


	void fromjson(const nlohmann::json& json) {
		for (const auto& layer : json["layers"]) {
			layers.push_back(Layer::fromJson(layer));
		}
	}
	nlohmann::json tojson() const {
		nlohmann::json out;
		
		for (const auto& layer : layers) {
			out["layers"].push_back(layer.tojson());
		}

		return out;
	}

	int rand() {
		m_randomState = (1103515245 * m_randomState + 12345) & 0x7fffffff;
		return m_randomState;
	}

	int m_randomState;

	std::vector<Layer> layers;
};

void experiment() {

	std::vector<float> sample;
	std::string csv;

	const float end = 3.14159f * 2.0f;
	const float begin = 0.0f;

	for (float x = begin; x <= end; x += 0.01f) {
		sample.push_back(x);
		csv += "x=" + std::to_string(x) + ",";
	}

	csv.pop_back();

	csv += "\n";


	Network network(std::vector<std::pair<int, ActivationFunctionType>>{{1, ActivationFunctionType::Tanh}, { 60, ActivationFunctionType::Tanh }, { 35, ActivationFunctionType::Tanh }, { 10, ActivationFunctionType::Tanh }, { 1, ActivationFunctionType::Tanh }}, 55);

	LabelledSet observations;

	const auto f = [](const float x) -> float { return sin(x); };

	const float delta = 0.0002;
	const int amount = (int)(2.0f / delta);
	observations.reserve(amount);

	const auto norm = [](const float x) -> float { //Map from [0, 2pi] to [-1, 1]
		return   2.0f * (x - 0.0f) / (2.0f * 3.14159f - 0.0f) - 1.0f;
		};


	for (float x = begin; x <= end; x += delta) {


		Eigen::VectorXf x_vec(1);
		x_vec(0) = norm(x);

		Eigen::VectorXf y_vec(1);
		y_vec(0) = f(x);

		observations.push_back({ x_vec, y_vec });

	}


	const auto evaluator = [&]() -> void {
		float error = 0.0f;
		for (const float x : sample) {
			Eigen::VectorXf vec(1);
			vec(0) = norm(x);
			const Eigen::VectorXf pred = network.predict(vec);
			const float y = pred(0);

			csv += std::to_string(y) + ",";

			error += ((sin(x)) - y) * ((sin(x)) - y);

		}
		error /= (float)sample.size();
		std::cout << "MSE: " << error << std::endl;
		csv.pop_back();
		csv += "\n";
		};

	network.train(observations, TrainingOptions
		{
			.batchSize = 10,
			.learningRate = 0.01f,
			.epochs = 50
		}, evaluator);

	network.WriteNetworkAsJson("C:/Users/Sam/Desktop/predictor.json");


	std::ofstream out("C:/Users/Sam/Desktop/output.csv");
	out << csv;
}

//tba: split test into validation and test
TrainingOptions gridsearch(const std::vector<TrainingOptions>& hyperparameters, const LabelledSet& trainingSet, const LabelledSet& validationSet) {
	std::vector<std::thread> threads;


	for (const auto& hyperparams : hyperparameters) {
		std::async([&]() -> std::pair<TrainingOptions, int> {
			const auto params = hyperparams;
			const auto train = trainingSet;
			Network network; //TBA add a struct for network layout
			});
	}
	
}

int main() {
	experiment();
	return 0;
	auto train = readLabelledData("C:\\Users\\Sam\\Downloads\\train-images.idx3-ubyte", "C:\\Users\\Sam\\Downloads\\train-labels.idx1-ubyte");
	const auto test = readLabelledData("C:\\Users\\Sam\\Downloads\\t10k-images.idx3-ubyte", "C:\\Users\\Sam\\Downloads\\t10k-labels.idx1-ubyte");
	const std::filesystem::path jsonPath = "C:/Users/Sam/Desktop/mynetwork.json";

	std::cout << "Train or test? ";
	std::string in;
	std::cin >> in;
	std::transform(in.begin(), in.end(), in.begin(),
		[](unsigned char c) { return std::tolower(c); });






	if (in == "train") {
		for (int i = 0; i < 10; i++) {
			print(train[i]);
		}

		const TrainingOptions options = {
		.batchSize = 10,
		.learningRate = 3.0f,
		.epochs = 30
		};

		const int sampleSize = 50;

		Network network(std::vector<std::pair<int, ActivationFunctionType>>{ {784, ActivationFunctionType::Sigmoid}, { 30, ActivationFunctionType::Sigmoid }, { 10, ActivationFunctionType::Sigmoid } }); //784 neurons in, 10 neurons out (prediction: [0-9].)
		
		std::random_device rd;
		std::mt19937 mersenne(rd());
		std::uniform_int_distribution<> sampler(0, train.size() - 1);

		const auto progressUpdater = [&]() -> void {
			int correct = 0;
			static int epoch = 1;

			for (int i = 0; i < sampleSize; i++) {
				const int idx = sampler(mersenne);

				const auto yhat = network.predict(train[idx].first);

				const auto& y = train[idx].second;

				Eigen::Index maxYHat, maxY;
				yhat.maxCoeff(&maxYHat);
				y.maxCoeff(&maxY);

				if (maxYHat == maxY) {
					correct++;
				}


			}

			std::cout << std::format("Epoch {}/{}. {}/{} correct ({}%).\n",
				epoch++, options.epochs,
				correct, sampleSize,
				static_cast<float>(correct) * 100.0f / static_cast<float>(sampleSize));
		};
		
		network.train(train, options, progressUpdater);

		network.WriteNetworkAsJson(jsonPath);
	}
	else if (in == "test") {
		Network network(jsonPath);
		while (true) {
			std::cout << std::format("\nEnter number in range [{}, {}]: ", 0, test.size() - 1);
			int idx;
			std::cin >> idx;
			assert(idx >= 0 && idx < test.size() );
			print(test[idx]);
			const auto pred = network.predict(test[idx].first);
			Eigen::Index maxidx;
			pred.maxCoeff(&maxidx);
			std::cout << std::format("Predicted {}.", (int)maxidx);
		}

	}
	else {
		return -1;
	}
	return 0;
} 


