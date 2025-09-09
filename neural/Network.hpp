#pragma once
#include "Layer.hpp"
#include <vector>
#include <unordered_map>
#include <functional>
#include <utility>
#include <fstream>
#include <iostream>
#include <format>
#include "Backpropagation.hpp"

using LabelledSet = std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>;

/*Slight regression for now, moving to a singular activation function for the whole of the network*/

struct Hyperparameters {
	std::vector<int> hiddenLayers;
	float learningRate;
	ActivationFunctionType activationFunction;
	//TBA: cost function
	//TBA: Optimizer (ADAM implementation)
	int epochs;
	int batchSize;
};

struct HyperparametersSearchSpace {
	std::vector<std::vector<int>> hiddenLayers;
	std::vector<float> learningRates;
	std::vector<ActivationFunctionType> activationFunctions;
	std::vector<int> epochs;
	std::vector<int> batchSize;
};



class Network {
public:
	Network() = default;

	inline Network(const std::filesystem::path& path) {
		std::ifstream in(path);

		const nlohmann::json json = nlohmann::json::parse(in);
		fromjson(json);
	}

	inline void train(const LabelledSet& trainingSet, const Hyperparameters& parameters) {
		std::random_device rng;

		train_setup(trainingSet, parameters, rng(), [&]() {
			std::cout << std::format("Epoch {}/{} completed\n", m_epochCount, parameters.epochs);
			});
	}

	inline void train(const LabelledSet& trainingSet, const Hyperparameters& parameters, const int randomState, const std::function<void()>& batchEvaluator) {
		train_setup(trainingSet, parameters, randomState, batchEvaluator);
	}

	inline void train(const LabelledSet& trainingSet, const Hyperparameters& parameters, const std::function<void()>& batchEvaluator) {
		std::random_device rng;
		
		train_setup(trainingSet, parameters, rng(), batchEvaluator);
	}

	inline void train(const LabelledSet& trainingSet, const Hyperparameters& parameters, const int randomState) {
		train_setup(trainingSet, parameters, randomState, [&]() {
			std::cout << std::format("Epoch {}/{} completed\n", m_epochCount, parameters.epochs);
			});
	}

	inline void WriteNetworkAsJson(const std::filesystem::path& dest) const {
		std::ofstream out(dest);
		assert(out);

		const nlohmann::json json = tojson();

		out << json;
	}

	inline Eigen::VectorXf predict(Eigen::VectorXf x) const {
		for (const auto& layer : m_layers) {
			x = layer.weights * x + layer.bias;
			x = x.unaryExpr(activationFunctions.at(layer.type));
		}

		return x;
	}
private:
	inline void train_setup(const LabelledSet& trainingSet, const Hyperparameters& parameters, const int randomState, const std::function<void()>& batchEvaluator) {
		//Assume that for all i, j in trainingSet, x_i.size() == x_j.size() && y_i.size() == y_j.size(). (Check later?)

		int neuronsIn = trainingSet[0].first.size();

		for (int i = 0; i < parameters.hiddenLayers.size(); i++) {
			const int output = parameters.hiddenLayers[i];
			m_layers.push_back(createLayer(neuronsIn, output, parameters.activationFunction, randomState));
			neuronsIn = output;
		}

		m_layers.push_back(createLayer(neuronsIn, trainingSet[0].second.size(), parameters.activationFunction, randomState));
		MatrixSGD(trainingSet, parameters, batchEvaluator, randomState);
	}

	inline void MatrixSGD(const LabelledSet& trainingSet, const Hyperparameters parameters, const std::function<void()>& progressUpdater, int randomState) {

		//Generate indices to shuffle here, so we don't have to copy the training set.
		std::vector<size_t> indices;
		indices.reserve(trainingSet.size());
		for (int i = 0; i < trainingSet.size(); i++) {
			indices.push_back(i);
		}

		//indicies only contains elements in the range [0, trainingSet.size() - 1], and trainingSet remains constant - so indexing is safe.

		std::mt19937 mersenne(randomState);
		std::uniform_int_distribution<> sampler(0, trainingSet.size() - 1);

		std::vector<Eigen::MatrixXf> biasMatrices; //create a bias matrix for n examples (just the bias vector augmented with itself n times) per layer.
		biasMatrices.reserve(m_layers.size());

		for (const auto& layer : m_layers) {
			biasMatrices.emplace_back(layer.bias.replicate(1, parameters.batchSize));
		}

		std::vector<Eigen::VectorXf> updateBias;
		std::vector<Eigen::MatrixXf> updateWeights;

		updateBias.reserve(m_layers.size());
		updateWeights.reserve(m_layers.size());

		for (const auto& layer : m_layers) {
			updateBias.push_back(Eigen::VectorXf::Zero(layer.bias.size()));
			updateWeights.push_back(Eigen::MatrixXf::Zero(layer.weights.rows(), layer.weights.cols() * parameters.batchSize)); //I think this is right?
		}

		for (int epoch = 0; epoch < parameters.epochs; epoch++) {
			m_epochCount = epoch; //auxilary variable used for logging

			std::shuffle(indices.begin(), indices.end(), mersenne);
			int indiciesIdx = 0;
			for (int batch = 0; batch < trainingSet.size() / parameters.batchSize; batch++) {

				for (int i = 0; i < m_layers.size(); i++) {
					updateBias[i].setZero();
					updateWeights[i].setZero();
				}

				const int x_rows = trainingSet[0].first.size(); //Number of rows of the input vector
				const int y_rows = trainingSet[0].second.size();

				Eigen::MatrixXf xs = Eigen::MatrixXf::Zero(x_rows, parameters.batchSize); //For MNIST, should be 784 x N matrix.
				Eigen::MatrixXf ys = Eigen::MatrixXf::Zero(y_rows, parameters.batchSize); //For MNIST, should be 10 X N matrix.

				//Stick them all together


				for (int i = 0; i < parameters.batchSize; i++) {
					const auto trainingSetIdx = indices[indiciesIdx];
					assert(trainingSetIdx < trainingSet.size());
					const auto& [x, y] = trainingSet[trainingSetIdx];

					indiciesIdx++;

					for (int j = 0; j < x_rows; j++) {
						xs(j, i) = x(j);
					}

					for (int j = 0; j < y_rows; j++) {
						ys(j, i) = y(j);
					}
				}

				matrix_based_backprop(xs, ys, m_layers, updateBias, updateWeights, biasMatrices);

				assert(m_layers.size() == updateBias.size());
				assert(updateWeights.size() == updateBias.size());

				for (int i = 0; i < m_layers.size(); i++) {
					m_layers[i].bias -= (parameters.learningRate / (float)parameters.batchSize) * updateBias[i];
					m_layers[i].weights -= (parameters.learningRate / (float)parameters.batchSize) * updateWeights[i];
				}

			}
			progressUpdater();
		}

	}


	inline void fromjson(const nlohmann::json& json) {
		for (const auto& layer : json["layers"]) {
			m_layers.push_back(Layer::fromJson(layer));
		}
	}

	inline nlohmann::json tojson() const {
		nlohmann::json out;

		for (const auto& layer : m_layers) {
			out["layers"].push_back(layer.tojson());
		}

		return out;
	}

	int m_epochCount;
	std::vector<Layer> m_layers;
};

//Grid search for best combination of hyperparameters to maximise score on validation set
//Evaluator should be a function that returns true if the prediction (the first parameter, yhat) is correct (determined by the label, y, the second parameter).

inline Hyperparameters gridSearch(const HyperparametersSearchSpace& searchSpace, const LabelledSet& trainingSet, const LabelledSet& validationSet, int randomState,
	const std::function<bool(const Eigen::VectorXf&, const Eigen::VectorXf&)> evaluator) {
	//Precondition: all vectors have at least one element

	Hyperparameters best = {
		.hiddenLayers = searchSpace.hiddenLayers[0],
		.learningRate = searchSpace.learningRates[0],
		.activationFunction = searchSpace.activationFunctions[0],
		.epochs = searchSpace.epochs[0],
		.batchSize = searchSpace.batchSize[0]
	};
	int bestScore = 0;

	//TODO: Parallelise

	for (const auto& hiddenLayerLayout : searchSpace.hiddenLayers) {
		for (const auto learningRate : searchSpace.learningRates) {
			for (const auto activationFunctions : searchSpace.activationFunctions) {
				for (const auto epochs : searchSpace.epochs) {
					for (const auto batchSize : searchSpace.batchSize) {
						const Hyperparameters tryParams = {
							.hiddenLayers = hiddenLayerLayout,
							.learningRate = learningRate,
							.activationFunction = activationFunctions,
							.epochs = epochs,
							.batchSize = batchSize
						};

						Network network;
						network.train(trainingSet, tryParams);

						int correct = 0;
						for (const auto& [x, y] : validationSet) {
							const auto yhat = network.predict(x);
							if (evaluator(yhat, y)) {
								correct++;
							}
						}
						//ValidationSet size remains constant, so just compare 
						if (correct >= bestScore) {
							best = tryParams;
						}
						bestScore = correct;
					}
				}
			}
		}
	}
	return best;
}

inline Hyperparameters gridSearch(const HyperparametersSearchSpace& searchSpace, const LabelledSet& trainingSet, const LabelledSet& validationSet,
	const std::function<bool(const Eigen::VectorXf&, const Eigen::VectorXf&)> evaluator) {
	std::random_device rng;
	return gridSearch(searchSpace, trainingSet, validationSet, rng(), evaluator);
}
