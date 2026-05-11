#pragma once
#include <vector>
#include <Eigen/Dense>
#include <cassert>
#include <unordered_map>
#include "Layer.hpp"

inline const std::unordered_map<ActivationFunctionType, std::function<float(float)>> activationFunctions = {
	{ActivationFunctionType::Sigmoid, [](const float x) -> float { return  1.0f / (1.0f + exp(-1.0f * x));}},
	{ActivationFunctionType::Tanh, [](const float x) -> float { return sinh(x) / cosh(x);}}
	,{ActivationFunctionType::LeakyReLU, [](const float x) -> float { return (x >= 0) ? x : 0.01f * x;}},
	{ActivationFunctionType::Identity, [](const float x) { return x;}}
};

inline const std::unordered_map<ActivationFunctionType, std::function<float(float)>> derivativeMap = {
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
inline void matrix_based_backprop(Eigen::MatrixXf xs, const Eigen::MatrixXf& ys, const std::vector<Layer>& layers, std::vector<Eigen::VectorXf>& biasErrors, std::vector<Eigen::MatrixXf>& weightErrors,
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

	//Compute error at Lth layer (cross-entropy, TBA: allow hyperparameter for loss function)
	const Eigen::MatrixXf cost_derivative = (xs - ys);
	//const Eigen::MatrixXf zs_derivative = zs.back().unaryExpr(derivativeMap.at(layers.back().type));

	Eigen::MatrixXf delta = cost_derivative;//.cwiseProduct(zs_derivative);

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