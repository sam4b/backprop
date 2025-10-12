#pragma once
#include "Common.hpp"

struct CPULayer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf bias;
	ActivationFunctionType type;

	nlohmann::json tojson() const {
		nlohmann::json out;

		out["weights"] = MatrixToJson(weights);
		out["bias"] = VectorToJson(bias);
		out["activation_function"] = static_cast<int>(type);
		return out;
	}

	inline static CPULayer fromJson(const nlohmann::json& json) {
		CPULayer out;
		out.weights = JsonToMatrix(json["weights"]);
		out.bias = JsonToVector(json["bias"]);
		const int type = json["activation_function"];
		assert(0 <= type <= ActivationFunctionType::Identity);
		return out;
	}
};

inline CPULayer createLayer(const int neuronsIn, const int neuronsOut, const ActivationFunctionType type) {
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