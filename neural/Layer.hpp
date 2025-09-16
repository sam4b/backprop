#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <random>

inline nlohmann::json VectorToJson(const Eigen::VectorXf& vector) {
	nlohmann::json out;

	out["elements"] = vector.size();

	for (int i = 0; i < vector.size(); i++) {
		out["data"].push_back(vector(i));
	}

	return out;
}

inline nlohmann::json MatrixToJson(const Eigen::MatrixXf& matrix) {
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

inline Eigen::MatrixXf JsonToMatrix(const nlohmann::json& json) {
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

inline Eigen::VectorXf JsonToVector(const nlohmann::json& json) {
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
		out["activation"] = static_cast<int>(type);

		return out;
	}

	inline static Layer fromJson(const nlohmann::json& json) {
		Layer out;
		out.weights = JsonToMatrix(json["weights"]);
		out.bias = JsonToVector(json["bias"]);
		out.type = json["activation"];
		assert(static_cast<int>(out.type) >= 0 && static_cast<int>(out.type) <= static_cast<int>(ActivationFunctionType::Identity));
		return out;
	}
};

//Xavier weight initialisation cuz sigmoid
inline Layer createLayer(int neuronsIn, int neuronsOut, ActivationFunctionType type, const int randomState) {
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
