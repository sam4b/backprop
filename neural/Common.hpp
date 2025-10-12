#pragma once
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <cassert>
#include <random>

enum class ActivationFunctionType {
	Sigmoid,
	Tanh,
	LeakyReLU,
	Identity
};

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

