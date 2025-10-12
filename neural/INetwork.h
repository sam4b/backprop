#pragma once
#include "Common.hpp"

struct Hyperparameters {
	std::vector<int> hiddenLayers;
	float learningRate;
	ActivationFunctionType activationFunction;
	//TBA: cost function
	//TBA: Optimizer (ADAM implementation)
	int epochs;
	int batchSize;
};

using EigenDataset = std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>; //each element is (x, label).

class INetwork {
public:
	virtual void train(const EigenDataset& train, const Hyperparameters params, const EigenDataset& validation) = 0;
	virtual void train(const EigenDataset& train, const Hyperparameters params) = 0;
	virtual Eigen::VectorXf predict(const Eigen::VectorXf x) const = 0;
	virtual nlohmann::json toJson() const = 0;
	virtual void loadFromJson(const nlohmann::json& json) const = 0;
};

class CPUNetwork : public INetwork {
public:
	// Inherited via INetwork
	void train(const EigenDataset& train, const Hyperparameters params, const EigenDataset& validation) override;
	void train(const EigenDataset& train, const Hyperparameters params) override;
	Eigen::VectorXf predict(const Eigen::VectorXf x) const override;
	nlohmann::json toJson() const override;
	void loadFromJson(const nlohmann::json& json) const override;
};

class DeviceNetwork : public INetwork {
public:
	// Inherited via INetwork
	void train(const EigenDataset& train, const Hyperparameters params, const EigenDataset& validation) override;
	void train(const EigenDataset& train, const Hyperparameters params) override;
	Eigen::VectorXf predict(const Eigen::VectorXf x) const override;
	nlohmann::json toJson() const override;
	void loadFromJson(const nlohmann::json& json) const override;
	DeviceMatrix predict(DeviceMatrix predict) {

	}

};