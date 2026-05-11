#pragma once
#include "INetwork.h"
#include "DeviceMatrix.hpp"
#include "GPUNetwork.cuh"

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
private:
	std::vector<DeviceLayer> layers;
};