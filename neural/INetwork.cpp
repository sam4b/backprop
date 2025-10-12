#include "INetwork.h"

Eigen::VectorXf DeviceNetwork::predict(const Eigen::VectorXf x) const
{
    return Eigen::VectorXf();
}

nlohmann::json DeviceNetwork::toJson() const
{
    return nlohmann::json();
}

void DeviceNetwork::loadFromJson(const nlohmann::json& json) const
{
}

void CPUNetwork::train(const EigenDataset& train, const Hyperparameters params, const EigenDataset& validation)
{
}

void CPUNetwork::train(const EigenDataset& train, const Hyperparameters params)
{
}

Eigen::VectorXf CPUNetwork::predict(const Eigen::VectorXf x) const
{
    return Eigen::VectorXf();
}

nlohmann::json CPUNetwork::toJson() const
{
    return nlohmann::json();
}

void CPUNetwork::loadFromJson(const nlohmann::json& json) const
{
}
