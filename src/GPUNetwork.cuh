#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#define EIGEN_NO_CUDA //fix build error on newer eigen
#include <Eigen/Dense>
#include "DeviceMatrix.hpp"
#include <iostream>

std::vector<DeviceLayer> GPUTrain(const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& data,
	const std::vector<size_t>& networkLayout);
 
Eigen::MatrixXf predict(const std::vector<DeviceLayer>& network, Eigen::MatrixXf data);