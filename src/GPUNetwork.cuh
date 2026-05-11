#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#define EIGEN_NO_CUDA //fix build error on newer eigen
#include <Eigen/Dense>
#include <iostream>

extern "C" void GPUTrain(const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& data,
	const std::vector<size_t>& networkLayout);