#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <Eigen/Dense>
#include <iostream>
#include "INetwork.h"


/*
	A thin wrapper around on-device (GPU) matrices.
*/
struct DeviceMatrix {
	float* data;
	int columns;
	int rows;
};

//plan: recreate layer as a wrapper around this that allows for usage with both eigen (pcu side) nad cuda
struct DeviceLayer {
	DeviceMatrix weights;
	DeviceMatrix bias;
	int neuronsIn;
	int neuronsOut;
};


extern "C" void GPUTrain(const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& data,
	const std::vector<size_t>& networkLayout);