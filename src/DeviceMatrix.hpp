#pragma once

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