#include "TestPrint.cuh"
#include "Network.hpp"

__global__ void f() {
	const int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Thread ID: %i.\n", threadId);
}

__device__ float tanhImplementation(const float x) {
	return (expf(x) - expf(-1.0f * x)) / (expf(x) + expf(-1.0f * x));
}

__global__ void applyTanh(const int N, float* data) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID >= N) {
		return;
	}

	data[threadID] = tanhImplementation(data[threadID]);
	
}

//plan: recreate layer as a wrapper around this that allows for usage with both eigen (pcu side) nad cuda
struct RawLayer {
	float* weights;
	float* bias;
	int neuronsIn;
	int neuronsOut;
};

struct Matrix {
	float* data;
	int columns;
	int rows;
};


std::vector<RawLayer> createNetwork(const std::vector<int>& layout, ActivationFunctionType function) {
	std::vector<RawLayer> out;
	out.reserve(layout.size());

	std::random_device random();
	std::mt19937 mersenne(random);

	int neuronsIn = layout[0];
	for (int i = 1; i < layout.size(); i++) {
		const int neuronsOut = layout[i];

		float scale = 0.0f;
		if (function == ActivationFunctionType::Sigmoid || function == ActivationFunctionType::Tanh || function == ActivationFunctionType::Identity) {
			float scale = std::sqrt(2.0f / (neuronsIn + neuronsOut));

		}
		else if (function == ActivationFunctionType::LeakyReLU) {
			float scale = 2.0f / (float)neuronsIn;

		}
		else {
			assert(false);
		}
		std::normal_distribution<float> dist(0.0f, scale);

		float* weights = new float[neuronsIn * neuronsOut];
		for (int j = 0; j < neuronsIn * neuronsOut; j++) {
			weights[j] = dist(mersenne);
		}

		float* bias = new float[neuronsOut];
		std::memset(bias, 0, sizeof(float) * neuronsOut);

		RawLayer layer;
		layer.bias = bias;
		layer.weights = weights;
		layer.neuronsIn = neuronsIn;
		layer.neuronsOut = neuronsOut;

		out.push_back(layer);

		neuronsIn = neuronsOut;
	}
}

void CUDA_FeedForward(const std::vector<RawLayer>& layers, float* vector, const int rows) {
	assert(rows == layers[0].neuronsIn);
	//assume this is a rowsx1 matrix for now 
	
	//copy weights and biases



}

void CUDA_SGD(const std::vector<std::pair<Matrix, Matrix>> trainingData, std::vector<RawLayer>& layers) {
	
	//for now, assume the training data can fit in vram (todo: add streaming)
	//actually, maybe i should just make the training data into two big matrices


}

void g() {
	f << <1, 5 >> > ();

	Eigen::MatrixXf mat{
		{0.0f, 200.0f, 3.0f},
		{0.01f, 5.0f, 6.0f}
	}; //2x3
	mat *= 1.0f / 5.0f;

	Eigen::MatrixXf mat2{
		{3.0f, 4.0f, 5.0f},
		{5.0f, 600.0f, 4.0f},
		{20.0f, 1.0f, 3.0f}
	};
	mat2 *= 1.0f / 100.0f;
	//3x3

	//= 2x3 mat

	const auto res = (mat * mat2).unaryExpr([](const float x) -> float {
		return (exp(x) - exp(-1.0f * x)) / (exp(x) + exp(-1.0f * x));
		});;

	std::cout << res << std::endl;

	float* a, * b, * c;

	float* c_host = new float[6];

	cudaMalloc(&a, sizeof(float) * 2 * 3);
	cudaMalloc(&b, sizeof(float) * 3 * 3);
	cudaMalloc(&c, sizeof(float) * 2 * 3);

	cudaMemcpy(a, mat.data(), sizeof(float) * 2 * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(b, mat2.data(), sizeof(float) * 3 * 3, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	const float ab_scaler = 1.0f, c_scaler = 0.0f;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 3, 3, &ab_scaler, a, 2, b, 3, &c_scaler, c, 2);

	applyTanh << <1, 6 >> > (6, c);

	cudaMemcpy(c_host, c, sizeof(float) * 6, cudaMemcpyDeviceToHost);

	cublasDestroy_v2(handle);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	Eigen::MatrixXf cMat = Eigen::Map<Eigen::MatrixXf>(c_host, 2, 3);
	std::cout << cMat << std::endl;

	for (int i = 0; i < 6; i++) {
		std::cout << c_host[i] << ",";
	};

	
}