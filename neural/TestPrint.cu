#include "TestPrint.cuh"
#include "Network.hpp"

__global__ void f() {
	const int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Thread ID: %i.\n", threadId);
}

__device__ float tanhImplementation(const float x) {
	return (expf(x) - expf(-1.0f * x)) / (expf(x) + expf(-1.0f * x));
}

/*
	c = a . b
*/
__global__ void hamdardProduct(int N, float* a, float* b, float* c) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	c[idx] = a[idx] * b[idx];
}

__global__ void applySigmoidDerivative(int N, float* data) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	data[idx] = sigmoidDerivative(m.data[idx]);
}

/*
	Map a nxm matrix -> nx1 vector by summing rowwise (see: Eigen MatrixXf::rowwise()::sum())
*/
__global__ void sumRows(float* __restrict__ input, float* __restrict__  output, int N, int M) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= N) return;

	float sum = 0.0f;
	for (int i = 0; i < M; i++) {
		sum += input[x + i * N]; //column major storage in eigen and cuda
	}

	output[x] = sum;
}

/*
	C = A + B

	rows(A) = rows(B), cols(A) = cols(B), or undefined behaviour.
*/
__global__ void addMatrices(float* a, float* b, int N, float* c) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId >= N) return;

	c[threadId] = a[threadId] + b[threadId];
}

__global__ void multiplyInPlace(float* a, float scalar, int N) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId >= N) return;

	a[threadId] *= scalar;
}


__device__ float sigmoidImplementation(const float x) {
	return (1.0f / (1.0f + expf(-1.0f * x)));
}

__global__ void applyTanh(const int N, float* data) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID >= N) {
		return;
	}

	data[threadID] = tanhImplementation(data[threadID]);
	
}

__global__ void applySigmoid(const int N, float* data) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID >= N) {
		return;
	}

	data[threadID] = sigmoidImplementation(data[threadID]);
}

__global__ void addVector(const int N, float* a, float* b, float* out) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID >= N) {
		return;
	}

	out[threadID] = a[threadID] + b[threadID];
}

//plan: recreate layer as a wrapper around this that allows for usage with both eigen (pcu side) nad cuda
struct DeviceLayer {
	DeviceMatrix weights;
	DeviceMatrix bias;
	int neuronsIn;
	int neuronsOut;
};

/*
	A thin wrapper around on-device (GPU) matrices.
*/
struct DeviceMatrix {
	float* data;
	int columns;
	int rows;
};

DeviceMatrix copy(DeviceMatrix& a) {
	float* newData;
	const size_t bytes = a.columns * a.rows * sizeof(float);

	cudaMalloc(&newData, bytes);
	cudaMemcpy(newData, a.data, bytes, cudaMemcpyDeviceToDevice);

	DeviceMatrix out;
	out.data = newData;
	out.columns = a.columns;
	out.rows = a.rows;
}

/*
	Memory mgmt: free a and copy b into it
*/
void reuse(DeviceMatrix& a, const DeviceMatrix b) {
	cudaFree(a.data);

	const size_t bytes = b.columns * b.rows * sizeof(float);

	cudaMalloc(&a.data, bytes);

	cudaMemcpy(a.data, b.data, bytes, cudaMemcpyDeviceToDevice);
}

void ApplyActivationDerivative(const DeviceMatrix a, DeviceMatrix& out) {
	out = copy(a);

	applySigmoidDerivative << <1, 1024 >> > (out);
}

/*
	Fills the matrix out with the data A * B (and likewise if A or B are transposed).

*/
void GPUMatMul(DeviceMatrix a, DeviceMatrix b, bool transposeA, bool transposeB, cublasHandle_t& handle
	, DeviceMatrix* out) {
	assert(a.columns == b.rows);

	float* c;
	cudaMalloc(&c, sizeof(float) * a.columns * a.rows);

	const int m = a.rows;
	const int n = b.columns;
	const int k = a.columns;

	//(mxk)(kxn) = (mxn)

	float alpha = 1.0f;
	float beta = 0.0f;

	//Even if A/B is transposed, give the number of rows for lda,ldb, ldc
	cublasSgemm_v2(handle, (transposeA) ? CUBLAS_OP_T : CUBLAS_OP_N, (transposeB) ? CUBLAS_OP_T : CUBLAS_OP_N,
		m, n, k,
		&alpha, a.data, a.rows,
		b.data, b.rows, &beta,
		c, m);


}


void RowwiseSum(const DeviceMatrix in, DeviceMatrix& out) {

}

void ApplyActivationFunction(DeviceMatrix& a) {
	applySigmoid << <1, 1024 >> > (a);
}


void GPUHammardProduct(DeviceMatrix a, DeviceMatrix b, DeviceMatrix result) {

}

void GPUAdd(DeviceMatrix a, DeviceMatrix b, DeviceMatrix result) {

}

void GPUSub(DeviceMatrix a, DeviceMatrix b, DeviceMatrix result) {

}

void GPUScaleMat(DeviceMatrix a, DeviceMatrix result, float scalar) {

}



/*
	xs : the inputs of the mini batch

	ys : the labels of the mini batch

	augmentedBiases : a vector of matrices that consist of the
	bias matrices for the l^th layer augmented with themselves
	m times, where m is the mini batch size,
	so we can do a matrix add for a mini batch

	layers : the layers of the network

	biasErrors : a vector of matrices that consist of the
	errors for each bias vector

	weightErrors : a vector of matrices that consist of the errors
	for each weight matrix

	cublasHandle_t : the handle for cublas 
*/
void GPUBackprop(DeviceMatrix xs, DeviceMatrix ys,
	const std::vector<DeviceLayer>& layers,
	const std::vector<DeviceMatrix>& augmentedBiases,
	cublasHandle_t& handle,
	std::vector<DeviceMatrix>& biasErrors,
	std::vector<DeviceMatrix>& weightErrors) {
	std::vector<DeviceMatrix> zs; //weighted sums
	std::vector<DeviceMatrix> as;

	as.push_back(copy(xs));

	int biasIdx = 0; 

	assert(xs.data);
	assert(augmentedBiases.size() > 0);
	assert(layers.size() == augmentedBiases.size());

	for (int layer = 0; layer < layers.size(); layer++) {
		DeviceMatrix bias = augmentedBiases[layer];

		DeviceMatrix z;
		GPUMatMul(layers[layer].weights, layers[layer].bias, false, false, handle, &z);

		zs.push_back(z);

		DeviceMatrix a = copy(z);
		ApplyActivationFunction(a);

		as.push_back(a);

		reuse(xs, a);
	}

	//Compute the error at the Lth layer
	DeviceMatrix cost_derivative;
	GPUSub(xs, ys, cost_derivative);

	DeviceMatrix zs_derivative;
	ApplyActivationDerivative(zs.back(), zs_derivative);

	DeviceMatrix delta;
	GPUHammardProduct(cost_derivative, zs_derivative, delta);

	RowwiseSum(delta, biasErrors.back());

	GPUMatMul(delta, as[as.size() - 2], false, true, handle,
		&weightErrors.back());

	for (int layer = layers.size() - 2; layer >= 0; layer--) {
		cudaFree(zs_derivative.data);

		ApplyActivationDerivative(zs[layer], zs_derivative);

		DeviceMatrix weight_error_product;
		GPUMatMul(layers[layer + 1].weights, delta, true, false, handle,
			&weight_error_product);

		GPUHammardProduct(zs_derivative, weight_error_product, delta);

		RowwiseSum(delta, biasErrors[layer]);
		GPUMatMul(delta, as[layer], false, true, handle, &weightErrors[layer]);
	}


}


/*
	Creates a network according to the best weight initialisation for the given activation function.
	Returns a vector<DeviceLayer>, consisting of all the layers loaded on device.
*/
std::vector<DeviceLayer> CreateNetwork(const std::vector<int>& layout, ActivationFunctionType function) {
	std::vector<DeviceLayer> out;
	out.reserve(layout.size());

	std::random_device random;
	std::mt19937 mersenne(random());

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

		DeviceLayer layer;

		const size_t weightsInBytes = sizeof(float) * neuronsIn * neuronsOut;

		cudaMalloc(&layer.weights.data, weightsInBytes);
		cudaMemcpy(layer.weights.data, weights, weightsInBytes, cudaMemcpyHostToDevice);
		
		layer.weights.columns = neuronsIn;
		layer.weights.rows = neuronsOut;

		delete[] weights;

		const size_t biasInBytes = sizeof(float) * neuronsOut;

		cudaMalloc(&layer.bias.data, biasInBytes);
		cudaMemcpy(layer.bias.data, bias, biasInBytes, cudaMemcpyHostToDevice);
		
		layer.bias.columns = 1;
		layer.bias.rows = neuronsOut;

		delete[] bias;

		layer.neuronsIn = neuronsIn;
		layer.neuronsOut = neuronsOut;

		out.push_back(layer);

		neuronsIn = neuronsOut;
	}
	return out;
}

/*
	Returns a device matrix. Can easily be converted ot an Eigen::VectorXf/MatrixXf (dependent on whether you're batch processing or not).
*/
DeviceMatrix GPUFeedForward(DeviceMatrix xs, const std::vector<DeviceLayer>& layers, cublasHandle_t& handle) {
	assert(layers.size() >= 1);
	assert(xs.columns = layers[0].neuronsIn);
	assert(xs.columns != 0);
	assert(xs.rows != 0);
	assert(layers[0].neuronsIn != 0);
	assert(layers[0].neuronsOut != 0);

	assert(xs.columns == 1 && "For now, no augmented bias matrices, so nx1 inputs");

	//Network is on-device, so feed-forward.

	//TODO: Preallocation of buffers.

	for (const DeviceLayer& layer : layers) {
		DeviceMatrix out;
		GPUMatMul(layer.weights, xs, false, false, handle, &out); //Can I alias?
		
		cudaFree(xs.data);
		xs = out;
		out.data = nullptr;

		GPUAdd(xs, layer.bias, out);

		cudaFree(xs.data);
		xs = out;

		ApplyActivationFunction(xs);
	}

	return xs;
}


void CUDA_SGD(const std::vector<std::pair<DeviceMatrix, DeviceMatrix>> trainingData, std::vector<DeviceLayer>& layers) {
	
	//for now, assume the training data can fit in vram (todo: add streaming)
	//actually, maybe i should just make the training data into two big matrices


}