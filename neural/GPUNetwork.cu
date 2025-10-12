#include "GPUNetwork.cuh"
#include "INetwork.h"
#include "Network.hpp"

#define CUDA_TESTING_ON 1


#ifdef CUDA_TESTING_ON
#define CUDA_WRAPPER(CUDACODE) \
{ const cudaError_t err = CUDACODE;\
if (err != cudaSuccess) {\
	std::cout << "CUDA error: " << cudaGetErrorString(err); \
	assert(false);\
}}
#else
#define CUDA_WRAPPER(CUDACODE) CUDACODE;
#endif


#ifdef CUDA_TESTING_ON
#define CUBLAS_WRAPPER(CUBLAS_CODE) \
{const cublasStatus_t err = CUBLAS_CODE;\
 if (err != CUBLAS_STATUS_SUCCESS) {\
std::cout << "cublas error: " << cublasGetStatusString(err);\
assert(false);\
}\
}
#define CUBLAS_WRAPPER(CUBLAS_CODE) CUBLAS_CODE;
#endif


/*
	This function TAKES OWNERSHIP of a. It proceeds to FREE a. If you wish to keep using a after calling this function,
	call copy(a) first.
*/
Eigen::MatrixXf DeviceToEigen(DeviceMatrix a) {
	Eigen::MatrixXf out(a.rows, a.columns);

	const size_t bytes = a.rows * a.columns * sizeof(float);

	CUDA_WRAPPER(cudaMemcpy(out.data(), a.data, bytes, cudaMemcpyDeviceToHost));
	CUDA_WRAPPER(cudaDeviceSynchronize());
	CUDA_WRAPPER(cudaFree(a.data));
	return out;
}



/*
	This function does not take ownership of a. Different semantics as Eigen::VectorXf has RAII.
*/
DeviceMatrix EigenToDevice(const Eigen::MatrixXf& a) {
	DeviceMatrix out;
	out.rows = a.rows();
	out.columns = a.cols();

	const size_t bytes = sizeof(float) * out.rows * out.columns;
	CUDA_WRAPPER(cudaMalloc(&out.data, bytes));

	//Eigen uses column-major too, so can just memcpy
	CUDA_WRAPPER(cudaMemcpy(out.data, a.data(), bytes, cudaMemcpyHostToDevice));

	return out;

}

DeviceMatrix copy(const DeviceMatrix& a) {
	float* newData;
	const size_t bytes = a.columns * a.rows * sizeof(float);

	CUDA_WRAPPER(cudaMalloc(&newData, bytes));
	CUDA_WRAPPER(cudaMemcpy(newData, a.data, bytes, cudaMemcpyDeviceToDevice));

	DeviceMatrix out;
	out.data = newData;
	out.columns = a.columns;
	out.rows = a.rows;

	return out;
}

/*
	No aliasing guarantees with the next two kernels.
*/

__global__ void ApplySigmoid(DeviceMatrix in, DeviceMatrix out) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x; 

	const int idx = row + col * in.rows;

	if (idx >= in.rows * in.columns) return;
	const float sigmoid = 1.0f / (1.0f + expf(-1.0f * in.data[idx]));

	out.data[idx] = sigmoid;
}

__global__ void ApplySigmoidDerivative(DeviceMatrix in, DeviceMatrix out) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;  

	const int idx = row + col * in.rows;

	if (idx >= in.rows * in.columns) return;

	const float sigmoid = 1.0f / (1.0f + expf(-1.0f * in.data[idx]));

	out.data[idx] = (1.0f - sigmoid) * sigmoid;
}


void ApplyActivation(const DeviceMatrix a, DeviceMatrix& out, const ActivationFunctionType type) {
	out = copy(a);

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	ApplySigmoid << <gridDim, blockDim >> > (a, out);

}

void ApplyActivationDerivative(const DeviceMatrix a, DeviceMatrix& out, const ActivationFunctionType) {
	out = copy(a);

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	ApplySigmoidDerivative << <gridDim, blockDim >> > (a, out);
}


void ApplyActivationInPlace(DeviceMatrix a, const ActivationFunctionType type) {
	assert(type == ActivationFunctionType::Sigmoid && "Only sigmoid supported atm");

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);



	ApplySigmoid <<<gridDim, blockDim>>>(a, a);
}

void ApplyActivationDerivativeInPlace(DeviceMatrix a, const ActivationFunctionType type) {
	assert(type == ActivationFunctionType::Sigmoid && "Only sigmoid supported atm");

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);


	ApplySigmoidDerivative <<<gridDim, blockDim>>>(a, a);
}


void GPUMatMulPreAllocated(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c, bool transposeA, bool transposeB, cublasHandle_t& handle) {
	if (transposeA && !transposeB) {
		assert(a.rows == b.rows);
	}
	else if (!transposeA && transposeB) {
		assert(a.columns == b.columns);
	}
	else if (transposeA && transposeB) {
		assert(a.rows == b.columns);
	}
	else {
		assert(a.columns == b.rows);
	}

	const int m = transposeA ? a.columns : a.rows;
	const int k = transposeA ? a.rows : a.columns;
	const int n = transposeB ? b.rows : b.columns;

	assert(c.rows == m);
	assert(c.columns == n);

	//(mxk)(kxn) = (mxn)

	float alpha = 1.0f;
	float beta = 0.0f;

	//Even if A/B is transposed, give the number of rows for lda,ldb, ldc

	CUBLAS_WRAPPER(cublasSgemm_v2(handle, (transposeA) ? CUBLAS_OP_T : CUBLAS_OP_N, (transposeB) ? CUBLAS_OP_T : CUBLAS_OP_N,
		m, n, k,
		&alpha, a.data, a.rows,
		b.data, b.rows, &beta,
		c.data, m));
}
/*
	Fills the matrix out with the data A * B (and likewise if A or B are transposed).

*/
void GPUMatMul(DeviceMatrix a, DeviceMatrix b, bool transposeA, bool transposeB, cublasHandle_t& handle
	, DeviceMatrix* out) {
	assert(out);

	const int m = transposeA ? a.columns : a.rows;
	const int k = transposeA ? a.rows : a.columns;
	const int n = transposeB ? b.rows : b.columns;

	float* c;
	CUDA_WRAPPER(cudaMalloc(&c, sizeof(float) * m * n));


	out->data = c;
	out->rows = m;
	out->columns = n;
	
	GPUMatMulPreAllocated(a, b, *out, transposeA, transposeB, handle);
}

/*
	Map a nxm matrix -> nx1 vector by summing rowwise (see: Eigen MatrixXf::rowwise()::sum())
	input CANNOT alias output (regardless, why would you sum the rows of an nx1 matrix into an nx1 anyway?)
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
	OUT MUST NOT ALIAS IN
*/
void RowwiseSum(const DeviceMatrix in, DeviceMatrix& out) {
	int blockDim = 256;                    
	int gridDim = (in.rows + blockDim - 1) / blockDim;

	sumRows << <gridDim, blockDim >> > (in.data, out.data, in.rows, in.columns);
}


__global__ void HammardProduct(DeviceMatrix a, DeviceMatrix b, DeviceMatrix result) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	const int idx = row + col * a.rows;

	if (row < a.rows && col < a.columns) {
		result.data[idx] = a.data[idx] * b.data[idx];
	}
}

void GPUHammardProductToDest(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c) {
	assert(a.rows == b.rows);
	assert(b.rows == c.rows);

	assert(a.columns == b.columns);
	assert(b.columns == c.columns);

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);


	HammardProduct << <gridDim, blockDim >> > (a, b, c);
}

/*
	Result may not alias A or B. This gets thrown into a new output matrix.
*/
void GPUHammardProduct(DeviceMatrix a, DeviceMatrix b, DeviceMatrix& result) {
	assert(a.rows == b.rows);
	assert(a.columns == b.columns);

	CUDA_WRAPPER(cudaMalloc(&result.data, sizeof(float) * a.rows * a.columns));
	result.rows = a.rows;
	result.columns = a.columns;

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x, 
		(a.rows + blockDim.y - 1) / blockDim.y  
	);


	HammardProduct <<<gridDim, blockDim >> > (a, b, result);
}

/*
	C = A + B
	(C may alias A or B)
*/
__global__ void AddMatrix(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c, float alpha,
	float beta) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	const int idx = row + col * a.rows;

	if (idx >= a.columns * a.rows) return;

	c.data[idx] = a.data[idx] * alpha + beta * b.data[idx];
}

/*
	C = alpha * A + beta * b.
	C MUST NOT ALIAS A OR B.
*/
void GPUAdd(DeviceMatrix a, DeviceMatrix b, DeviceMatrix& result, float alpha, float beta) {
	CUDA_WRAPPER(cudaMalloc(&result.data, sizeof(float) * a.rows * a.columns));
	result.rows = a.rows;
	result.columns = a.columns;

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	AddMatrix << <gridDim, blockDim >> > (a, b, result, alpha, beta);
}

__global__ void AddMatrixInPlace(DeviceMatrix a, DeviceMatrix b, float alpha, float beta) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	const int idx = row + col * a.rows;

	if (idx >= a.rows * a.columns) return;

	b.data[idx] = (alpha * a.data[idx]) + (beta * b.data[idx]);
}

__global__ void AddMatrixToDest(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c, float alpha, float beta) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	const int idx = row + col * a.rows;

	if (idx >= a.rows * a.columns) return;

	c.data[idx] = (alpha * a.data[idx]) + (beta * b.data[idx]);
}

/*
	C = alpha * A + beta * B
	C is emplaced in B.
*/
void GPUAddInPlace(DeviceMatrix a, DeviceMatrix b, float alpha, float beta) {
	assert(a.columns == b.columns);

	assert(a.rows == b.rows);
	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	AddMatrixInPlace << <gridDim, blockDim >> > (a, b, alpha, beta);
}


void GPUAddInPlaceDestination(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c, float alpha, float beta) {
	assert(a.columns == b.columns);
	assert(b.columns == c.columns);

	assert(a.rows == b.rows);
	assert(b.rows == c.rows);

	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	AddMatrixToDest << <gridDim, blockDim >> > (a, b, c, alpha, beta);
}

//No aliasing guarantees.
__global__ void ScaleMatrix(DeviceMatrix a, DeviceMatrix b, float scalar) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	const int idx = row + col * a.rows;

	if (idx >= a.columns * a.rows) return;

	b.data[idx] = scalar * a.data[idx];

}
void GPUScaleMat(DeviceMatrix a, DeviceMatrix& result, float scalar) {
	result.rows = a.rows;
	result.columns = a.columns;

	CUDA_WRAPPER(cudaMalloc(&result.data, sizeof(float) * a.rows * a.columns));


	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	ScaleMatrix << <gridDim, blockDim >> > (a, result, scalar);
}


__global__ void ScaleMatrixInPlace(DeviceMatrix a, float scalar) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	const int idx = row + col * a.rows;

	if (idx >= a.columns * a.rows) return;

	a.data[idx] = scalar * a.data[idx];
}

void GPUScaleMatInPlace(DeviceMatrix a, float scalar) {
	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	ScaleMatrix << <gridDim, blockDim >> > (a, a, scalar);
}


/*
	xs : the inputs of the mini batch

	ys : the labels of the mini batch

	augmentedBiases : a vector of matrices that consist of the
	bias matrices for the l^th layer augmented with themselves
	m times, where m is the mini batch size,
	so we can do a matrix add for a mini batch

	layers : the layers of the network

	memory : a bunch of matrices the backprop algorithm contains, but allocated once for reuse !

	biasErrors : a vector of matrices that consist of the
	errors for each bias vector

	weightErrors : a vector of matrices that consist of the errors
	for each weight matrix

	cublasHandle_t : the handle for cublas 
*/

struct PooledMemory {
	std::vector<DeviceMatrix>& zs; //weighted sums
	std::vector<DeviceMatrix>& zs_derivatives;
	std::vector<DeviceMatrix>& as;
	std::vector<DeviceMatrix>& weight_error_products;
	std::vector<DeviceMatrix>& deltas;
	DeviceMatrix cost_derivative;
};


void GPUBackprop(DeviceMatrix xs, DeviceMatrix ys,
	const std::vector<DeviceLayer>& layers,
	const std::vector<DeviceMatrix>& augmentedBiases,
	PooledMemory& memory,
	cublasHandle_t& handle,
	std::vector<DeviceMatrix>& biasErrors,
	std::vector<DeviceMatrix>& weightErrors) {

	int biasIdx = 0; 
	
	assert(xs.data);
	assert(augmentedBiases.size() > 0);
	assert(layers.size() == augmentedBiases.size());


	assert(memory.as[0].rows == xs.rows);
	assert(memory.as[0].columns == xs.columns);

	CUDA_WRAPPER(cudaMemcpy(memory.as[0].data, xs.data, sizeof(float) * xs.rows * xs.columns, cudaMemcpyDeviceToDevice));
	CUDA_WRAPPER(cudaDeviceSynchronize());

	for (int layer = 0; layer < layers.size(); layer++) {
		DeviceMatrix bias = augmentedBiases[layer];

		//as[0] = activation
		//write into zs[0]
		//write into as[0 + 1]
		// 
		//read as[1]
		//write into zs[1]
		//write into as[2]

		//read as[2]
		//write into zs[2]
		//write into as[3]

		GPUMatMulPreAllocated(layers[layer].weights, memory.as[layer], memory.zs[layer], false, false,
			handle);

		GPUAddInPlace(bias, memory.zs[layer], 1.0f, 1.0f);

		assert(memory.zs[layer].columns == memory.as[layer + 1].columns);
		assert(memory.zs[layer].rows == memory.as[layer + 1].rows);


		const size_t output_size = memory.zs[layer].columns * memory.zs[layer].rows * sizeof(float);
		CUDA_WRAPPER(cudaDeviceSynchronize());

		CUDA_WRAPPER(cudaMemcpy(memory.as[layer + 1].data, memory.zs[layer].data, output_size, cudaMemcpyDeviceToDevice));
		CUDA_WRAPPER(cudaMemcpy(memory.zs_derivatives[layer].data, memory.zs[layer].data, output_size, cudaMemcpyDeviceToDevice));

		ApplyActivationDerivativeInPlace(memory.zs_derivatives[layer], ActivationFunctionType::Sigmoid);
		ApplyActivationInPlace(memory.as[layer + 1], ActivationFunctionType::Sigmoid);

	}

	//Compute the error at the Lth layer
	GPUAddInPlaceDestination(memory.as.back(), ys, memory.cost_derivative, 1.0f, -1.0f); //a^L - y

	GPUHammardProductToDest(memory.cost_derivative, memory.zs_derivatives.back(), memory.deltas.back());
	
	//Write bias error and weight errors
	RowwiseSum(memory.deltas.back(), biasErrors.back());

	GPUMatMulPreAllocated(memory.deltas.back(), memory.as[memory.as.size() - 2], weightErrors.back(), false, true, handle);

	//Going backwards	
	for (int layer = layers.size() - 2; layer >= 0; layer--) {
		GPUMatMulPreAllocated(layers[layer + 1].weights, memory.deltas[layer + 1], memory.weight_error_products[layer],
			true, false, handle);

		//Hammard product being valid between zs_derivatives[layer] and weight_error_products[layer] implies 
		//dimensions are equal.
		GPUHammardProductToDest(memory.zs_derivatives[layer], memory.weight_error_products[layer], memory.deltas[layer]);

		RowwiseSum(memory.deltas[layer], biasErrors[layer]);
		GPUMatMulPreAllocated(memory.deltas[layer], memory.as[layer], weightErrors[layer], false, true, handle);
	}
}


/*
	Creates a network according to the best weight initialisation for the given activation function.
	Returns a vector<DeviceLayer>, consisting of all the layers loaded on device.
*/
std::vector<DeviceLayer> CreateNetwork(const std::vector<size_t>& layout, ActivationFunctionType function) {
	std::vector<DeviceLayer> out;
	out.reserve(layout.size());

	std::random_device random;
	std::mt19937 mersenne(random());

	int neuronsIn = layout[0];
	for (int i = 1; i < layout.size(); i++) {
		const int neuronsOut = layout[i];

		float scale = 0.0f;
		if (function == ActivationFunctionType::Sigmoid || function == ActivationFunctionType::Tanh || function == ActivationFunctionType::Identity) {
			scale = std::sqrt(2.0f / (neuronsIn + neuronsOut));
		}
		else if (function == ActivationFunctionType::LeakyReLU) {
			scale = 2.0f / (float)neuronsIn;

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

		CUDA_WRAPPER(cudaMalloc(&layer.weights.data, weightsInBytes));
		CUDA_WRAPPER(cudaMemcpy(layer.weights.data, weights, weightsInBytes, cudaMemcpyHostToDevice));
		
		layer.weights.columns = neuronsIn;
		layer.weights.rows = neuronsOut;

		delete[] weights;

		const size_t biasInBytes = sizeof(float) * neuronsOut;

		CUDA_WRAPPER(cudaMalloc(&layer.bias.data, biasInBytes));
		CUDA_WRAPPER(cudaMemcpy(layer.bias.data, bias, biasInBytes, cudaMemcpyHostToDevice));
		
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
DeviceMatrix GPUFeedForward(const DeviceMatrix& in, const std::vector<DeviceLayer>& layers, cublasHandle_t& handle) {
	assert(layers.size() >= 1);
	
	DeviceMatrix xs = copy(in);

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
		
		GPUAddInPlace(layer.bias, out, 1.0f, 1.0f);

		CUDA_WRAPPER(cudaDeviceSynchronize());
		CUDA_WRAPPER(cudaFree(xs.data));
		xs = out;

		ApplyActivationInPlace(xs, ActivationFunctionType::Sigmoid);
	}

	return xs;
}

std::pair<int, int> maxElementWithIndex(float* data, int n) {
	float max = -std::numeric_limits<float>::infinity();
	assert(n > 0);

	int idx = -1;

	for (int i = 0; i < n; i++) {
		if (data[i] >= max) {
			idx = i;
			max = data[i];
		}
	}

	return { max, idx };
}

void CUDA_SGD(const std::vector<std::pair<DeviceMatrix, DeviceMatrix>> trainingData, std::vector<DeviceLayer>& layers,
	const int randomState,
	const int minibatchSize,
	const float learningRate) {
	

	//Creating our random sample for testing
	constexpr int sampleSize = 250;
	assert(trainingData.size() >= sampleSize);
	std::vector<int> samples(trainingData.size());
	std::iota(samples.begin(), samples.end(), 0);


	cublasHandle_t handle;
	CUBLAS_WRAPPER(cublasCreate_v2(&handle));

	assert(true);

	//for now, assume the training data can fit in vram (todo: add streaming)
	//actually, maybe i should just make the training data into two big matrices

	//Generate indices to shuffle here, so we don't have to copy the training set.
	std::vector<size_t> indices;
	indices.reserve(trainingData.size());
	for (int i = 0; i < trainingData.size(); i++) {
		indices.push_back(i);
	}

	//indicies only contains elements in the range [0, trainingSet.size() - 1], and trainingSet remains constant - so indexing is safe.

	std::mt19937 mersenne(randomState);
	std::uniform_int_distribution<> sampler(0, trainingData.size() - 1);


	std::vector<DeviceMatrix> biasErrors;
	biasErrors.reserve(layers.size());
	std::vector<DeviceMatrix> weightErrors;
	weightErrors.reserve(layers.size());

	for (const auto& layer : layers) {
		DeviceMatrix bias;
		bias.rows = layer.neuronsOut;
		bias.columns = 1;
		CUDA_WRAPPER(cudaMalloc(&bias.data, bias.rows * bias.columns * sizeof(float)));
		CUDA_WRAPPER(cudaMemset(bias.data, 0, bias.rows * bias.columns * sizeof(float)));

		biasErrors.push_back(bias);

		DeviceMatrix weights;
		weights.rows = layer.neuronsOut;
		weights.columns = layer.neuronsIn;
		CUDA_WRAPPER(cudaMalloc(&weights.data, weights.rows * weights.columns * sizeof(float)));
		CUDA_WRAPPER(cudaMemset(weights.data, 0, weights.rows * weights.columns * sizeof(float)));

		weightErrors.push_back(weights);
	}

	const int no_epochs = 30;


	//Setup augmented biases (pool to avoid reallocating for each batch).

	std::vector<DeviceMatrix> augmentedBiases;
	augmentedBiases.reserve(layers.size());
	for (const auto& layer : layers) {
		DeviceMatrix bias;
		bias.rows = layer.neuronsOut;
		bias.columns = minibatchSize;

		const int bytes = bias.rows * bias.columns * sizeof(float);

		CUDA_WRAPPER(cudaMalloc(&bias.data, bytes));
		augmentedBiases.push_back(bias);
	}


	//Pool xs and ys (size doesn't change / batch)

	DeviceMatrix xs;
	xs.rows = trainingData[0].first.rows;
	xs.columns = minibatchSize;
	CUDA_WRAPPER(cudaMalloc(&xs.data, sizeof(float) * xs.rows * xs.columns));

	DeviceMatrix ys;
	ys.rows = trainingData[0].second.rows;
	ys.columns = minibatchSize;
	CUDA_WRAPPER(cudaMalloc(&ys.data, sizeof(float) * ys.rows * ys.columns));



	/*
		A bunch of horrible stuff to pool memory
	*/
	DeviceMatrix cost_derivative;
	cost_derivative.columns = ys.columns;
	cost_derivative.rows = ys.rows;
	CUDA_WRAPPER(cudaMalloc(&cost_derivative.data, sizeof(float) * ys.columns * ys.rows));

	std::vector<DeviceMatrix> zs; //weighted sums
	std::vector<DeviceMatrix> zs_derivatives;
	std::vector<DeviceMatrix> as;
	std::vector<DeviceMatrix> weight_error_products;
	std::vector<DeviceMatrix> deltas;

	as.resize(layers.size() + 1);
	zs.resize(layers.size());
	zs_derivatives.resize(layers.size());
	weight_error_products.resize(layers.size());
	deltas.resize(layers.size());

	as[0].rows = xs.rows;
	as[0].columns = xs.columns;
	CUDA_WRAPPER(cudaMalloc(&as[0].data, xs.rows * xs.columns * sizeof(float)));


	for (int layer = 0; layer < layers.size(); layer++) {
		const int neuronsIn = layers[layer].neuronsIn;
		const int neuronsOut = layers[layer].neuronsOut;

		//dim (neuronsOutxneuronsIn);

		const size_t activation_size_bytes = sizeof(float) * neuronsOut * minibatchSize;

		DeviceMatrix z;
		z.rows = neuronsOut;
		z.columns = minibatchSize;
		CUDA_WRAPPER(cudaMalloc(&z.data, activation_size_bytes));

		zs[layer] = z;

		DeviceMatrix zs_derivative = z;
		zs_derivative.data = nullptr; //just in case..
		CUDA_WRAPPER(cudaMalloc(&zs_derivative.data, activation_size_bytes));

		zs_derivatives[layer] = zs_derivative;

		DeviceMatrix weight_error_product = z;
		weight_error_product.data = nullptr; //just in case..
		CUDA_WRAPPER(cudaMalloc(&weight_error_product.data, activation_size_bytes));

		weight_error_products[layer] = weight_error_product;

		DeviceMatrix a;
		a.rows = neuronsOut;
		a.columns = minibatchSize;
		CUDA_WRAPPER(cudaMalloc(&a.data, activation_size_bytes));

		as[layer + 1] = a; //layer + 1 to account for putting the initial input into as.

		//Since for all layers l in 0..L, the error (delta[layer]) consists of a Hadamard product of z^l,
		//then the shape is just z^l's shape

		DeviceMatrix delta;
		delta.rows = z.rows;
		delta.columns = minibatchSize;
		CUDA_WRAPPER(cudaMalloc(&delta.data, activation_size_bytes));

		deltas[layer] = delta;
	}

	PooledMemory memory{
		.zs = zs,
		.zs_derivatives = zs_derivatives,
		.as = as,
		.weight_error_products = weight_error_products,
		.deltas = deltas,
		.cost_derivative = cost_derivative
	};



	for (int epoch = 0; epoch < no_epochs; epoch++) {
		int batchPtr = 0;

		for (int batch = 0; batch < trainingData.size() / minibatchSize; batch++) {
			std::shuffle(indices.begin(), indices.end(), mersenne);

			assert(biasErrors.size() == weightErrors.size());
			for (int i = 0; i < biasErrors.size(); i++) {
				CUDA_WRAPPER(cudaMemset(biasErrors[i].data, 0, biasErrors[i].columns * biasErrors[i].rows * sizeof(float)));
				CUDA_WRAPPER(cudaMemset(weightErrors[i].data, 0, weightErrors[i].columns * weightErrors[i].rows * sizeof(float)));
			}


			//Do this each iteration for updated biases
			//Generate n_lxm augmented bias matrices (where n_l is the size of the neurons outputted by the l^th layer, m the mini batch size)

			assert(layers.size() == augmentedBiases.size());
			for (int i = 0; i < layers.size(); i++) {
				DeviceMatrix bias = augmentedBiases[i];
				for (int j = 0; j < minibatchSize; j++) { //EVIL! (Loop for i = 0...m-1, copying. SHOULD be fine as column major order (probably slow though...)

					const int offset = j * bias.rows; //move bias rows along each time
					const int size = bias.rows * sizeof(float); //copy the vector once;
					CUDA_WRAPPER(cudaMemcpy(bias.data + offset, layers[i].bias.data, size, cudaMemcpyDeviceToDevice));;
				}
			}

			//Create batch
			assert(batchPtr + minibatchSize < trainingData.size() && "Loop never goes out of bounds.");
			for (int i = 0; i < minibatchSize; i++) {
				//copy the (batchPtr + i)th input
				CUDA_WRAPPER(cudaMemcpy(xs.data + (i * xs.rows), trainingData[batchPtr + i].first.data, sizeof(float) * xs.rows * 1,
					cudaMemcpyDeviceToDevice));

				CUDA_WRAPPER(cudaMemcpy(ys.data + (i * ys.rows), trainingData[batchPtr + i].second.data, sizeof(float) * ys.rows * 1,
					cudaMemcpyDeviceToDevice));

			}
			batchPtr += minibatchSize;

			GPUBackprop(xs, ys, layers, augmentedBiases, memory, handle,
				biasErrors, weightErrors);

			CUDA_WRAPPER(cudaDeviceSynchronize());

			//Update with noisy estimate

			const float scalar = learningRate / (float)minibatchSize;

			assert(layers.size() == biasErrors.size());
			assert(biasErrors.size() == weightErrors.size());

			for (int i = 0; i < layers.size(); i++) {
				GPUScaleMatInPlace(biasErrors[i], scalar);
				GPUScaleMatInPlace(weightErrors[i], scalar);

				GPUAddInPlace(biasErrors[i], layers[i].bias, -1.0f, 1.0f); //commutativity of addition is helpful
				GPUAddInPlace(weightErrors[i], layers[i].weights, -1.0f, 1.0f);
			}
		}

		//Randomly sample sampleSize # of training examples to observe progress each epoch.
		CUDA_WRAPPER(cudaDeviceSynchronize());
		std::shuffle(samples.begin(), samples.end(), mersenne);

		const size_t label_size = trainingData[0].second.rows; //For multi-class classification, we take the argmax of yhat and y 
		//and make sure they're equal. For regression, MSE should be implemented.
		float* x_device = new float[label_size];
		float* y_device = new float[label_size];
		int count = 0;
		for (int i = 0; i < sampleSize; i++) {
			const int idx = samples[i];

			const auto in = trainingData[idx].first;
			DeviceMatrix x = GPUFeedForward(in, layers, handle);
			DeviceMatrix y = trainingData[idx].second;

			CUDA_WRAPPER(cudaDeviceSynchronize());
			CUDA_WRAPPER(cudaMemcpy(x_device, x.data, label_size * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_WRAPPER(cudaMemcpy(y_device, y.data, label_size * sizeof(float), cudaMemcpyDeviceToHost));

			const auto [x_max, x_idx] = maxElementWithIndex(x_device, label_size);
			const auto [y_max, y_idx] = maxElementWithIndex(y_device, label_size);

			if (y_idx == x_idx) count++;

			CUDA_WRAPPER(cudaDeviceSynchronize());
			CUDA_WRAPPER(cudaFree(x.data));
		}

		float percent_correct = ((float)count) * 100.0f / 250.0f;

		std::cout << "Epoch complete, got: " << percent_correct << "% accuracy.\n";

		delete[] x_device;
		delete[] y_device;

	
	}
	//Free mini batch data.
	CUDA_WRAPPER(cudaFree(xs.data));
	CUDA_WRAPPER(cudaFree(ys.data));

	for (auto& bias : augmentedBiases) {
		CUDA_WRAPPER(cudaFree(bias.data));
	}

	for (auto& a : memory.as) {
		CUDA_WRAPPER(cudaFree(a.data));
	}

	CUDA_WRAPPER(cudaFree(memory.cost_derivative.data));

	for (auto& i : memory.weight_error_products) {
		CUDA_WRAPPER(cudaFree(i.data));
	}

	for (auto& i : memory.zs) {
		CUDA_WRAPPER(cudaFree(i.data));
	}

	for (auto& i : memory.zs_derivatives) {
		CUDA_WRAPPER(cudaFree(i.data));
	}

	for (auto& i : memory.deltas) {
		CUDA_WRAPPER(cudaFree(i.data));
	}

}


std::pair<DeviceMatrix, DeviceMatrix> copyPair(const Eigen::VectorXf& x, const Eigen::VectorXf& y) {
	DeviceMatrix x_device;
	
	const size_t x_bytes = x.rows() * x.cols() * sizeof(float);
	CUDA_WRAPPER(cudaMalloc(&x_device.data, x_bytes));
	CUDA_WRAPPER(cudaMemcpy(x_device.data, x.data() /*safe as eigen stores column-major, like cuBLAS, by default*/, x_bytes, cudaMemcpyHostToDevice));
	x_device.rows = x.rows();
	x_device.columns = x.cols();

	DeviceMatrix y_device;

	const size_t y_bytes = y.rows() * y.cols() * sizeof(float);
	CUDA_WRAPPER(cudaMalloc(&y_device.data, y_bytes));
	CUDA_WRAPPER(cudaMemcpy(y_device.data, y.data(), y_bytes, cudaMemcpyHostToDevice));
	y_device.rows = y.rows();
	y_device.columns = y.cols();

	return { x_device, y_device };

	
}

/*
	Assumption: All training data can fit into the GPU VRAM. TODO: Add a streaming option.
*/
void CopyData(const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& data,
	std::vector<std::pair<DeviceMatrix, DeviceMatrix>>& out) {


	out.reserve(data.size());
	for (const auto& [x, y] : data) {
		out.push_back(copyPair(x, y));
	}
}

class GPUNetwork : public Network {
public:
	GPUNetwork()
};

/*
	Creates a neural network on the GPU (sigmoid activation function) and trains it on the data provided.
*/
void GPUTrain(const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& data, 
	const std::vector<size_t>& networkLayout
	) {

	std::vector<std::pair<DeviceMatrix, DeviceMatrix>> train;
	CopyData(data, train); //add random state later
	std::cout << "Copied data successfully.\n";

	assert(train.size() == data.size());

	std::vector<DeviceLayer> network = CreateNetwork(networkLayout, ActivationFunctionType::Sigmoid);

	CUDA_SGD(train, network, 0, 30, 0.1f);
}