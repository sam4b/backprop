#include "TestPrint.cuh"
#include "Network.hpp"


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


DeviceMatrix copy(const DeviceMatrix& a) {
	float* newData;
	const size_t bytes = a.columns * a.rows * sizeof(float);

	cudaMalloc(&newData, bytes);
	cudaMemcpy(newData, a.data, bytes, cudaMemcpyDeviceToDevice);

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

/*
	Fills the matrix out with the data A * B (and likewise if A or B are transposed).

*/
void GPUMatMul(DeviceMatrix a, DeviceMatrix b, bool transposeA, bool transposeB, cublasHandle_t& handle
	, DeviceMatrix* out) {
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

	float* c;
	cudaMalloc(&c, sizeof(float) * m * n);


	out->data = c;
	out->rows = m;
	out->columns = n;

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

/*
	Result may not alias A or B. This gets thrown into a new output matrix.
*/
void GPUHammardProduct(DeviceMatrix a, DeviceMatrix b, DeviceMatrix& result) {
	assert(a.rows == b.rows);
	assert(a.columns == b.columns);

	cudaMalloc(&result.data, sizeof(float) * a.rows * a.columns);
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
	cudaMalloc(&result.data, sizeof(float) * a.rows * a.columns);
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

	if (idx >= a.rows * b.rows) return;

	b.data[idx] = (alpha * a.data[idx]) + (beta * b.data[idx]);
}

/*
	C = alpha * A + beta * B
	C is emplaced in B.
*/
void GPUAddInPlace(DeviceMatrix a, DeviceMatrix b, float alpha, float beta) {
	dim3 blockDim(16, 16);
	dim3 gridDim(
		(a.columns + blockDim.x - 1) / blockDim.x,
		(a.rows + blockDim.y - 1) / blockDim.y
	);

	AddMatrixInPlace << <gridDim, blockDim >> > (a, b, alpha, beta);
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

	cudaMalloc(&result.data, sizeof(float) * a.rows * a.columns);


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
		GPUMatMul(layers[layer].weights, xs, false, false, handle, &z);
		GPUAddInPlace(bias, z, 1.0f, 1.0f);
		zs.push_back(z);

		DeviceMatrix a = copy(z);
		ApplyActivationInPlace(a, ActivationFunctionType::Sigmoid);

		as.push_back(a);

		cudaDeviceSynchronize();
		cudaFree(xs.data);
		xs = copy(a);
	}

	//Compute the error at the Lth layer
	DeviceMatrix cost_derivative; //MSE: a^L - y
	GPUAdd(xs, ys, cost_derivative, 1.0f, -1.0f);

	DeviceMatrix zs_derivative;
	ApplyActivationDerivative(zs.back(), zs_derivative, ActivationFunctionType::Sigmoid);

	DeviceMatrix delta;
	GPUHammardProduct(cost_derivative, zs_derivative, delta);

	RowwiseSum(delta, biasErrors.back());

	GPUMatMul(delta, as[as.size() - 2], false, true, handle,
		&weightErrors.back());

	for (int layer = layers.size() - 2; layer >= 0; layer--) {
		cudaDeviceSynchronize();
		cudaFree(zs_derivative.data);

		ApplyActivationDerivative(zs[layer], zs_derivative, ActivationFunctionType::Sigmoid);
		zs_derivative.rows = zs[layer].rows;
		zs_derivative.columns = zs[layer].columns;

		DeviceMatrix weight_error_product;
		GPUMatMul(layers[layer + 1].weights, delta, true, false, handle,
			&weight_error_product);

		cudaDeviceSynchronize();
		cudaFree(delta.data);
		GPUHammardProduct(zs_derivative, weight_error_product, delta);

		RowwiseSum(delta, biasErrors[layer]);
		GPUMatMul(delta, as[layer], false, true, handle, &weightErrors[layer]);
	}

	cudaDeviceSynchronize();

	cudaFree(delta.data);

	for (auto& a : as) {
		cudaFree(a.data);
	}

	for (auto& z : zs) {
		cudaFree(z.data);
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

		cudaDeviceSynchronize();
		cudaFree(xs.data);
		xs = out;

		ApplyActivationInPlace(xs, ActivationFunctionType::Sigmoid);
	}

	return xs;
}

std::pair<int, int> maxElementWithIndex(float* data, int n) {
	int max = INT_MIN;
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
	cublasCreate_v2(&handle);

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
		cudaMalloc(&bias.data, bias.rows * bias.columns * sizeof(float));
		cudaMemset(bias.data, 0, bias.rows * bias.columns * sizeof(float));
		biasErrors.push_back(bias);

		DeviceMatrix weights;
		weights.rows = layer.neuronsOut;
		weights.columns = layer.neuronsIn;
		cudaMalloc(&weights.data, weights.rows * weights.columns * sizeof(float));
		cudaMemset(weights.data, 0, weights.rows * weights.columns * sizeof(float));

		weightErrors.push_back(weights);
	}

	const int no_epochs = 30;

	for (int epoch = 0; epoch < no_epochs; epoch++) {
		int batchPtr = 0;

		for (int batch = 0; batch < trainingData.size() / minibatchSize; batch++) {
			std::shuffle(indices.begin(), indices.end(), mersenne);

			assert(biasErrors.size() == weightErrors.size());
			for (int i = 0; i < biasErrors.size(); i++) {
				cudaMemset(biasErrors[i].data, 0, biasErrors[i].columns * biasErrors[i].rows * sizeof(float));
				cudaMemset(weightErrors[i].data, 0, weightErrors[i].columns * weightErrors[i].rows * sizeof(float));
			}


//Do this each iteration for updated biases
//Generate n_lxm augmented bias matrices (where n_l is the size of the neurons outputted by the l^th layer, m the mini batch size)

			std::vector<DeviceMatrix> augmentedBiases;
			augmentedBiases.reserve(layers.size());
			for (const auto& layer : layers) {
				DeviceMatrix bias;

				bias.rows = layer.neuronsOut;
				bias.columns = minibatchSize;

				const int bytes = bias.rows * bias.columns * sizeof(float);

				cudaMalloc(&bias.data, bytes);
				for (int i = 0; i < minibatchSize; i++) { //EVIL! (Loop for i = 0...m-1, copying. SHOULD be fine as column major order (probably slow though...)

					const int offset = i * bias.rows; //move bias rows along each time
					const int size = bias.rows * sizeof(float); //copy the vector once;
					cudaMemcpy(bias.data + offset, layer.bias.data, size, cudaMemcpyDeviceToDevice);
				}

				augmentedBiases.push_back(bias);
			}

			//Create batch

			DeviceMatrix xs;
			xs.rows = trainingData[0].first.rows;
			xs.columns = minibatchSize;
			auto err = cudaMalloc(&xs.data, sizeof(float) * xs.rows * xs.columns);
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err);
				assert(false);
			}
			DeviceMatrix ys;
			ys.rows = trainingData[0].second.rows;
			ys.columns = minibatchSize;
			err = cudaMalloc(&ys.data, sizeof(float) * ys.rows * ys.columns);
			if (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err);
				assert(false);
			}
			for (int i = 0; i < minibatchSize; i++) {
				assert(batchPtr + i < trainingData.size());
				//copy the (batchPtr + i)th input
				cudaMemcpy(xs.data + (i * xs.rows), trainingData[batchPtr + i].first.data, sizeof(float) * xs.rows * 1,
					cudaMemcpyDeviceToDevice);

				cudaMemcpy(ys.data + (i * ys.rows), trainingData[batchPtr + i].second.data, sizeof(float) * ys.rows * 1,
					cudaMemcpyDeviceToDevice);

			}
			batchPtr += minibatchSize;

			GPUBackprop(xs, ys, layers, augmentedBiases, handle,
				biasErrors, weightErrors);

			cudaDeviceSynchronize();
			cudaFree(xs.data);
			cudaFree(ys.data);

			//Update with noisy estimate

			const float scalar = learningRate / (float)minibatchSize;

			assert(layers.size() == biasErrors.size());
			assert(biasErrors.size() == weightErrors.size());

			for (int i = 0; i < layers.size(); i++) {
				GPUScaleMat(biasErrors[i], biasErrors[i], scalar);
				GPUScaleMat(weightErrors[i], weightErrors[i], scalar);

				GPUAddInPlace(biasErrors[i], layers[i].bias, -1.0f, 1.0f); //commutativity of addition is helpful
				GPUAddInPlace(weightErrors[i], layers[i].weights, -1.0f, 1.0f);
			}

			cudaDeviceSynchronize();
			for (auto& bias : augmentedBiases) {
				cudaFree(bias.data);
			}

	
		}


		//Randomly sample sampleSize # of training examples to observe progress each epoch.
		std::shuffle(samples.begin(), samples.end(), mersenne);
		float* x_device = new float[10];
		float* y_device = new float[10];
		int count = 0;
		for (int i = 0; i < sampleSize; i++) {
			const int idx = samples[i];

			const auto in = trainingData[idx].first;
			DeviceMatrix x = GPUFeedForward(in, layers, handle);
			DeviceMatrix y = trainingData[idx].second;

			cudaMemcpy(x_device, x.data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(y_device, y.data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

			const auto [x_max, x_idx] = maxElementWithIndex(x_device, 10);
			const auto [y_max, y_idx] = maxElementWithIndex(y_device, 10);

			if (y_idx == x_idx) count++;

			cudaDeviceSynchronize();
			cudaFree(x.data);
		}

		float percent_correct = ((float)count) * 100.0f / 250.0f;

		std::cout << "Epoch complete, got: " << percent_correct << "% accuracy.\n";

		delete[] x_device;
		delete[] y_device;


	}
}

std::pair<DeviceMatrix, DeviceMatrix> copyPair(const Eigen::VectorXf& x, const Eigen::VectorXf& y) {
	DeviceMatrix x_device;
	
	const size_t x_bytes = x.rows() * x.cols() * sizeof(float);
	cudaMalloc(&x_device.data, x_bytes);
	cudaMemcpy(x_device.data, x.data() /*safe as eigen stores column-major, like cuBLAS, by default*/, x_bytes, cudaMemcpyHostToDevice);
	x_device.rows = x.rows();
	x_device.columns = x.cols();

	DeviceMatrix y_device;

	const size_t y_bytes = y.rows() * y.cols() * sizeof(float);
	cudaMalloc(&y_device.data, y_bytes);
	cudaMemcpy(y_device.data, y.data(), y_bytes, cudaMemcpyHostToDevice);
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

	CUDA_SGD(train, network, 0, 30, 3.0f);
}