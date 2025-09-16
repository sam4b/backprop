#include "TestPrint.cuh"
#include "Network.hpp"

__global__ void f() {
	const int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Thread ID: %i.\n", threadId);
}

__device__ float tanhImplementation(const float x) {
	return (expf(x) - expf(-1.0f * x)) / (expf(x) + expf(-1.0f * x));
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

		RawLayer layer;
		layer.bias = bias;
		layer.weights = weights;
		layer.neuronsIn = neuronsIn;
		layer.neuronsOut = neuronsOut;

		out.push_back(layer);

		neuronsIn = neuronsOut;
	}
	return out;
}

std::vector<RawLayer> CUDA_LoadNetwork(const std::vector<RawLayer>& layers) {
	std::vector<RawLayer> device_layers;
	device_layers.reserve(layers.size());

	for (const auto& layer : layers) {
		float* device_weights;
		float* device_bias;

		cudaMalloc(&device_weights, sizeof(float) * layer.neuronsIn * layer.neuronsOut);
		cudaMalloc(&device_bias, sizeof(float) * layer.neuronsOut);

		cudaMemcpy(device_weights, layer.weights, sizeof(float) * layer.neuronsIn * layer.neuronsOut, cudaMemcpyHostToDevice);
		cudaMemcpy(device_bias, layer.bias, sizeof(float) * layer.neuronsOut, cudaMemcpyHostToDevice);

		RawLayer device_layer;
		device_layer.weights = device_weights;
		device_layer.bias = device_bias;
		device_layer.neuronsIn = layer.neuronsIn;
		device_layer.neuronsOut = layer.neuronsOut;

		device_layers.push_back(device_layer);
	}

	return device_layers;
}

/*
	c = a . b
*/
__global__ void hamdardProduct(int N, float* a, float* b, float* c) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	c[idx] = a[idx] * b[idx];
}

__global__ void applySigmoidDerivative(Matrix m) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= m.rows * m.columns) return;

	//m.data[idx] = sigmoidDerivative(m.data[idx]);
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


void GPUBackprop(std::vector<RawLayer>& layers, const std::vector<Matrix>& biasMatrices,
	const std::vector<Matrix>& biasErrors, const std::vector<Matrix>& weightErrors,
	int minibatchsize
) {
	std::vector<Matrix> zs;
	std::vector<Matrix> as;


	cublasHandle_t handle;

	std::vector<float*> deltas;

	int deltaIdx = 0;

	for (int layer = biasMatrices.size() - 2; layer >= 0; layer--) {
		float* weight_error_product;
		cudaMalloc(&weight_error_product, -1);

		layers[layer + 1].weights.transpose()* delta[deltaIdx];


		float* delta;
		cudaMalloc(&delta, sizeof(float) * layers[layer + 1].neuronsOut * layers[layer + 1].neuronsIn);

		//c = a * b where a is an mxk, b is a kxn and c is an mxn
		//a = weights
		//b = delta[deltaIdx]
					
							//w^{l+1}^T  //delta remains non transposed

		const float alpha = 1.0f;
		const float beta = 0.0f;
		//delta is a vector, well actually no its not given that i have a matrix so what is k, mini bathc size?
		//yes
		cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			//m,n,k
			layers[layer+1].neuronsOut, minibatchsize, layers[layer+1].neuronsIn,
		&alpha, layers[layer+1].weights, layers[layer+1].neuronsOut,
			deltas[deltaIdx], minibatchsize, &beta,
			delta, layers[layer + 1].neuronsOut
		); //:vomit_emoji:

		applySigmoidDerivative<<<1,1024>>>(zs[layer]); //sigma'

		float* delta;
		cudaMalloc(&delta, sizeof(float) * zs[layer].rows);
		
		sumRows<<<1,500>>>(zs[layer].data, delta, zs[layer].rows, zs[layer].columns);

		hamdardProduct<<<1,500>>>(zs[layer].rows, delta, weight_error_product, delta);

		

	}

	
}

float* CUDA_FeedForward(const std::vector<RawLayer>& layers, float* vector, const int rows) {
	assert(rows == layers[0].neuronsIn);
	//assume this is a rowsx1 matrix for now 
	
	//copy weights and biases

	std::vector<RawLayer> device_network = CUDA_LoadNetwork(layers);

	float* device_vector;
	cudaMalloc(&device_vector, sizeof(float) * rows);
	cudaMemcpy(device_vector, vector, sizeof(float) * rows, cudaMemcpyHostToDevice);
	
	//TODO: Preallocate all the vectors to be stored for reuse so 0 allocations done on feedforward.

	cublasHandle_t handle;
	cublasCreate_v2(&handle);


	for (RawLayer layer : device_network) {
		float* output;
		//neuronsOut x neuronsIn
		cudaMalloc(&output, sizeof(float) * layer.neuronsOut);
	

		//w size = layer.neuronsOut * layer.neuronsIn
		const int m = layer.neuronsOut;

		//x size = layer.neuronsIn * 1
		const int n = 1;
		
		//wx size = layer.neuronsOut * 1
		const int k = layer.neuronsIn; 

		float wx_scalar = 1.0f, c_scalar = 0.0f;

		//w * x, as c = a * b where a is an mxk, b is a kxn and c is an mxn
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &wx_scalar, layer.weights, layer.neuronsOut, device_vector, layer.neuronsIn, &c_scalar, output, layer.neuronsOut);

		//now add bias and then apply activation
		
		addVector << <1, 1024 >> > (layer.neuronsOut, layer.bias, output, output); //i hope aliasing isn't an issue.

		applySigmoid << <1, 1024 >> > (layer.neuronsOut, output);


		//store result in device vector, free output so we can reues it next iteration
		
		cudaFree(device_vector);
		cudaMalloc(&device_vector, sizeof(float) * layer.neuronsOut);
		cudaMemcpy(device_vector, output, sizeof(float) * layer.neuronsOut, cudaMemcpyDeviceToDevice);
		cudaFree(output);	

	}

	//result now stored in device_vector

	float* result = new float[layers.back().neuronsIn * layers.back().neuronsOut];
	cudaMemcpy(result, device_vector, sizeof(float) * layers.back().neuronsIn * layers.back().neuronsOut, cudaMemcpyDeviceToHost);
	cudaFree(device_vector);
	cublasDestroy_v2(handle);
	return result;


}

void CUDA_SGD(const std::vector<std::pair<Matrix, Matrix>> trainingData, std::vector<RawLayer>& layers) {
	
	//for now, assume the training data can fit in vram (todo: add streaming)
	//actually, maybe i should just make the training data into two big matrices


}

void g() {


	const Eigen::MatrixXf toBeSummed{
		{1,2,3,4,5},
		{5,6,7,8,6},
		{10,15,20, 25,7},
		{100,200,300, 400,8}
	};

	float* input;
	
	printf("Rows:%d,Cols:%d\n", toBeSummed.rows(), toBeSummed.cols());

	const int input_size_bytes = sizeof(float) * toBeSummed.rows() * toBeSummed.cols();

	cudaMalloc(&input, input_size_bytes);

	cudaMemcpy(input, toBeSummed.data(), input_size_bytes, cudaMemcpyHostToDevice);

	const int output_size_bytes = sizeof(float) * toBeSummed.rows() * 1; //1 not necessary but matrix dimensions
	float* output;
	cudaMalloc(&output, output_size_bytes);
	
	sumRows << <1, 10 >> >(input, output, toBeSummed.rows(), toBeSummed.cols());

	float* device_output = new float[toBeSummed.rows()];
	cudaMemcpy(device_output, output, output_size_bytes, cudaMemcpyDeviceToHost);

	Eigen::MatrixXf out_mat = Eigen::Map<Eigen::MatrixXf>(device_output, toBeSummed.rows(), 1);

	Eigen::MatrixXf expected_mat = toBeSummed.rowwise().sum();

	std::cout << out_mat << std::endl << expected_mat << std::endl;

	return;
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

	cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 3, 3, &ab_scaler, a, 2, b, 3, &c_scaler, c, 2);

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