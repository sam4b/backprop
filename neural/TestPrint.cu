#include "TestPrint.cuh"


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