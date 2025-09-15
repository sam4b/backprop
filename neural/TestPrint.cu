#include "TestPrint.cuh"

__global__ void f() {
	const int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Thread ID: %i.\n", threadId);
}

void g() {
	f << <1, 5 >> > ();
}