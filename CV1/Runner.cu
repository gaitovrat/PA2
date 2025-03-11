#include <cudaDefs.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int THREADS_PER_BLOCK = 256;
constexpr unsigned int MEMBLOCKS_PER_THREADBLOCK = 2;

using namespace std;

__global__ void add1(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c)
{
	unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int skip = gridDim.x * blockDim.x;

	while (offset < length)
	{
		c[offset] = a[offset] + b[offset];
		offset += skip;
	}
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	constexpr unsigned int length = 1 << 10;
	constexpr unsigned int sizeInBytes = length * sizeof(int);

	int* pA = new int[length];
	int* pB = new int[length];
	int* pC = new int[length];

	for (unsigned int i = 0; i < length; ++i)
	{
		pA[i] = i;
		pB[i] = i;
	}

	int* dA = nullptr;
	int* dB = nullptr;
	int* dC = nullptr;
	checkCudaErrors(cudaMalloc((void**)&dA, sizeInBytes));
	checkCudaErrors(cudaMalloc((void**)&dB, sizeInBytes));
	checkCudaErrors(cudaMalloc((void**)&dC, sizeInBytes));

	checkCudaErrors(cudaMemcpy(dA, pA, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dB, pB, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

	dim3 dimBlock{ THREADS_PER_BLOCK, 1, 1 };
	dim3 dimGrid{ 1, 1, 1 };

	add1<<<dimGrid, dimBlock>>>(dA, dB, length, dC);

	checkCudaErrors(cudaMemcpy(pC, dC, sizeInBytes, cudaMemcpyDeviceToHost));
	checkHostMatrix(pC, sizeInBytes, 1, length, "%d ", "C: ");

	SAFE_DELETE_ARRAY(pA);
	SAFE_DELETE_ARRAY(pB);
	SAFE_DELETE_ARRAY(pC);

	SAFE_DELETE_CUDA(dA);
	SAFE_DELETE_CUDA(dB);
	SAFE_DELETE_CUDA(dC);
}