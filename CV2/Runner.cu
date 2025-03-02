#include <cudaDefs.h>

constexpr unsigned int THREADS_PER_BLOCK_DIM = 8;				//=64 threads in block

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

__global__ void initMatrix(float* devMatrix, size_t pitch, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float* rowPtr = (float*)((char*)devMatrix + row * pitch);
        int linearIndex = row * width + col;
		printf("%d %d %d\n", row, col, linearIndex);
        rowPtr[col] = (float)linearIndex;
		*rowPtr = 52;
    }
}

__global__ void incrementMatrix(float* devMatrix, size_t pitch, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float* rowPtr = (float*)((char*)devMatrix + row * pitch);
        rowPtr[col] += 1.0f;
    }
}


int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	float* pMatrix;
	float* dMatrix;
	size_t pitch;

	const unsigned int mRows = 5;
	const unsigned int mCols = 10;
	constexpr unsigned int length = mRows * mCols;
	constexpr unsigned int sizeInBytes = length * sizeof(float);

	//TODO: Allocate Pitch memory
	checkCudaErrors(cudaMallocPitch((void**)&dMatrix, &pitch, mCols * sizeof(float), mRows));

	printf("Matrix size: %d x %d\n", mRows, mCols);
	printf("Device pitch: %llu bytes\n", pitch);
	printf("Element size: %llu bytes\n", sizeof(float));
	printf("Pitch alignment: %llu elements\n", pitch / sizeof(float));

	//TODO: Prepare grid, blocks
	dim3 blockDim(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);
	dim3 gridDim(2, 1);

	//TODO: Call kernel
	initMatrix<<<gridDim, blockDim>>>(dMatrix, pitch, mCols, mRows);
	//incrementMatrix<<<gridDim, blockDim>>>(dMatrix, pitch, mCols, mRows);

	checkCudaErrors(cudaDeviceSynchronize());

	//TODO: Allocate Host memory and copy back Device data
	checkCudaErrors(cudaMallocHost((void**)&pMatrix, sizeInBytes));
	checkCudaErrors(cudaMemcpy2D(pMatrix, mCols * sizeof(float), dMatrix, pitch, mCols * sizeof(float), mRows, cudaMemcpyDeviceToHost));

	//TODO: Check data
	checkHostMatrix(pMatrix, sizeInBytes, mRows, mCols, "%f ", "Matrix: ");

	//TODO: Free memory
	SAFE_DELETE_CUDAHOST(pMatrix);
	SAFE_DELETE_CUDA(dMatrix);

	return 0;
}
