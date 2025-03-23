#include <cudaDefs.h>
#include <benchmark.h>
#include <random>
#include <iostream>
#include <limits>

using real_t = float;
using discrete_t = uint32_t;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

static constexpr size_t N = 1 << 20;
static constexpr size_t M = 128;
static constexpr discrete_t MAX_DISCRETE_VALUE = std::numeric_limits<discrete_t>::max();

__host__ real_t* createData(const size_t length)
{
	std::cout << "Create data of length: " << length << std::endl;
    real_t* data = new real_t[length];

    std::random_device rd;
    std::uniform_real_distribution<real_t> dist(-100.f, 100.f);

    for (int i = 0; i < length; ++i)
    {
        data[i] = dist(rd);
    }

    return data;
}

__host__ void findMinMaxByDimension(real_t* matrix, real_t* min_vals, real_t* max_vals, size_t num_vectors, size_t dim) 
{
	std::cout << "Find min max by dimension" << std::endl;

    for (size_t d = 0; d < dim; d++) 
    {
        min_vals[d] = FLT_MAX;
        max_vals[d] = FLT_MIN;

        for (size_t i = 0; i < num_vectors; i++) 
        {
            real_t val = matrix[i * dim + d];
            min_vals[d] = std::min(min_vals[d], val);
            max_vals[d] = std::max(max_vals[d], val);
        }
    }
}

__global__ void discretizeKernel(
        real_t* __restrict__ input,
        const int inputPitch,
        discrete_t* __restrict__ output,
        const int outputPitch,
        real_t* __restrict__ minValues, 
        real_t* __restrict__ maxValues, 
        const int cols, 
        const int rows,
        const discrete_t maxDiscreteValue) 
{
    int col = threadIdx.x;
    int row = 0;

    while (col < cols && row < rows) {
        real_t* rowInput = (real_t*)((char*)input + row * inputPitch);
        discrete_t *rowOutput = (discrete_t*)((char*)output + row * outputPitch);

        real_t value = rowInput[col];
        real_t minValue = minValues[col];
        real_t maxValue = maxValues[col];
        real_t range = maxValue - minValue;

        real_t normalized = (value - minValue) / range;
        rowOutput[col] = static_cast<discrete_t>(normalized * maxDiscreteValue);
  
        row += 1;
    }
}

__global__ void distanceToOriginKernel(
    discrete_t* __restrict__ input, 
    const int pitch,
    unsigned long long* __restrict__ distances, 
    const int cols, 
    const int rows) 
{
	int idx = threadIdx.x;
	int offset = blockDim.x;

	while (idx < rows) {
        unsigned long long distance = 0;
		discrete_t* row = (discrete_t*)((char*)input + idx * pitch);

		for (int d = 0; d < cols; d++) {
			unsigned long long val = row[d];
			distance += val * val;
		}

		distances[idx] = distance;
		idx += offset;
	}
}

__global__ void findMaxDistanceKernel(unsigned long long* __restrict__ distances, unsigned long long* __restrict__ maxDistance, const int size) {
    __shared__ unsigned long long sMaxDistances[256];
    unsigned int tid = threadIdx.x;
	int offset = blockDim.x;
    unsigned long long localMax = 0;

    for (int i = tid; i < size; i += offset) {
        localMax = MAX(localMax, distances[i]);
    }

    sMaxDistances[tid] = localMax;
    __syncthreads();

    for (unsigned int s = offset / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sMaxDistances[tid] = MAX(sMaxDistances[tid], sMaxDistances[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *maxDistance = sMaxDistances[0];
    }
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	uint32_t length = N * M;
	real_t* hData = createData(length);
    uint32_t sizeInBytes = length * sizeof(real_t);

	real_t* hMinValues = new real_t[M];
	real_t* hMaxValues = new real_t[M];

    findMinMaxByDimension(hData, hMinValues, hMaxValues, N, M);

    // Allocate device memory
	real_t* dData = nullptr;
    size_t dDataPitch = 0;
	real_t* dMinValues = nullptr;
	real_t* dMaxValues = nullptr;
	discrete_t* dDiscreteData = nullptr;
	size_t dDiscreteDataPitch = 0;

    checkCudaErrors(cudaMallocPitch((void**)&dData, &dDataPitch, M * sizeof(real_t), N));
	checkCudaErrors(cudaMalloc((void**)&dMinValues, M * sizeof(real_t)));
	checkCudaErrors(cudaMalloc((void**)&dMaxValues, M * sizeof(real_t)));
	checkCudaErrors(cudaMallocPitch((void**)&dDiscreteData, &dDiscreteDataPitch, M * sizeof(discrete_t), N));

    checkCudaErrors(cudaMemcpy2D(dData, dDataPitch, hData, M * sizeof(real_t), M * sizeof(real_t), N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dMinValues, hMinValues, M * sizeof(real_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dMaxValues, hMaxValues, M * sizeof(real_t), cudaMemcpyHostToDevice));

	delete[] hData;
    delete[] hMinValues;
    delete[] hMaxValues;

	dim3 block{ M, 1, 1 };
    dim3 grid{ 1, 1, 1 };
	std::cout << "Discretize kernel launch parameters: " << block.x << " " << block.y << std::endl;
    auto discretizeKernelFn = [&] {
        discretizeKernel<<<grid, block>>>(dData, dDataPitch, dDiscreteData, dDiscreteDataPitch, dMinValues, dMaxValues, M, N, MAX_DISCRETE_VALUE);
    };
    gpubenchmark::print_time("discretizeKernel", discretizeKernelFn, 10);
    cudaFree(dData);
    cudaFree(dMinValues);
    cudaFree(dMaxValues);

	unsigned long long* dDistances = nullptr;
	checkCudaErrors(cudaMalloc((void**)&dDistances, N * sizeof(unsigned long long)));
	block = { 256, 1, 1 };
	grid = { 1, 1, 1 };
	std::cout << "Distance to origin kernel launch parameters: " << block.x << " " << block.y << std::endl;
	auto distanceToOriginKernelFn = [&] {
		distanceToOriginKernel<<<grid, block>>>(dDiscreteData, dDiscreteDataPitch, dDistances, M, N);
	};
    gpubenchmark::print_time("distanceToOriginKernel", distanceToOriginKernelFn, 100);

	unsigned long long* dMaxDistance = nullptr;
	checkCudaErrors(cudaMalloc((void**)&dMaxDistance, sizeof(unsigned long long)));
	block = { 256, 1, 1 };
	grid = { 1, 1, 1 };
	std::cout << "Find max distance kernel launch parameters: " << block.x << " " << block.y << std::endl;
    auto findMaxDistanceKernelFn = [&] {
        findMaxDistanceKernel<<<grid, block>>>(dDistances, dMaxDistance, N);
    };
	gpubenchmark::print_time("findMaxDistanceKernel", findMaxDistanceKernelFn, 100);

	unsigned long long hMaxDistance = 0;
	checkCudaErrors(cudaMemcpy(&hMaxDistance, dMaxDistance, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	std::cout << "Max distance: " << hMaxDistance << std::endl;

	cudaFree(dDiscreteData);
	cudaFree(dDistances);
	cudaFree(dMaxDistance);

	return 0;
}
