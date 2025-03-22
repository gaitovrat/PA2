#include <cudaDefs.h>
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

__global__ void discretizeKernel(real_t* __restrict__ input, discrete_t* __restrict__ output, real_t* __restrict__ minValues, real_t* __restrict__ maxValues, const size_t N, const size_t M) 
{
    size_t idx = threadIdx.x;
    const size_t offset = blockDim.x;

    while (idx < N * M) {
        size_t vector_idx = idx / M;
        size_t dim_idx = idx % M;

        real_t val = input[idx];
        real_t min_val = minValues[dim_idx];
        real_t max_val = maxValues[dim_idx];
        real_t range = max_val - min_val;

        real_t normalized = (val - min_val) / range;
        output[idx] = static_cast<discrete_t>(normalized * MAX_DISCRETE_VALUE);
  
        idx += offset;
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
	real_t* dMinValues = nullptr;
	real_t* dMaxValues = nullptr;
	discrete_t* dDiscreteData = nullptr;

	checkCudaErrors(cudaMalloc((void**)&dData, sizeInBytes));
	checkCudaErrors(cudaMalloc((void**)&dMinValues, M * sizeof(real_t)));
	checkCudaErrors(cudaMalloc((void**)&dMaxValues, M * sizeof(real_t)));
	checkCudaErrors(cudaMalloc((void**)&dDiscreteData, length * sizeof(discrete_t)));

    checkCudaErrors(cudaMemcpy(dData, hData, sizeInBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dMinValues, hMinValues, M * sizeof(real_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dMaxValues, hMaxValues, M * sizeof(real_t), cudaMemcpyHostToDevice));

	dim3 block{ 256, 1, 1 };
    dim3 grid{ 1, 1, 1 };
	std::cout << "Discretize kernel launch parameters: " << grid.x << " " << block.x << std::endl;
    discretizeKernel<<<grid, block>>>(dData, dDiscreteData, dMinValues, dMaxValues, N, M);

	discrete_t* hDiscreteData = new discrete_t[length];
	checkCudaErrors(cudaMemcpy(hDiscreteData, dDiscreteData, length * sizeof(discrete_t), cudaMemcpyDeviceToHost));
    checkHostMatrix(hDiscreteData, length * sizeof(discrete_t), 1, length, "%d ", "C: ");
	
	// Cleanup
	delete[] hData;
	delete[] hMinValues;
	delete[] hMaxValues;
	delete[] hDiscreteData;

	cudaFree(dData);
	cudaFree(dMinValues);
	cudaFree(dMaxValues);
	cudaFree(dDiscreteData);

	return 0;
}
