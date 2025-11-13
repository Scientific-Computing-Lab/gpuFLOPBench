Example Target Kernel Name: void example_kernel<T>(const T*, T*, int, int)
Example Execution Arguments: [10 3000 1500]
Example Grid Size: (250, 125, 1)
Example Block Size: (8, 10, 1) 
Example Total Number of Threads: 2500000
Example Source Code:
```
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
#include <cstdlib>

#define BLOCK_SIZE_MIN 8
#define BLOCK_SIZE_MAX 32

// Templated CUDA kernel for 2D 5-point stencil (default: float)
template<typename T = double>
__global__ void example_kernel(const T* __restrict__ in, T* __restrict__ out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int scaleFactor = (x * y > 1000) ? 5 : 1;

    // Only compute for internal points (avoid boundary)
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        int idx = y * width + x;
        out[idx] = static_cast<T>(scaleFactor) * (in[idx] + in[idx-1] + in[idx+1] + in[idx-width] + in[idx+width]) / static_cast<T>(5);
    }
}

template<typename T = double>
void run_stencil2d(int num_iterations, int width, int height) {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type");

    size_t size = static_cast<size_t>(width) * height * sizeof(T);

    // Allocate host memory
    T* h_in = new T[static_cast<size_t>(width)*height];
    T* h_out = new T[static_cast<size_t>(width)*height];

    // Initialize input
    for (int i = 0; i < width * height; ++i)
        h_in[i] = static_cast<T>((i % 100) * 0.1);

    // Allocate device memory
    T *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Loop for num_iterations
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Vary block size between BLOCK_SIZE_MIN and BLOCK_SIZE_MAX
        int block_size = BLOCK_SIZE_MIN + (iter % (BLOCK_SIZE_MAX - BLOCK_SIZE_MIN + 1));
        if (block_size > BLOCK_SIZE_MAX) block_size = BLOCK_SIZE_MAX;
        if (block_size < BLOCK_SIZE_MIN) block_size = BLOCK_SIZE_MIN;
        dim3 block(block_size, block_size);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        example_kernel<T><<<grid, block>>>(d_in, d_out, width, height);
        cudaDeviceSynchronize(); // Ensure kernel launch is finished before swapping

        // Swap input and output pointers for the next iteration
        std::swap(d_in, d_out);
    }

    // Copy result back (from the last output buffer, which is d_in if even iterations, d_out if odd)
    T* d_result = (num_iterations % 2 == 0) ? d_in : d_out;
    cudaMemcpy(h_out, d_result, size, cudaMemcpyDeviceToHost);

    // Print a small sample of results for verification
    int yc = height / 2, xc = width / 2;
    int y0 = std::max(0, yc - 2), y1 = std::min(height - 1, yc + 2);
    int x0 = std::max(0, xc - 2), x1 = std::min(width - 1, xc + 2);

    std::cout << "Sample output (center 5x5 region):\n";
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            std::cout << h_out[y*width + x] << " ";
        }
        std::cout << "\n";
    }

    // Free memory
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
}

int main(int argc, char* argv[]) {
    int num_iterations = 1;
    int width = 2000;
    int height = 2000;
    if (argc > 1) {
        num_iterations = std::atoi(argv[1]);
        if (num_iterations < 1) num_iterations = 1;
    }
    if (argc > 2) {
        width = std::atoi(argv[2]);
        if (width < 3) width = 3;
    }
    if (argc > 3) {
        height = std::atoi(argv[3]);
        if (height < 3) height = 3;
    }
    run_stencil2d<float>(num_iterations, width, height);
    // To run for double, use: run_stencil2d<double>(num_iterations, width, height);
    return 0;
}
```