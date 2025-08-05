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

    // int scaleFactor = (x * y > 1000) ? 5 : 1;
    // CONVERTED TERNARY TO IF STATEMENT
    int scaleFactor = 1;
    if (x * y > 1000){
        scaleFactor = 5;
    }

    // Only compute for internal points (avoid boundary)
    // if (x > 0 && x < width-1 && y > 0 && y < height-1) {
    if (x > 0 && x < 3000-1 && y > 0 && y < 1500-1) {
    // if (x > 0 && x < 2999 && y > 0 && y < 1499) { // Calculated values
        // int idx = y * width + x;
        int idx = y * 3000 + x;
        // out[idx] = static_cast<T>(scaleFactor) * (in[idx] + in[idx-1] + in[idx+1] + in[idx-width] + in[idx+width]) / static_cast<T>(5);
        out[idx] = static_cast<float>(scaleFactor) * (in[idx] + in[idx-1] + in[idx+1] + in[idx-3000] + in[idx+3000]) / static_cast<float>(5);
    }
}

template<typename T = double>
void run_stencil2d(int num_iterations, int width, int height) {
    // static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type");
    static_assert(std::is_arithmetic<float>::value, "T must be arithmetic type");

    // size_t size = static_cast<size_t>(width) * height * sizeof(T);
    size_t size = static_cast<size_t>(3000) * 1500 * sizeof(float);

    // Allocate host memory
    // T* h_in = new T[static_cast<size_t>(width)*height];
    float* h_in = new float[static_cast<size_t>(3000)*1500];
    // float* h_in = new float[static_cast<size_t>(4500000)]; // Calculated values

    // T* h_out = new T[static_cast<size_t>(width)*height];
    float* h_out = new float[static_cast<size_t>(3000)*1500];
    // float* h_out = new float[static_cast<size_t>(4500000)]; // Calculated values

    // Initialize input
    // for (int i = 0; i < width * height; ++i)
    for (int i = 0; i < 3000 * 1500; ++i)
    // for (int i = 0; i < 4500000; ++i) // Calculated values
        // h_in[i] = static_cast<T>((i % 100) * 0.1);
        h_in[i] = static_cast<float>((i % 100) * 0.1);

    // Allocate device memory
    // T *d_in, *d_out;
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Loop for num_iterations
    // for (int iter = 0; iter < num_iterations; ++iter) {
    for (int iter = 0; iter < 10; ++iter) {
        // Vary block size between BLOCK_SIZE_MIN and BLOCK_SIZE_MAX
        // int block_size = BLOCK_SIZE_MIN + (iter % (BLOCK_SIZE_MAX - BLOCK_SIZE_MIN + 1));
        int block_size = 8 + (iter % (32 - 8 + 1));
        // int block_size = 8 + (iter % (25)); // Calculated values
        // if (block_size > BLOCK_SIZE_MAX) block_size = BLOCK_SIZE_MAX;
        if (block_size > 32) block_size = 32;
        // if (block_size < BLOCK_SIZE_MIN) block_size = BLOCK_SIZE_MIN;
        if (block_size < 8) block_size = 8;
        dim3 block(block_size, block_size);
        // dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        dim3 grid((3000 + block.x - 1) / block.x, (1500 + block.y - 1) / block.y);

        // example_kernel<T><<<grid, block>>>(d_in, d_out, width, height);
        example_kernel<T><<<grid, block>>>(d_in, d_out, 3000, 1500);
        cudaDeviceSynchronize(); // Ensure kernel launch is finished before swapping

        // Swap input and output pointers for the next iteration
        std::swap(d_in, d_out);
    }

    // Copy result back (from the last output buffer, which is d_in if even iterations, d_out if odd)
    // T* d_result = (num_iterations % 2 == 0) ? d_in : d_out;
    float* d_result = (10 % 2 == 0) ? d_in : d_out;
    // float* d_result = (true) ? d_in : d_out; // Calculated value
    // CONVERTED TERNARY TO IF STATEMENT
    if (1) {
        d_result = d_in;
    } else {
        d_result = d_out;
    }
    cudaMemcpy(h_out, d_result, size, cudaMemcpyDeviceToHost);

    // Print a small sample of results for verification
    // int yc = height / 2, xc = width / 2;
    int yc = 1500 / 2, xc = 3000 / 2;
    // int yc = 750, xc = 1500; //Calculated values

    // int y0 = max(0, yc - 2), y1 = min(height - 1, yc + 2);
    int y0 = max(0, 750 - 2), y1 = min(1500 - 1, 750 + 2);
    // int y0 = max(0, 748), y1 = min(1499, 752); // Calculated values
    // int y0 = 748, y1 = 752; // Calculated values

    //int x0 = std::max(0, xc - 2), x1 = std::min(width - 1, xc + 2);
    int x0 = std::max(0, 1500 - 2), x1 = std::min(3000 - 1, 1500 + 2);
    // int x0 = std::max(0, 1498), x1 = std::min(2999, 1502); // Calculated values
    // int x0 = 1498, x1 = 1502; // Calculated values

    std::cout << "Sample output (center 5x5 region):\n";
    // for (int y = y0; y <= y1; ++y) {
    for (int y = 748; y <= 752; ++y) {
        // for (int x = x0; x <= x1; ++x) {
        for (int x = 1498; x <= 1502; ++x) {
            // std::cout << h_out[y*width + x] << " ";
            std::cout << h_out[y*3000 + x] << " ";
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
        // num_iterations = std::atoi(argv[1]);
        num_iterations = 10; // From Execution Args
        if (num_iterations < 1) num_iterations = 1;
    }
    if (argc > 2) {
        // width = std::atoi(argv[2]);
        width = 3000; // From Execution Args
        if (width < 3) width = 3;
    }
    if (argc > 3) {
        // height = std::atoi(argv[3]);
        height = 1500; // From Execution Args
        if (height < 3) height = 3;
    }
    // run_stencil2d<float>(num_iterations, width, height);
    run_stencil2d<float>(10, 3000, 1500); 
    // To run for double, use: run_stencil2d<double>(num_iterations, width, height);
    return 0;
}
```