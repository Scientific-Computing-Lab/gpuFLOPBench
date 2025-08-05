Example Target Kernel Name: example_kernel 

Example Kernel Invocation Arguments and Descriptions:
// Template arguments for the example_kernel invocation:
// T (default template argument, instantiated as 'float' for this call):
//   Specifies the data type of the elements in the input and output arrays.
//   For this call, it's 'float'.

// Parameter descriptions for the example_kernel<T> invocation:
// d_in: float*, pointer to the input 2D array data on the GPU.
//       The data is a contiguous block of (width * height) = (2000 * 1000) elements of type float.
//       Layout: Flattened 1D array in row-major order, logically representing a 2000 x 1000 2D grid.
//       Size on GPU: 2,000,000 * sizeof(float) bytes.
// d_out: float*, pointer to the output 2D array data on the GPU.
//        The data is a contiguous block of (width * height) = (2000 * 1000) elements of type float.
//        Layout: Flattened 1D array in row-major order, logically representing a 2000 x 1000 2D grid.
//        Size on GPU: 2,000,000 * sizeof(float) bytes.
// width: int, the width (number of columns) of the logical 2D grid being processed.
//        Value for this call: 2000.
// height: int, the height (number of rows) of the logical 2D grid being processed.
//         Value for this call: 1000.

// Description for kernel launch configuration:
// Grid Size: grid=( (2000 + 8 - 1) / 8 = 250, (1000 + 8 - 1) / 8 = 125, 1 )
// Block Size: block=(BLOCK_SIZE_MIN=8, BLOCK_SIZE_MAX=10) = (8, 10, 1)
// Total Number of Threads: 250 * 125 * 1 * 8 * 10 * 1 = 2500000 threads

example_kernel<T=float><<<grid=(250, 125, 1), block=(8, 10, 1)>>>(d_in, d_out, width=2000, height=1000);

Example Grid Size: (250, 125, 1)
Example Block Size: (8, 10, 1)
Example Total Number of Threads: 2500000 

Example Source Code:
```
template<typename T = float>
__global__ void example_kernel(const T* __restrict__ in, T* __restrict__ out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto scale_factor = width * height / 1600;

    int applyScaleFactor = (scale_factor > 1000) ? 1 : 0;

    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        int idx = y * width + x;

        out[idx] = (in[idx] + in[idx-1] + in[idx+1] + in[idx-width] + in[idx+width]);

        if (applyScaleFactor) {
            out[idx] *= 1/static_cast<T>(scale_factor); // Example operation based on scale factor
        }
    }
}
```