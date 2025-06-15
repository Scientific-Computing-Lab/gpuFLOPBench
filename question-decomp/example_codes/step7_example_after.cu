```
template<typename T = float>
__global__ void example_kernel(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.x * 8 + threadIdx.x;
    // blockIdx.x range = (0, 249)
    // threadIdx.x range = (0, 7)
    // int x range = (0, 1999)

    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.y * 10 + threadIdx.y;
    // blockIdx.y range = (0, 124)
    // threadIdx.y range = (0, 9)
    // int y range = (0, 12409)

    // auto scale_factor = width * height / 1600;
    float scale_factor = 2000 * 1000 / 1600;
    // float scale_factor = 1250.0; // Calculated value

    // int applyScaleFactor = (scale_factor > 1000) ? 1 : 0;
    int applyScaleFactor = (1250.0 > 1000) ? 1 : 0;
    // int applyScaleFactor = 1; // Calculated value

    // WARP DIVERGENCE POINT -- TOTAL THREADS ENTERING REASONING
    // x > 0 && x < 2000-1 --> x in (1, 1998) --> 1998 valid x values
    // y > 0 && y < 1000-1 --> y in (1, 998) --> 998 valid y values
    // x and y must both be valid for the computation to proceed
    // Only threads with valid x and y will enter this region
    // Total valid threads entering this region = 998 * 1998 = 1996004
    // WARP DIVERGENCE POINT -- TOTAL NUM THREADS ENTERING REGION: 1996004
    // if (x > 0 && x < width-1 && y > 0 && y < height-1) {
    if (x > 0 && x < 2000-1 && y > 0 && y < 1000-1) {
        // int idx = y * width + x;
        int idx = y * 2000 + x;

        // out[idx] = (in[idx] + in[idx-1] + in[idx+1] + in[idx-width] + in[idx+width]);
        out[idx] = (in[idx] + in[idx-1] + in[idx+1] + in[idx-2000] + in[idx+2000]);

        // WARP DIVERGENCE POINT -- TOTAL THREADS ENTERING REASONING
        // Condition is always true, so all threads in this region will execute this
        // WARP DIVERGENCE POINT -- TOTAL NUM THREADS ENTERING REGION: 1996004
        // if (applyScaleFactor){
        if (1){
            //out[idx] *= 1/static_cast<T>(scale_factor);
            out[idx] *= 1/static_cast<float>(1250.0);
        }
    }
}
```