```
template<typename T = float>
__global__ void example_kernel(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.x * 8 + threadIdx.x;
    // blockIdx.x range = [0, 249]
    // threadIdx.x range = [0, 7]
    // int x range = [0, 1999]

    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.y * 10 + threadIdx.y;
    // blockIdx.y range = [0, 124]
    // threadIdx.y range = [0, 9]
    // int y range = [0, 12409]

    // auto scale_factor = width * height / 1600;
    float scale_factor = 2000 * 1000 / 1600;
    // float scale_factor = 1250.0; // Calculated value

    // int applyScaleFactor = (scale_factor > 1000) ? 1 : 0;
    // WARP DIVERGENCE POINT -- VARIABLES REASONING
    // Condition is checking if scale_factor is greater than 1000
    // this condition is always true, this region will always be executed
    int applyScaleFactor = (1250.0 > 1000) ? 1 : 0;
    // int applyScaleFactor = 1; // Calculated value

    // if (x > 0 && x < width-1 && y > 0 && y < height-1) {
    // WARP DIVERGENCE POINT -- VARIABLES REASONING
    // IF statement is checking if x and y are within valid bounds
    // x is in [0, 1999]
    // y is in [0, 12409]
    // entry condition: x > 0 && x < 2000-1 --> x must be in [1, 1998] --> 1998 valid x values
    // entry condition: y > 0 && y < 1000-1 --> y must be in [1, 998] --> 998 valid y values
    // BOTH entry conditions must be met to enter this region 
    if (x > 0 && x < 2000-1 && y > 0 && y < 1000-1) {
        // int idx = y * width + x;
        int idx = y * 2000 + x;

        // for (int i = x; i < y; i += blockDim.x){
        // WARP DIVERGENCE POINT -- VARIABLES REASONING 
        // For-loop is iterating from x to y with a step of 8
        // x is in [1, 1998]
        // y is in [1, 998]
        // for-loop entry condition: x+8*n < y, where n is an integer >= 0
        for (int i = x; i < y; i += 8){
            // out[idx] = (in[idx] + in[idx-1] + in[idx+1] + in[idx-width] + in[idx+width]);
            out[idx] = (in[idx] + in[idx-1] + in[idx+1] + in[idx-2000] + in[idx+2000]);
        }

        // if (applyScaleFactor){
        // WARP DIVERGENCE POINT -- VARIABLES REASONING 
        // Condition is always true, this region will always be executed
        if (1){
            //out[idx] *= 1/static_cast<T>(scale_factor);
            out[idx] *= 1/static_cast<float>(1250.0);
        }
    }
}
```