```
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <math.h>

__global__ void float_double_examples_kernel(
    int a, double b, int c, float fc, double d, int e, int f, double g, float h, char ch, unsigned long long ull,
    short s, int i, double db, float fl,
    float* float_results, double* double_results
) {

    // Do not annotate this line, it does not perform any floating-point operations
    // as all of its operands are integers
    int xx = a / c + e;

    // Do not annotate this line, it does not perform any floating-point operations
    // as all of its operands turn out to be integers
    auto xx2 = c * a / f + e;

    // --- float examples ---

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //  - negation operator (-) causes 1 FADD operation
    //  - SP-FLOP: 1
    //  - DP-FLOP: 0 (no double-precision)
    float_results[0] = -fc;

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - static_cast<float>(a): Converts an int to float, this does not involve any FADD, FMUL, or FFMA operation
    //   - "+=" is 1 FADD operation
    //   - SP-FLOP: 1 
    //   - DP-FLOP: 0 (no double-precision)
    float_results[0] += static_cast<float>(a);

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - static_cast<float>(): Converts an int to float, this does not involve any FADD, FMUL, or FFMA operation
    //   - "+ (float) 4": Promotes integer "a" to float and adds 4 to it via 1 FADD operation
    //   - SP-FLOP: 1
    //   - DP-FLOP: 0 (no double-precision)
    float_results[1] = static_cast<float>(a + (float)4);

    // 2 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - static_cast<float>(): Converts an int to float, this does not involve any FADD, FMUL, or FFMA operation
    //   - "+ (float) 4": Promotes integer "a" to float and adds 4 to it via 1 FADD operation
    //   - "+=" writes the final result to the output array with another FADD operation
    //   - SP-FLOP: 2
    //   - DP-FLOP: 0 (no double-precision)
    float_results[2] += static_cast<float>(a + (float)4);

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - "a + 4": Performs an integer addition 
    //   - static_cast<float>(): Converts the addition result to a float, this does not involve any FADD, FMUL, or FFMA operation
    //   - "+=" writes the final result to the output array with 1 FADD operation
    //   - SP-FLOP: 1
    //   - DP-FLOP: 0 (no double-precision)
    float_results[3] += static_cast<float>(a + 4);

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - "a + 4": Performs an integer addition 
    //   - static_cast<float>(): Converts the addition result to a float, this does not involve any FADD, FMUL, or FFMA operation
    //   - "*=" writes the final result to the output array with 1 FMUL operation
    //   - SP-FLOP: 1
    //   - DP-FLOP: 0 (no double-precision)
    float_results[4] *= static_cast<float>(a + 4);

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //   - d + e: e (int) promoted to double 
    //   - Addition in double is 1 DADD operation
    //   - Result converted from double to float does not use any FADD, FMUL, or FFMA operations  
    //   - SP-FLOP: 0 (no single-precision)
    //   - DP-FLOP: 1 
    float_results[5] = static_cast<float>(d + e);

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - c + fc: c (int) promoted to float (no FADD, FMUL, or FFMA operation)
    //   - Addition in float is 1 FADD operation
    //   - SP-FLOP: 1 
    //   - DP-FLOP: 0 (no double-precision)
    float_results[6] = c + fc;

    // 0 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - c (int) promoted to double (no DADD, DMUL, or DFMA operations)
    //   - fc is promoted to double (no DADD, DMUL, or DFMA operations)
    //   - Compiler uses 1 DFMA operation for "c * d + fc"
    //   - DFMA counts as 2 operations: 1 DMUL and 1 DADD
    //   - Convert double result of "c * d + fc" to float (no FADD, FMUL, or FFMA operations)
    //   - SP-FLOP: 0 (no single-precision)
    //   - DP-FLOP: 2
    float_results[7] = c * d + fc;

    // 0 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - static_cast<float>(): Converts an int to float, this does not involve any FADD, FMUL, or FFMA operation
    //   - f (int) promoted to double (no DADD, DMUL, or DFMA operations)
    //   - h (float) promoted to double (no DADD, DMUL, or DFMA operations)
    //   - Compiler uses 2 DADD operations for "f + g + h"
    //   - SP-FLOP: 0 (no single-precision)
    //   - DP-FLOP: 2
    float_results[8] = static_cast<float>(f + g + h);

    // 0 SP-FLOP, 3 DP-FLOP
    // Explanation:
    //   - static_cast<float>(): Converts an int to float, this does not involve any FADD, FMUL, or FFMA operation
    //   - f (int) promoted to double (no DADD, DMUL, or DFMA operations)
    //   - h (float) promoted to double (no DADD, DMUL, or DFMA operations)
    //   - Compiler uses 2 DADD operations for "f + g + h"
    //   - "+=" is done by promoting the result to a double, then using 1 DADD operation to update the result, and then converting the result back to float
    //   - SP-FLOP: 0 (no single-precision)
    //   - DP-FLOP: 3
    float_results[9] += static_cast<float>(f + g + h);

    // --- double examples (mirror float above) ---

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //  - negation operator (-) causes 1 DADD operation
    //  - SP-FLOP: 0 (no single-precision)
    //  - DP-FLOP: 1 
    double_results[0] = -g;

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //   - static_cast<double>(a): Converts int to double (no DADD/DMUL/DFMA)
    //   - "+=" is 1 DADD operation
    //   - SP-FLOP: 0
    //   - DP-FLOP: 1
    double_results[0] += static_cast<double>(a);

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //   - static_cast<double>(): Converts int to double (no DADD/DMUL/DFMA)
    //   - "+ (double)4": Promotes 4 to double and adds it via 1 DADD operation
    //   - SP-FLOP: 0
    //   - DP-FLOP: 1
    double_results[1] = static_cast<double>(a + (double)4);

    // 0 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - static_cast<double>(): Converts int to double (no DADD/DMUL/DFMA)
    //   - "+ (double)4": 1 DADD for inner addition
    //   - "+=" writes final result with another DADD
    //   - SP-FLOP: 0
    //   - DP-FLOP: 2
    double_results[2] += static_cast<double>(a + (double)4);

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //   - "a + 4": Integer addition (no FP ops)
    //   - static_cast<double>(): Converts to double (no DADD/DMUL/DFMA)
    //   - "+=" uses 1 DADD
    //   - SP-FLOP: 0
    //   - DP-FLOP: 1
    double_results[3] += static_cast<double>(a + 4);

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //   - "a + 4": Integer addition (no FP ops)
    //   - static_cast<double>(): Converts to double (no DADD/DMUL/DFMA)
    //   - "*=" uses 1 DMUL
    //   - SP-FLOP: 0
    //   - DP-FLOP: 1
    double_results[4] *= static_cast<double>(a + 4);

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //   - d + e: e promoted to double
    //   - 1 DADD operation
    //   - SP-FLOP: 0
    //   - DP-FLOP: 1
    double_results[5] = d + e;

    // 0 SP-FLOP, 1 DP-FLOP
    // Explanation:
    //   - c + d: c promoted to double (no FP convert op count)
    //   - 1 DADD operation
    //   - SP-FLOP: 0
    //   - DP-FLOP: 1
    double_results[6] = c + d;

    // 0 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - c promoted to double, fc promoted to double (no counted ops)
    //   - Compiler uses 1 DFMA for "c * d + fc"
    //   - DFMA counts as DMUL + DADD = 2 DP-FLOPs
    //   - SP-FLOP: 0
    //   - DP-FLOP: 2
    double_results[7] = c * d + fc;

    // 0 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - f promoted to double, h promoted to double (no counted ops)
    //   - Compiler uses 2 DADD operations for "f + g + h"
    //   - SP-FLOP: 0
    //   - DP-FLOP: 2
    double_results[8] = f + g + h;

    // 0 SP-FLOP, 3 DP-FLOP
    // Explanation:
    //   - f promoted to double, h promoted to double (no counted ops)
    //   - 2 DADDs for "f + g + h"
    //   - "+=" uses 1 more DADD to update the stored value
    //   - SP-FLOP: 0
    //   - DP-FLOP: 3
    double_results[9] += f + g + h;
}

int main() {
    // Host variables
    int a = 5;
    double b = 3.14159;
    int c = 2;
    float fc = 4.5f;
    double d = 7.7;
    int e = 3;
    int f = 1;
    double g = 2.2;
    float h = 3.3f;
    char ch = 65; // ASCII for 'A'
    unsigned long long ull = 1234567890ULL;
    short s = 10;
    int i = 20;
    double db = 30.5;
    float fl = 40.5f;

    constexpr int num_float_results = 15;
    constexpr int num_double_results = 11;

    float h_float_results[num_float_results];
    double h_double_results[num_double_results];

    float* d_float_results;
    double* d_double_results;

    cudaMalloc(&d_float_results, sizeof(float) * num_float_results);
    cudaMalloc(&d_double_results, sizeof(double) * num_double_results);

    float_double_examples_kernel<<<1, 1>>>(
        a, b, c, fc, d, e, f, g, h, ch, ull, s, i, db, fl,
        d_float_results, d_double_results
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_float_results, d_float_results, sizeof(float) * num_float_results, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_double_results, d_double_results, sizeof(double) * num_double_results, cudaMemcpyDeviceToHost);

    cudaFree(d_float_results);
    cudaFree(d_double_results);

    return 0;
}
```