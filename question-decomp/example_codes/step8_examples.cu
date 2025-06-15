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

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - static_cast<float>(a): Converts an int to float.
    //   - SP-FLOP: 1 (conversion to float)
    //   - DP-FLOP: 0 (no double-precision)
    float_results[0] = static_cast<float>(a);

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - static_cast<float>(b): Converts a double to float.
    //   - SP-FLOP: 1 (conversion to float)
    //   - DP-FLOP: 0 (the double value is just stored, conversion is to float)
    float_results[1] = static_cast<float>(b);

    // 2 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - c + fc: c (int) promoted to float (SP-FLOP: 1)
    //   - Addition in float (SP-FLOP: 1)
    //   - DP-FLOP: 0 (no double-precision)
    float_results[2] = c + fc;

    // 1 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - d + e: e (int) promoted to double (DP-FLOP: 1)
    //   - Addition in double (DP-FLOP: 1)
    //   - Result converted from double to float (SP-FLOP: 1)
    float_results[3] = static_cast<float>(d + e);

    // 1 SP-FLOP, 4 DP-FLOP
    // Explanation:
    //   - f + g: f (int) promoted to double (DP-FLOP: 1)
    //   - h (float) promoted to double for addition (DP-FLOP: 1)
    //   - Two additions in double (DP-FLOP: 2)
    //   - Final result converted from double to float (SP-FLOP: 1)
    float_results[4] = static_cast<float>(f + g + h);

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - static_cast<float>(ch): char (via int) is converted to float.
    //   - SP-FLOP: 1 (conversion to float)
    //   - DP-FLOP: 0 (no double-precision)
    float_results[5] = static_cast<float>(ch);

    // 1 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - static_cast<float>(ull): unsigned long long is converted to float.
    //   - SP-FLOP: 1 (conversion to float)
    //   - DP-FLOP: 0 (no double-precision)
    float_results[6] = static_cast<float>(ull);

    // 2 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - i + s: s (short) promoted to int (not a FP op)
    //   - (i + s) promoted to double for addition with db (DP-FLOP: 1)
    //   - Addition in double (DP-FLOP: 1)
    //   - Result converted to float (SP-FLOP: 1)
    //   - Addition with fl (float) (SP-FLOP: 1)
    float_results[7] = static_cast<float>((i + s) + db) + fl;

    // 2 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - c * fc: c (int) promoted to float (SP-FLOP: 1)
    //   - Multiplication in float (SP-FLOP: 1)
    float_results[8] = c * fc;

    // 2 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - fc - c: c (int) promoted to float (SP-FLOP: 1)
    //   - Subtraction in float (SP-FLOP: 1)
    float_results[9] = fc - c;

    // 2 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - fc / c: c (int) promoted to float (SP-FLOP: 1)
    //   - Division in float (SP-FLOP: 1)
    float_results[10] = fc / c;

    // 1 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - d / e: e (int) promoted to double (DP-FLOP: 1)
    //   - Division in double (DP-FLOP: 1)
    //   - Result converted from double to float (SP-FLOP: 1)
    float_results[11] = static_cast<float>(d / e);

    // 1 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - d - e: e (int) promoted to double (DP-FLOP: 1)
    //   - Subtraction in double (DP-FLOP: 1)
    //   - Result converted from double to float (SP-FLOP: 1)
    float_results[12] = static_cast<float>(d - e);

    // 1 SP-FLOP, 2 DP-FLOP
    // Explanation:
    //   - d * e: e (int) promoted to double (DP-FLOP: 1)
    //   - Multiplication in double (DP-FLOP: 1)
    //   - Result converted from double to float (SP-FLOP: 1)
    float_results[13] = static_cast<float>(d * e);

    // 3 SP-FLOP, 0 DP-FLOP
    // Explanation:
    //   - fmaf(fc, float(c), h): c (int) promoted to float (SP-FLOP: 1, for promotion)
    //   - Fused multiply-add in float (fmaf, SP-FLOP: 2; counted as two operations: one multiply, one add)
    //   - DP-FLOP: 0 (no double-precision)
    float_results[14] = fmaf(fc, static_cast<float>(c), h);

    // 1 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - static_cast<double>(a): Converts int to double.
    //   - DP-FLOP: 1 (conversion to double)
    //   - SP-FLOP: 0 (no single-precision)
    double_results[0] = static_cast<double>(a);

    // 1 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - static_cast<double>(fc): Converts float to double.
    //   - DP-FLOP: 1 (conversion to double)
    //   - SP-FLOP: 0 (no single-precision)
    double_results[1] = static_cast<double>(fc);

    // 2 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - c + d: c (int) promoted to double (DP-FLOP: 1)
    //   - Addition in double (DP-FLOP: 1)
    double_results[2] = c + d;

    // 3 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - e + b: e (int) promoted to double (DP-FLOP: 1)
    //   - fc (float) promoted to double for addition (DP-FLOP: 1)
    //   - Two additions in double (DP-FLOP: 2)
    double_results[3] = e + b + static_cast<double>(fc);

    // 1 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - static_cast<double>(ch): char (via int) to double.
    double_results[4] = static_cast<double>(ch);

    // 1 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - static_cast<double>(ull): unsigned long long to double.
    double_results[5] = static_cast<double>(ull);

    // 2 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - (i + s): s (short) promoted to int (not a FP op)
    //   - (i + s) promoted to double for addition with db (DP-FLOP: 1)
    //   - Addition in double (DP-FLOP: 1)
    double_results[6] = (i + s) + db;

    // VARIATION EXAMPLES (double)

    // 2 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - c * d: c (int) promoted to double (DP-FLOP: 1)
    //   - Multiplication in double (DP-FLOP: 1)
    double_results[7] = c * d;

    // 2 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - d - c: c (int) promoted to double (DP-FLOP: 1)
    //   - Subtraction in double (DP-FLOP: 1)
    double_results[8] = d - c;

    // 2 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - d / c: c (int) promoted to double (DP-FLOP: 1)
    //   - Division in double (DP-FLOP: 1)
    double_results[9] = d / c;

    // 3 DP-FLOP, 0 SP-FLOP
    // Explanation:
    //   - fma(d, double(e), g): e (int) promoted to double (DP-FLOP: 1, for promotion)
    //   - Fused multiply-add in double (fma, DP-FLOP: 2; counted as two operations: one multiply, one add)
    //   - SP-FLOP: 0 (no single-precision)
    double_results[10] = fma(d, static_cast<double>(e), g);
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

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "int to float: " << h_float_results[0] << std::endl;
    std::cout << "double to float: " << h_float_results[1] << std::endl;
    std::cout << "int + float: " << h_float_results[2] << std::endl;
    std::cout << "double + int, converted to float: " << h_float_results[3] << std::endl;
    std::cout << "int + double + float, result as float: " << h_float_results[4] << std::endl;
    std::cout << "char to float: " << h_float_results[5] << std::endl;
    std::cout << "unsigned long long to float: " << h_float_results[6] << std::endl;
    std::cout << "short + int + double, result to float, then add float: " << h_float_results[7] << std::endl;
    std::cout << "int * float: " << h_float_results[8] << std::endl;
    std::cout << "float - int: " << h_float_results[9] << std::endl;
    std::cout << "float / int: " << h_float_results[10] << std::endl;
    std::cout << "double / int, converted to float: " << h_float_results[11] << std::endl;
    std::cout << "double - int, converted to float: " << h_float_results[12] << std::endl;
    std::cout << "double * int, converted to float: " << h_float_results[13] << std::endl;
    std::cout << "fmaf(float, float, float) [counts as 2 SP-FLOP + 1 SP-FLOP for promotion = 3]: " << h_float_results[14] << std::endl;

    std::cout << "int to double: " << h_double_results[0] << std::endl;
    std::cout << "float to double: " << h_double_results[1] << std::endl;
    std::cout << "int + double: " << h_double_results[2] << std::endl;
    std::cout << "int + double + float(as double): " << h_double_results[3] << std::endl;
    std::cout << "char to double: " << h_double_results[4] << std::endl;
    std::cout << "unsigned long long to double: " << h_double_results[5] << std::endl;
    std::cout << "short + int + double: " << h_double_results[6] << std::endl;
    std::cout << "int * double: " << h_double_results[7] << std::endl;
    std::cout << "double - int: " << h_double_results[8] << std::endl;
    std::cout << "double / int: " << h_double_results[9] << std::endl;
    std::cout << "fma(double, double, double) [counts as 2 DP-FLOP + 1 DP-FLOP for promotion = 3]: " << h_double_results[10] << std::endl;

    return 0;
}
```