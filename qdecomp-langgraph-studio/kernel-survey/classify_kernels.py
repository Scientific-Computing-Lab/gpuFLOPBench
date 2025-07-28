import argparse
import json
import os
import sys
from utils.dataset import kernels_data, target_names

# Add the current directory to the path to allow for absolute imports
sys.path.append(os.path.dirname(__file__))

from static_passes.float_div_check import check_has_float_division
from static_passes.external_lib_call_check import check_external_lib_calls
from static_passes.recursion_check import check_has_recursion
from static_passes.warp_divergence_check import check_has_warp_divergence
from static_passes.dd_warp_divergence_check import check_has_dd_warp_divergence
from static_passes.common_subexpr_check import check_has_common_subexpr
from static_passes.math_fnct_check import check_has_math_fnct_calls
from static_passes.TargetKernel import TargetKernel, TargetKernelEncoder


def print_code_with_line_numbers(source_code):
    for i, line in enumerate(source_code.split('\n')):
        print(f"{i+1:4d}: {line}")


def classify_kernel(kernel_source):
    kernel = TargetKernel(kernel_source)

    check_has_float_division(kernel)
    check_external_lib_calls(kernel)
    check_has_recursion(kernel)
    check_has_warp_divergence(kernel)
    check_has_dd_warp_divergence(kernel)
    check_has_common_subexpr(kernel)
    check_has_math_fnct_calls(kernel)

    if kernel.has_special_math_function:
        print("Special math function found in kernel source code!")
        print_code_with_line_numbers(kernel.source_code)
        print("Line numbers:", kernel.special_math_function_line_num)

    # Analyze the kernel source code and update the kernel object
    return kernel


def main():
    results = {}

    # Visit each kernel in the JSON data
    for benchmark_name, files in kernels_data.items():
        results[benchmark_name] = {}
        for file_path, kernels in files.items():
            results[benchmark_name][file_path] = []
            for i, kernel_source in enumerate(kernels):
                print(f"Classifying kernel {i+1} from {file_path} in {benchmark_name}...")
                result = classify_kernel(kernel_source)
                results[benchmark_name][file_path].append(result)

    # Save the results
    with open("classification_results.json", "w") as f:
        json.dump(results, f, indent=4, cls=TargetKernelEncoder)

    print("Classification complete. Results saved to classification_results.json")

if __name__ == "__main__":
    main()
