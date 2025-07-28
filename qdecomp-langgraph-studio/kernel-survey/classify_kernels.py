import argparse
import json
import os
import sys
from utils.dataset import kernels_data, target_names

# Add the current directory to the path to allow for absolute imports
sys.path.append(os.path.dirname(__file__))

from static_passes.float_div_check import check_has_float_division
from static_passes.TargetKernel import TargetKernel, TargetKernelEncoder


def classify_kernel(kernel_source):
    kernel = TargetKernel(kernel_source)

    check_has_float_division(kernel)

    if kernel.has_float_division:
        print("Floating-point division found in kernel source code!")
        print(kernel.source_code)

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
