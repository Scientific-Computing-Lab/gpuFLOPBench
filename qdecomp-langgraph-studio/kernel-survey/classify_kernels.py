import argparse
import json
import os
import tiktoken
from agent import create_llm_chain
from utils.dataset import kernels_data, target_names

def user_confirm_before_continue(total_kernels, total_tokens):
    """Asks the user for confirmation to proceed with classification."""
    print(f"You are about to classify {total_kernels} kernels.")
    print(f"This will result in approximately {total_tokens} tokens being processed.")
    confirmation = input("Press ENTER to continue or SPACE then ENTER to abort: ")
    if confirmation == ' ':
        print("Aborting.")
        return False
    return True

def calculate_tokens(kernels_data, model_name, system_prompt):
    """Calculates the total number of kernels and tokens."""
    total_kernels = 0
    total_tokens = 0
    for _, files in kernels_data.items():
        for _, kernels in files.items():
            total_kernels += len(kernels)

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    system_prompt_tokens = len(encoding.encode(system_prompt))
    for _, files in kernels_data.items():
        for _, kernels in files.items():
            for kernel_source in kernels:
                total_tokens += system_prompt_tokens + len(encoding.encode(kernel_source))
    
    return total_kernels, total_tokens

def main():
    parser = argparse.ArgumentParser(description="Classify CUDA kernels using an LLM.")
    parser.add_argument("--model_name", type=str, default="openai/o3-mini", help="The name of the model to use.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.1, help="The top_p for sampling.")
    args = parser.parse_args()

    # Load the system prompt
    try:
        with open("kernel_survey_system_prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("Error: kernel_survey_system_prompt.txt not found.")
        return

    # Calculate the number of kernels and tokens
    total_kernels, total_tokens = calculate_tokens(kernels_data, args.model_name, system_prompt)

    # Ask for confirmation
    if not user_confirm_before_continue(total_kernels, total_tokens):
        return

    # Initialize the LLM chain
    chain = create_llm_chain(args.model_name, args.temperature, args.top_p, system_prompt)

    results = {}

    # Visit each kernel in the JSON data
    for benchmark_name, files in kernels_data.items():
        results[benchmark_name] = {}
        for file_path, kernels in files.items():
            results[benchmark_name][file_path] = []
            for i, kernel_source in enumerate(kernels):
                print(f"Classifying kernel {i+1} from {file_path} in {benchmark_name}...")
                result = chain.invoke({
                    "system_prompt": system_prompt,
                    "kernel_source_code_string": kernel_source
                })
                results[benchmark_name][file_path].append(result.content)
                print(result.content)

    # Save the results
    with open("classification_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Classification complete. Results saved to classification_results.json")

if __name__ == "__main__":
    main()
