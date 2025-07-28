import os
import json
from collections import defaultdict

import re
from tree_sitter import Language, Parser
import tree_sitter_cuda

CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
parser = Parser(CUDA_LANGUAGE)

def get_all_files():
    """
    Surveys the '../../src/' directory to find C/C++ and CUDA files
    in subdirectories ending with '-omp' or '-cuda'.

    Returns:
        dict: A dictionary where keys are the subdirectory names and
              values are lists of file paths within those subdirectories.
    """
    # ../../src relative to kernel-survey/
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    
    if not os.path.isdir(src_dir):
        print(f"Error: Directory '{src_dir}' not found.")
        return {}

    all_files_dict = defaultdict(list)
    
    valid_extensions = ['.c', '.cpp', '.h', '.cu', '.cuh', '.cc']

    for entry in os.scandir(src_dir):
        if entry.is_dir() and (entry.name.endswith('-omp') or entry.name.endswith('-cuda')):
            subdir_path = entry.path
            
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if any(file.endswith(ext) for ext in valid_extensions):
                        all_files_dict[entry.name].append(os.path.join(root, file))

    return dict(all_files_dict)

def extract_all_CUDA_function_defs(all_files_dict):
    """
    Parses files to extract CUDA function definitions using tree-sitter.

    Args:
        all_files_dict (dict): A dictionary from get_all_files.

    Returns:
        dict: A nested dictionary with the same keys as input, but values
              are dicts mapping filenames to lists of function definition strings.
    """
    extracted_defs = defaultdict(lambda: defaultdict(list))

    def find_function_definitions(node, function_list):
        if node.type == 'function_definition':
            # Check for __global__ or __device__ keywords
            declaration = node.text.decode('utf8').split('{')[0]
            if '__global__' in declaration or '__device__' in declaration:
                function_list.append(node)
            return
        for child in node.children:
            find_function_definitions(child, function_list)

    for dir_name, files in all_files_dict.items():
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()
                
                tree = parser.parse(bytes(source_code, "utf8"))
                root_node = tree.root_node
                
                function_nodes = []
                find_function_definitions(root_node, function_nodes)

                if function_nodes:
                    for node in function_nodes:
                        extracted_defs[dir_name][file_path].append(node.text.decode('utf8'))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return json.loads(json.dumps(extracted_defs))


if __name__ == '__main__':
    files_dict = get_all_files()
    if files_dict:
        cuda_defs = extract_all_CUDA_function_defs(files_dict)
        
        output_filename = "extracted_CUDA_kernels.json"
        with open(output_filename, 'w') as f:
            json.dump(cuda_defs, f, indent=4)
        
        print(f"Extracted CUDA kernels saved to [{output_filename}]")
