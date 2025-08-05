import os
import json
from collections import defaultdict

from utils.ts_helper import *
from tqdm import tqdm

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
    Parses files to extract CUDA __global__ function definitions and any __device__
    or other __global__ functions they call. It searches across all provided files
    to find the definitions of called functions.

    A __global__ function is only extracted as a standalone definition if it is not
    called by any other __global__ or __device__ function within the set of
    surveyed files.

    Args:
        all_files_dict (dict): A dictionary from get_all_files.

    Returns:
        dict: A nested dictionary with the same structure as the input,
              mapping filenames to lists of function definitions. Each definition
              for a top-level __global__ function is prepended with the source code
              of any functions it calls.
    """
    extracted_defs = defaultdict(lambda: defaultdict(list))
    all_functions = {}  # Maps function names to their node, source, and type
    all_function_calls = defaultdict(set) # Maps function name to set of called functions
    device_variables = defaultdict(dict) # Maps file_path to {var_name: declaration_text}
    preprocessor_defs = defaultdict(dict) # Maps file_path to {macro_name: definition_text}


    # First pass: Collect all global and device functions and __device__ variables from all files
    for dir_name, files in tqdm(all_files_dict.items(), desc="First pass: Collecting functions"):
        for file_path in tqdm(files, desc=f"Processing {dir_name}", leave=False):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()
                tree = parser.parse(bytes(source_code, "utf8"))
                # Find all __device__ variables declarations anywhere in the file
                find_device_variable_declarations(tree.root_node, file_path, device_variables)
                find_preprocessor_defs(tree.root_node, file_path, preprocessor_defs)
                # Collect all function and template declarations anywhere in the file
                for declaration_node in find_all_declaration_nodes(tree.root_node):
                    # The text needs to be from the outer node for template_declarations
                    if declaration_node.type == 'template_declaration':
                        func_def_child = next((child for child in declaration_node.children if child.type == 'function_definition'), None)
                        if not func_def_child:
                            continue
                    declaration = declaration_node.text.decode('utf8').split('{')[0]
                    func_name = get_function_name(declaration_node)
                    if not func_name:
                        continue

                    func_type = None
                     
                    func_def_node = declaration_node
                    if declaration_node.type == 'template_declaration':
                        func_def_node = next((child for child in declaration_node.children if child.type == 'function_definition'), None)

                    if func_def_node:
                        specifiers_nodes = []
                        for child in func_def_node.children:
                            if child.type in ['function_declarator', 'compound_statement']:
                                break
                            specifiers_nodes.append(child)
                        
                        # Deep search for __global__ or __device__ nodes
                        q = specifiers_nodes
                        while q:
                            curr = q.pop(0)
                            if curr.type == '__global__':
                                func_type = 'global'
                                break
                            if curr.type == '__device__':
                                func_type = 'device'
                                break
                            q.extend(curr.children)
                        
                    if func_type:
                        all_functions[func_name] = {
                            'node': declaration_node, # Use the outer node for call graph analysis
                            'source': declaration_node.text.decode('utf8'),
                            'type': func_type,
                            'file_path': file_path,
                            'dir_name': dir_name
                        }
            except Exception as e:
                print(f"Error processing file {file_path} in first pass: {e}")

    # Second pass: Build the call graph for all functions
    for func_name, func_data in tqdm(all_functions.items(), desc="Second pass: Building call graph"):
        find_function_calls(func_data['node'], all_function_calls[func_name])

    # Identify all functions that are called by other functions
    called_functions = set()
    for calls in all_function_calls.values():
        called_functions.update(calls)

    # Third pass: Identify top-level __global__ functions and generate their code
    # Only include functions explicitly marked as global and not called by others
    # Determine which functions are true top-level __global__ kernels
    top_level_global_funcs = {
        name: data
        for name, data in all_functions.items()
        # only dict entries with explicit 'global' type and never called
        if isinstance(data, dict)
           and data.get('type') == 'global'
           and name not in called_functions
    }


    for func_name, func_data in tqdm(top_level_global_funcs.items(), desc="Third pass: Generating code"):
        # The dependencies are all functions in the call graph starting from this function.
        # We use a new visited set for each top-level function.
        deps_to_prepend = get_transitive_calls(func_name, set(), all_function_calls, all_functions)
        
        # Get all functions in the call chain, including the top-level one
        call_chain_func_names = [func_name] + deps_to_prepend
        call_chain_funcs = [all_functions[name] for name in call_chain_func_names if name in all_functions]

        # Find all identifiers used in the function bodies of the call chain
        used_identifiers = set()
        for f_data in call_chain_funcs:
            body = next((child for child in f_data['node'].children if child.type == 'compound_statement'), None)
            if body:
                find_used_identifiers(body, used_identifiers)

        # Collect all device variables from all files to check against used identifiers
        all_device_vars = {}
        for file_vars in device_variables.values():
            all_device_vars.update(file_vars)
        
        # Collect all preprocessor defs from all files
        all_preproc_defs = {}
        for file_defs in preprocessor_defs.values():
            all_preproc_defs.update(file_defs)

        # Find used macros
        used_macros = set()
        for f_data in call_chain_funcs:
            find_used_macros(f_data['node'], used_macros, all_preproc_defs)

        # Filter device variables to only those used in the call chain
        used_device_vars = set()
        for var_name, declaration in all_device_vars.items():
            if var_name in used_identifiers:
                used_device_vars.add(declaration)

        prepended_code = sorted(list(used_device_vars))

        # Add used preprocessor definitions
        for macro_name in sorted(list(used_macros)):
            if macro_name in all_preproc_defs:
                prepended_code.append(all_preproc_defs[macro_name])

        for dep_name in deps_to_prepend:
            if dep_name in all_functions:
                prepended_code.append(all_functions[dep_name]['source'])
        
        final_code = "\n\n".join(prepended_code)
        if final_code:
            final_code += "\n\n"
        final_code += func_data['source']
        
        extracted_defs[func_data['dir_name']][func_data['file_path']].append(final_code)

    return json.loads(json.dumps(extracted_defs))


if __name__ == '__main__':
    files_dict = get_all_files()
    if files_dict:
        cuda_defs = extract_all_CUDA_function_defs(files_dict)
        
        output_filename = "extracted_CUDA_kernels.json"
        with open(output_filename, 'w') as f:
            json.dump(cuda_defs, f, indent=4)
        
        print(f"Extracted CUDA kernels saved to [{output_filename}]")
