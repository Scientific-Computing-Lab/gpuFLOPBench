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
    
    def get_function_name(node):
        # function_definition -> function_declarator -> identifier
        declarator = next((child for child in node.children if child.type == 'function_declarator'), None)
        if declarator:
            name_node = next((child for child in declarator.children if child.type == 'identifier'), None)
            if name_node:
                return name_node.text.decode('utf8')
        return None

    def find_function_calls(node, called_functions):
        if node.type == 'call_expression':
            # call_expression -> identifier
            identifier_node = next((child for child in node.children if child.type == 'identifier'), None)
            if identifier_node:
                called_functions.add(identifier_node.text.decode('utf8'))
        
        for child in node.children:
            find_function_calls(child, called_functions)

    def find_used_identifiers(node, identifiers):
        if node.type == 'identifier':
            identifiers.add(node.text.decode('utf8'))
        
        for child in node.children:
            find_used_identifiers(child, identifiers)

    def find_used_macros(node, used_macros, all_defs):
        if node.type == 'type_identifier' and node.text.decode('utf8') in all_defs:
            used_macros.add(node.text.decode('utf8'))

        for child in node.children:
            find_used_macros(child, used_macros, all_defs)

    def find_device_variable_declarations(node, file_path):
        if node.type == 'declaration' and '__device__' in node.text.decode('utf8'):
            declaration_text = node.text.decode('utf8')
            
            declarator_list = next((child for child in node.children if child.type in ['init_declarator_list', 'declaration_list']), None)
            array_declarator = next((child for child in node.children if child.type == 'array_declarator'), None)

            if declarator_list:
                for declarator in declarator_list.children:
                    if declarator.type == 'init_declarator':
                        identifier_node = next((child for child in declarator.children if child.type == 'identifier'), None)
                        if not identifier_node: # check in nested declarator
                            nested_declarator = next((child for child in declarator.children if child.type == 'declarator'), None)
                            if nested_declarator:
                                identifier_node = next((child for child in nested_declarator.children if child.type == 'identifier'), None)
                        
                        if identifier_node:
                            var_name = identifier_node.text.decode('utf8')
                            device_variables[file_path][var_name] = declaration_text
            elif array_declarator:
                identifier_node = next((child for child in array_declarator.children if child.type == 'identifier'), None)
                if identifier_node:
                    var_name = identifier_node.text.decode('utf8')
                    device_variables[file_path][var_name] = declaration_text
        
        for child in node.children:
            find_device_variable_declarations(child, file_path)

    def find_preprocessor_defs(node, file_path):
        if node.type == 'preproc_def':
            define_text = node.text.decode('utf8')
            children = node.children
            # #define NAME ...
            if len(children) > 1 and children[1].type == 'identifier':
                macro_name = children[1].text.decode('utf8')
                preprocessor_defs[file_path][macro_name] = define_text
        
        for child in node.children:
            find_preprocessor_defs(child, file_path)

    # First pass: Collect all global and device functions and __device__ variables from all files
    for dir_name, files in all_files_dict.items():
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()
                
                tree = parser.parse(bytes(source_code, "utf8"))
                
                # Find all __device__ variables declarations anywhere in the file
                find_device_variable_declarations(tree.root_node, file_path)
                find_preprocessor_defs(tree.root_node, file_path)

                for node in tree.root_node.children:
                    if node.type == 'function_definition':
                        declaration = node.text.decode('utf8').split('{')[0]
                        func_name = get_function_name(node)
                        if not func_name:
                            continue

                        func_type = None
                        if '__global__' in declaration:
                            func_type = 'global'
                        elif '__device__' in declaration:
                            func_type = 'device'
                        
                        if func_type:
                            all_functions[func_name] = {
                                'node': node,
                                'source': node.text.decode('utf8'),
                                'type': func_type,
                                'file_path': file_path,
                                'dir_name': dir_name
                            }
            except Exception as e:
                print(f"Error processing file {file_path} in first pass: {e}")

    # Second pass: Build the call graph for all functions
    for func_name, func_data in all_functions.items():
        find_function_calls(func_data['node'], all_function_calls[func_name])

    # Identify all functions that are called by other functions
    called_functions = set()
    for calls in all_function_calls.values():
        called_functions.update(calls)

    # Third pass: Identify top-level global functions and generate their code
    top_level_global_funcs = {
        name: data for name, data in all_functions.items()
        if data['type'] == 'global' and name not in called_functions
    }

    def get_transitive_calls(func_name, visited):
        if func_name in visited:
            return []
        visited.add(func_name)
        
        dependencies = []
        # Direct dependencies in a deterministic order
        direct_calls = sorted(list(all_function_calls.get(func_name, set())))
        
        # Recursively find dependencies of dependencies
        for called_func in direct_calls:
            if called_func in all_functions:
                # Pass the same visited set to the recursive call
                deps = get_transitive_calls(called_func, visited)
                for d in deps:
                    if d not in dependencies:
                        dependencies.append(d)
        
        # Add the direct dependencies after their own dependencies
        for called_func in direct_calls:
             if called_func in all_functions and called_func not in dependencies:
                dependencies.append(called_func)

        return dependencies

    for func_name, func_data in top_level_global_funcs.items():
        # The dependencies are all functions in the call graph starting from this function.
        # We use a new visited set for each top-level function.
        deps_to_prepend = get_transitive_calls(func_name, set())
        
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
