from tree_sitter import Language, Parser
import tree_sitter_cuda

CUDA_LANGUAGE = Language(tree_sitter_cuda.language())
parser = Parser(CUDA_LANGUAGE)


def get_function_name(node):
    if node.type == 'template_declaration':
        # template_declaration -> function_definition -> function_declarator -> identifier
        func_def = next((child for child in node.children if child.type == 'function_definition'), None)
        if func_def:
            return get_function_name(func_def)  # Recurse on the function_definition node
    elif node.type == 'function_definition':
        # function_definition -> function_declarator -> identifier (including templates)
        declarator = next((child for child in node.children if child.type == 'function_declarator'), None)
        if declarator:
            # recursively find the first identifier under declarator
            def find_id(n):
                if n.type == 'identifier':
                    return n
                for c in n.children:
                    res = find_id(c)
                    if res:
                        return res
                return None

            name_node = find_id(declarator)
            if name_node:
                return name_node.text.decode('utf8')
    return None



def find_function_calls(node, called_functions):
    if node.type == 'call_expression':
        # A call expression can be a simple identifier or a template function.
        
        # Case 1: Simple identifier (e.g., myFunction())
        # call_expression -> identifier
        identifier_node = next((child for child in node.children if child.type == 'identifier'), None)
        if identifier_node:
            called_functions.add(identifier_node.text.decode('utf8'))

        # Case 2: Template function (e.g., myTemplate<T>())
        # call_expression -> template_function -> identifier
        template_function_node = next((child for child in node.children if child.type == 'template_function'), None)
        if template_function_node:
            # The identifier is a child of template_function
            template_identifier_node = next((child for child in template_function_node.children if child.type == 'identifier'), None)
            if template_identifier_node:
                called_functions.add(template_identifier_node.text.decode('utf8'))
    
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

def find_device_variable_declarations(node, file_path, device_variables):
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
        find_device_variable_declarations(child, file_path, device_variables)



def find_preprocessor_defs(node, file_path, preprocessor_defs):
    if node.type == 'preproc_def':
        define_text = node.text.decode('utf8')
        children = node.children
        # #define NAME ...
        if len(children) > 1 and children[1].type == 'identifier':
            macro_name = children[1].text.decode('utf8')
            preprocessor_defs[file_path][macro_name] = define_text
    
    for child in node.children:
        find_preprocessor_defs(child, file_path, preprocessor_defs)

def find_all_declaration_nodes(root_node):
    """
    Recursively collect all function_definition and template_declaration nodes in the AST.
    """
    nodes = []
    def recurse(n):
        if n.type in ('function_definition', 'template_declaration'):
            nodes.append(n)
        for c in n.children:
            recurse(c)
    recurse(root_node)
    return nodes



def get_transitive_calls(func_name, visited, all_function_calls, all_functions):
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
            deps = get_transitive_calls(called_func, visited, all_function_calls, all_functions)
            for d in deps:
                if d not in dependencies:
                    dependencies.append(d)
    
    # Add the direct dependencies after their own dependencies
    for called_func in direct_calls:
         if called_func in all_functions and called_func not in dependencies:
            dependencies.append(called_func)

    return dependencies