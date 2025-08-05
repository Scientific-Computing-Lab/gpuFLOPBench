from .TargetKernel import TargetKernel

def find_function_name(declarator_node):
    """Helper function to extract the function name (identifier) from a declarator node."""
    if not declarator_node:
        return None
    
    if declarator_node.type == 'identifier':
        return declarator_node.text.decode()
    
    # Recurse through common declarator types that wrap the actual function name
    if declarator_node.type in ('pointer_declarator', 'function_declarator', 'parenthesized_declarator', 'array_declarator'):
        return find_function_name(declarator_node.child_by_field_name('declarator'))
        
    return None

def check_has_recursion(input: TargetKernel):
    """
    Check if the kernel source code contains recursive function calls.
    A recursive call is a call to a function from within its own body.
    """
    
    # Traverse the entire tree to find all function definitions
    function_definitions = []
    queue = [input.root_node]
    while queue:
        node = queue.pop(0)
        if node.type == 'function_definition':
            function_definitions.append(node)
        queue.extend(node.children)

    for func_def_node in function_definitions:
        # 1. Get the name of the function being defined.
        declarator = next((child for child in func_def_node.children if child.type == 'function_declarator'), None)
        function_name = find_function_name(declarator)
        
        if not function_name:
            continue

        # 2. Traverse the body of that function.
        body = func_def_node.child_by_field_name('body')
        if not body:
            continue
            
        traversal_queue = [body]
        
        while traversal_queue:
            current_node = traversal_queue.pop(0)
            
            # 3. Look for call_expression nodes.
            if current_node.type == 'call_expression':
                called_function_node = current_node.child_by_field_name('function')
                
                # Handle cases where the function call is through a pointer or other complex expression
                called_function_name = find_function_name(called_function_node)

                if called_function_name:
                    # 4. If it's a call to the same function, it's recursion.
                    if called_function_name == function_name:
                        input.has_recursion = True
                        input.recursion_line_num.append(current_node.start_point[0] + 1)

            # Add children to continue traversal, but don't go into nested functions
            # as that would be its own recursion check.
            for child in current_node.children:
                if child.type != 'function_definition':
                    traversal_queue.append(child)
