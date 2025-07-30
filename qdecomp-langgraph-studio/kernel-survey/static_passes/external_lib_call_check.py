from .TargetKernel import TargetKernel
from .math_fnct_check import CUDA_MATH_FUNCTIONS

IGNORE_CASES = CUDA_MATH_FUNCTIONS | {
    '__syncthreads',
    'reinterpret_cast',
    'static_cast',
}

def check_external_lib_calls(input: TargetKernel):
    """
    Check if the kernel source code contains calls to external functions.
    """
    # First, find all function definitions within the source code to exclude them from being considered external.
    defined_functions = set()
    queue = [input.root_node]
    while queue:
        node = queue.pop(0)
        if node.type == 'function_definition':
            declarator = node.child_by_field_name('declarator')
            if declarator:
                # Need to find the identifier within the declarator
                # e.g., could be pointer_declarator with another declarator inside
                while declarator.child_by_field_name('declarator'):
                    declarator = declarator.child_by_field_name('declarator')
                
                identifier_node = declarator.child_by_field_name('declarator') # for simple cases
                if identifier_node and identifier_node.type == 'identifier':
                     defined_functions.add(identifier_node.text.decode())
                elif declarator.type == 'identifier': # for __global__ void kernel_name()
                    defined_functions.add(declarator.text.decode())

        for child in node.children:
            queue.append(child)


    # Now, walk the tree to find call expressions
    nodes_to_visit = [input.root_node]
    
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        
        if node.type == 'call_expression':
            function_node = node.child_by_field_name('function')
            if function_node and function_node.type == 'identifier':
                function_name = function_node.text.decode()
                
                # Check if it's not defined in the current file and not a CUDA math function
                if (function_name not in defined_functions) and (function_name not in IGNORE_CASES):
                    input.has_external_function_calls = True
                    input.external_function_call_line_num.append(node.start_point[0] + 1)

        # Add children to the queue for visiting
        for child in node.children:
            nodes_to_visit.append(child)
