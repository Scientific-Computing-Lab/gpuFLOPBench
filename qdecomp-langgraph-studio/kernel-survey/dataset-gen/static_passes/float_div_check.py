from .TargetKernel import TargetKernel

def check_has_float_division(input: TargetKernel):
    """    Check if the kernel source code contains floating-point division operations."""

    floating_point_types = ['float', 'double', 'nv_bfloat16', '__half', '__nv_half', '__nv_bfloat16', 'nv_half', 'bfloat16']

    # Keep track of variables that are of type float
    float_variables = set()

    def find_identifier_in_declarator(declarator_node):
        """Recursively finds the identifier node within a declarator node."""
        if not declarator_node:
            return None
        if declarator_node.type == 'identifier':
            return declarator_node
        
        # Most complex declarators (pointer, array, init) have a 'declarator' field.
        child_declarator = declarator_node.child_by_field_name('declarator')
        if child_declarator:
            return find_identifier_in_declarator(child_declarator)
        
        return None

    # Helper function to get the type of an expression
    def get_expression_type(node, float_variables):
        if not node:
            return 'unknown'
        if node.type == 'float_literal':
            return 'float'
        if node.type == 'identifier' and node.text.decode() in float_variables:
            return 'float'
        if node.type == 'call_expression':
            # This is a simplification. We might need a list of functions that return floats.
            # For now, let's assume functions with 'f' in their name might return floats, e.g., 'sqrtf'.
            function_name_node = node.child_by_field_name('function')
            if function_name_node and 'f' in function_name_node.text.decode().lower():
                return 'float'
        if node.type == 'cast_expression':
            type_node = node.child_by_field_name('type')
            if type_node and type_node.text.decode() in floating_point_types:
                return 'float'
        if node.type == 'binary_expression':
            left_type = get_expression_type(node.child_by_field_name('left'), float_variables)
            right_type = get_expression_type(node.child_by_field_name('right'), float_variables)
            if left_type == 'float' or right_type == 'float':
                return 'float'
        if node.type == 'parenthesized_expression':
            # Look inside parentheses: (expr)
            return get_expression_type(node.named_child(0), float_variables)
        return 'unknown'


    # Walk the tree to find float variable declarations and division operations
    nodes_to_visit = [input.root_node]
    
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        
        # Check for float variable declarations
        if node.type == 'declaration':
            type_node = node.child_by_field_name('type')
            if type_node and type_node.text.decode() in floating_point_types:
                for declarator_node in node.children_by_field_name('declarator'):
                    identifier_node = find_identifier_in_declarator(declarator_node)
                    if identifier_node:
                        float_variables.add(identifier_node.text.decode())
            
            # Handle variables initialized with a float value or cast
            for declarator_node in node.children_by_field_name('declarator'):
                if declarator_node.type == 'init_declarator':
                    value_node = declarator_node.child_by_field_name('value')
                    if get_expression_type(value_node, float_variables) == 'float':
                        identifier_node = find_identifier_in_declarator(declarator_node)
                        if identifier_node:
                            float_variables.add(identifier_node.text.decode())

        # Check for assignments that result in a float type
        if node.type == 'assignment_expression':
            right_node = node.child_by_field_name('right')
            if get_expression_type(right_node, float_variables) == 'float':
                left_node = node.child_by_field_name('left')
                if left_node and left_node.type == 'identifier':
                    float_variables.add(left_node.text.decode())

            # --- new: catch '/=' compound‐division (handles '/  =' too) ---
            operator_node = node.child_by_field_name('operator')
            if operator_node:
                # strip out all whitespace in the operator token
                op = ''.join(operator_node.text.decode().split())
                if op == '/=':
                    left_operand  = node.child_by_field_name('left')
                    right_operand = node.child_by_field_name('right')
                    left_type  = get_expression_type(left_operand,  float_variables)
                    right_type = get_expression_type(right_operand, float_variables)
                    if left_type == 'float' or right_type == 'float':
                        input.has_float_division = True
                        input.float_division_line_num.append(node.start_point[0] + 1)

        # Check for division operations
        if node.type == 'binary_expression' and node.children[1].type == '/':
            left_operand = node.child_by_field_name('left')
            right_operand = node.child_by_field_name('right')

            left_type = get_expression_type(left_operand, float_variables)
            right_type = get_expression_type(right_operand, float_variables)

            if left_type == 'float' or right_type == 'float':
                input.has_float_division = True
                input.float_division_line_num.append(node.start_point[0] + 1)

        # Check for fdividef function call
        if node.type == 'call_expression':
            function_node = node.child_by_field_name('function')
            if function_node and function_node.text.decode() == 'fdividef':
                input.has_float_division = True
                input.float_division_line_num.append(node.start_point[0] + 1)

        # Add children to the queue for visiting
        for child in node.children:
            nodes_to_visit.append(child)
