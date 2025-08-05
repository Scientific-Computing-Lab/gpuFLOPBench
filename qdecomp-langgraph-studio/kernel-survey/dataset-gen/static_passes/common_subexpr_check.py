from .TargetKernel import TargetKernel
from collections import defaultdict

# We are interested in expressions that might involve some computation.
# A simple heuristic is to check for expressions that are not just simple identifiers.
EXPRESSION_TYPES_TO_CHECK = {
    'binary_expression',
    'unary_expression',
    'call_expression',
    'subscript_expression',
    'field_expression',
    'parenthesized_expression',
}

def check_has_common_subexpr(input: TargetKernel):
    """
    Finds repeated non-trivial subexpressions within function bodies.
    This check is focused on subexpressions involving floating point numbers.
    """
    floating_point_types = ['float', 'double', 'nv_bfloat16', '__half', '__nv_half', '__nv_bfloat16', 'nv_half', 'bfloat16']
    float_variables = set()

    def find_identifier_in_declarator(declarator_node):
        """Recursively finds the identifier node within a declarator node."""
        if not declarator_node:
            return None
        if declarator_node.type == 'identifier':
            return declarator_node
        
        child_declarator = declarator_node.child_by_field_name('declarator')
        if child_declarator:
            return find_identifier_in_declarator(child_declarator)
        
        return None

    def get_expression_type(node, float_variables):
        if not node:
            return 'unknown'
        if node.type == 'float_literal':
            return 'float'
        if node.type == 'identifier' and node.text.decode() in float_variables:
            return 'float'
        if node.type == 'call_expression':
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
            return get_expression_type(node.named_child(0), float_variables)
        return 'unknown'

    # First pass: find all float variables
    q = [input.root_node]
    while q:
        node = q.pop(0)
        if node.type == 'declaration':
            type_node = node.child_by_field_name('type')
            if type_node and type_node.text.decode() in floating_point_types:
                for declarator_node in node.children_by_field_name('declarator'):
                    identifier_node = find_identifier_in_declarator(declarator_node)
                    if identifier_node:
                        float_variables.add(identifier_node.text.decode())
            
            for declarator_node in node.children_by_field_name('declarator'):
                if declarator_node.type == 'init_declarator':
                    value_node = declarator_node.child_by_field_name('value')
                    if get_expression_type(value_node, float_variables) == 'float':
                        identifier_node = find_identifier_in_declarator(declarator_node)
                        if identifier_node:
                            float_variables.add(identifier_node.text.decode())

        if node.type == 'assignment_expression':
            right_node = node.child_by_field_name('right')
            if get_expression_type(right_node, float_variables) == 'float':
                left_node = node.child_by_field_name('left')
                if left_node and left_node.type == 'identifier':
                    float_variables.add(left_node.text.decode())
        
        q.extend(node.children)

    # Find all function definitions in the kernel source
    function_definitions = [node for node in input.root_node.children if node.type == 'function_definition']

    all_found_lines = set()

    for func_def in function_definitions:
        body = func_def.child_by_field_name('body')
        if not body:
            continue

        expressions_map = defaultdict(list)
        
        queue = [body]
        visited_nodes = set()

        while queue:
            node = queue.pop(0)
            
            if node in visited_nodes:
                continue
            visited_nodes.add(node)

            if node.type in EXPRESSION_TYPES_TO_CHECK:
                if get_expression_type(node, float_variables) == 'float':
                    expr_text = node.text.decode().strip()
                    expressions_map[expr_text].append(node.start_point[0] + 1)

            queue.extend(node.children)

        for expr, lines in expressions_map.items():
            if len(lines) > 1:
                input.has_common_subexpression = True
                for line in lines:
                    all_found_lines.add(line)

    if all_found_lines:
        input.common_subexpression_line_num = sorted(list(all_found_lines))
