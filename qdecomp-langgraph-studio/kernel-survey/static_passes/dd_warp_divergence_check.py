from .TargetKernel import TargetKernel

DIVERGENCE_NODE_TYPES = {
    'if_statement',
    'while_statement',
    'do_statement',
    'for_statement',
    'switch_statement',
    'conditional_expression',
}

def is_data_dependent(condition_node):
    """
    Traverses the condition's AST to check for data-dependent operations,
    like array subscripting or pointer dereferencing.
    """
    if not condition_node:
        return False
        
    queue = [condition_node]
    while queue:
        node = queue.pop(0)
        
        # Check for array access, e.g., array[i]
        if node.type == 'subscript_expression':
            return True
            
        # Check for pointer dereferencing, e.g., *ptr
        if node.type == 'pointer_expression':
            # The first child of a pointer_expression for dereferencing is '*'
            if node.children and node.children[0].type == '*':
                return True

        queue.extend(node.children)
        
    return False

def get_condition_node(node):
    """
    Extracts the condition node from a control flow statement node.
    """
    if node.type in ('if_statement', 'while_statement', 'switch_statement', 'conditional_expression'):
        return node.child_by_field_name('condition')
    
    if node.type == 'do_statement':
        # The condition is in a while clause at the end
        return node.child_by_field_name('condition')

    if node.type == 'for_statement':
        # The condition is the second expression in the parentheses
        # It's not a named field, so we have to get it by index among children.
        # Children are '(', declaration/expression, ';', condition, ';', update, ')', body
        for child in node.children:
            if child.type == ';':
                # The condition is the node after the first semicolon
                condition = child.next_sibling
                if condition and condition.type != ';':
                    return condition
                else: # No condition, e.g., for(;;)
                    return None
    return None


def check_has_dd_warp_divergence(input: TargetKernel):
    """
    Checks for data-dependent warp divergence. This occurs when a branch condition
    depends on data that can vary across threads in a warp (e.g., from an array).
    """
    
    nodes_to_visit = [input.root_node]
    
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        
        if node.type in DIVERGENCE_NODE_TYPES:
            condition_node = get_condition_node(node)
            
            if is_data_dependent(condition_node):
                input.has_data_dependent_warp_divergence = True
                input.data_dependent_warp_divergence_line_num.append(node.start_point[0] + 1)

        nodes_to_visit.extend(node.children)
