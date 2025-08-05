from .TargetKernel import TargetKernel

DIVERGENCE_NODE_TYPES = {
    'if_statement',
    'while_statement',
    'do_statement',
    'for_statement',
    'switch_statement',
    'conditional_expression',
}

def check_has_warp_divergence(input: TargetKernel):
    """
    Checks for potential warp divergence by looking for control flow statements
    (if, for, while, do-while, switch) inside the kernel.
    This is a simple check and may flag constructs that are not truly divergent
    if the condition is uniform across the warp.
    """
    
    nodes_to_visit = [input.root_node]
    
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        
        if node.type in DIVERGENCE_NODE_TYPES:
            # Any of these control flow statements is a potential source of warp divergence.
            input.has_warp_divergence = True
            input.warp_divergence_line_num.append(node.start_point[0] + 1)

        # Continue traversing the tree
        nodes_to_visit.extend(node.children)
