from .preprocessing_common import *

import graphviz

def collect_identifiers(node):
    """
    Recursively collect all identifier nodes under the given node.
    """
    results = []
    if node.type == 'identifier':
        results.append(node)
    for child in node.children:
        results.extend(collect_identifiers(child))
    return results

def build_dependency_graph(combined_sources: str) -> graphviz.Digraph:
    """
    Given a combined string of multiple CUDA source files, parse them,
    build a dependency graph of variables, functions, arguments, and literals,
    and return a graphviz.Digraph object.
    """
    # Parse the combined sources into individual files
    sources = parse_combined_sources(combined_sources)
    # Store parsed trees
    trees = {fname: parser.parse(src.encode('utf8')).root_node for fname, src in sources.items()}

    # Data structures to hold definitions
    functions = {}       # func_name -> node
    variables = set()    # var names
    arguments = set()    # (func_name, arg_name)
    literals = set()     # literal values

    # First pass: collect definitions
    def collect_defs(node):
        # Function definitions
        if node.type == 'function_definition':
            for decl in node.children:
                if decl.type == 'function_declarator':
                    # function name
                    for c in decl.children:
                        if c.type == 'identifier':
                            fname = c.text.decode('utf8')
                            functions[fname] = node
                    # parameters
                    for param_list in decl.children:
                        if param_list.type == 'parameter_list':
                            for p in param_list.named_children:
                                if p.type == 'parameter_declaration':
                                    for c2 in p.children:
                                        if c2.type == 'identifier':
                                            arguments.add((fname, c2.text.decode('utf8')))
                                            break
        # Variable declarations
        if node.type == 'init_declarator':
            for c in node.children:
                if c.type == 'identifier':
                    variables.add(c.text.decode('utf8'))
        # Literal definitions
        if node.type in ('string_literal', 'char_literal', 'number_literal'):
            literals.add(node.text.decode('utf8'))

        for child in node.children:
            collect_defs(child)

    for tree in trees.values():
        collect_defs(tree)

    # Prepare the graphviz Digraph
    g = graphviz.Digraph('DependencyGraph', format='png')
    added = set()
    def add_node(nid, label, shape='ellipse'):
        if nid not in added:
            g.node(nid, label=label, shape=shape)
            added.add(nid)

    # Create nodes
    for fname in functions:
        add_node(f'func:{fname}', fname, shape='box')
    for var in variables:
        add_node(f'var:{var}', var, shape='oval')
    for fname, arg in arguments:
        add_node(f'arg:{fname}:{arg}', arg, shape='diamond')
    for lit in literals:
        add_node(f'lit:{lit}', lit, shape='note')

    # Second pass: collect dependencies (edges)
    def traverse(node, current_func=None):
        # Enter function context
        if node.type == 'function_definition':
            for decl in node.children:
                if decl.type == 'function_declarator':
                    for c in decl.children:
                        if c.type == 'identifier':
                            current_func = c.text.decode('utf8')

        # Function call
        if node.type == 'call_expression':
            fn_node = node.child_by_field_name('function')
            if fn_node and fn_node.type == 'identifier':
                called = fn_node.text.decode('utf8')
                if called in functions:
                    src = f'func:{current_func or "global"}'
                    dst = f'func:{called}'
                    add_node(src, current_func or 'global', shape='box')
                    add_node(dst, called, shape='box')
                    g.edge(src, dst, label='calls')

        # Assignment (write and read)
        if node.type == 'assignment_expression':
            left = node.child_by_field_name('left')
            right = node.child_by_field_name('right')
            if left and left.type == 'identifier':
                var = left.text.decode('utf8')
                if var in variables:
                    src = f'func:{current_func or "global"}'
                    dst = f'var:{var}'
                    add_node(src, current_func or 'global', shape='box')
                    add_node(dst, var, shape='oval')
                    g.edge(src, dst, label='writes')
            # read from right side
            if right:
                for idn in collect_identifiers(right):
                    name = idn.text.decode('utf8')
                    if name in variables:
                        src = f'var:{name}'
                        dst = f'func:{current_func or "global"}'
                        add_node(src, name, shape='oval')
                        add_node(dst, current_func or 'global', shape='box')
                        g.edge(src, dst, label='reads')

        # Return literal
        if node.type == 'return_statement':
            for c in node.children:
                if c.type in ('string_literal', 'number_literal', 'char_literal'):
                    lit = c.text.decode('utf8')
                    src = f'func:{current_func or "global"}'
                    dst = f'lit:{lit}'
                    add_node(src, current_func or 'global', shape='box')
                    add_node(dst, lit, shape='note')
                    g.edge(src, dst, label='returns')

        # Identifier read outside assignment
        if node.type == 'identifier':
            parent = node.parent
            skip = ('init_declarator', 'function_declarator', 'parameter_declaration')
            if parent and parent.type not in skip:
                name = node.text.decode('utf8')
                if name in variables:
                    src = f'var:{name}'
                    dst = f'func:{current_func or "global"}'
                    add_node(src, name, shape='oval')
                    add_node(dst, current_func or 'global', shape='box')
                    g.edge(src, dst, label='reads')

        for child in node.children:
            traverse(child, current_func)

    for tree in trees.values():
        traverse(tree, current_func=None)

    return g