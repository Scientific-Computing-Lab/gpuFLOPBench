from .preprocessing_common import *

# -------------------------------------------------------------------
# A simple text‐replacement rewriter based on byte offsets.
# -------------------------------------------------------------------
class Rewriter:
    def __init__(self, source_bytes: bytes):
        self._source = source_bytes
        self._edits = []

    def replace(self, start_byte: int, end_byte: int, new_text: str):
        self._edits.append((start_byte, end_byte, new_text.encode('utf8')))

    def apply(self) -> bytes:
        """
        Apply all edits in reverse order and return the new source bytes.
        """
        result = self._source
        for start, end, replacement in sorted(self._edits, key=lambda e: -e[0]):
            result = result[:start] + replacement + result[end:]
        return result


# -------------------------------------------------------------------
# PASS 1: Collect argv-based and simple literal assignments.
# -------------------------------------------------------------------
def collect_initial_literals(tree, source_bytes: bytes, argv_values: list, rewriter: Rewriter) -> dict:
    """
    Walk the AST, replace argv[N] and atoi(argv[N]) calls with literals,
    record simple string/number assignments into var_map.
    """
    var_map = {}

    def text_for_node(node):
        return source_bytes[node.start_byte:node.end_byte].decode('utf8')

    def visit(node, parent=None):
        # Replace argv[N] → "value"
        if node.type == 'subscript_expression':
            left, index_list = node.children[0], node.children[1]
            if (left.type == 'identifier' and text_for_node(left) == 'argv'
                    and index_list.type == 'subscript_argument_list'
                    and len(index_list.children) >= 2):
                idx = int(text_for_node(index_list.children[1]))
                if 0 <= idx < len(argv_values):
                    literal = argv_values[idx]
                    cpp_literal = '"' + literal.replace('"', '\\"') + '"'
                    rewriter.replace(node.start_byte, node.end_byte, cpp_literal)
                    return

        # Replace atoi(argv[N]) → integer literal, bind to var_map if assigned.
        if node.type == 'call_expression' and len(node.children) >= 2:
            function_node, args_node = node.children[0], node.children[1]
            if (function_node.type == 'identifier'
                    and text_for_node(function_node) == 'atoi'):
                for arg in args_node.named_children:
                    if arg.type == 'subscript_expression':
                        left, index_list = arg.children[0], arg.children[1]
                        if (left.type == 'identifier'
                                and text_for_node(left) == 'argv'
                                and index_list.type == 'subscript_argument_list'
                                and len(index_list.children) >= 2):
                            idx = int(text_for_node(index_list.children[1]))
                            if 0 <= idx < len(argv_values):
                                lit_str = argv_values[idx]
                                try:
                                    int_literal = str(int(lit_str))
                                except ValueError:
                                    int_literal = '0'
                                # record if this call is the RHS of an assignment
                                if parent and parent.type in (
                                    'assignment_expression',
                                    'variable_declaration',
                                    'init_declarator'
                                ):
                                    for child in parent.children:
                                        if child.type == 'identifier':
                                            var_map[text_for_node(child)] = int_literal
                                            break
                                rewriter.replace(node.start_byte, node.end_byte, int_literal)
                                return

        # Track simple assignments: x = "foo"; x = 42; x = y;
        if node.type in ('variable_declaration', 'assignment_expression'):
            lhs = None
            equals_index = None
            for idx, child in enumerate(node.children):
                if child.type == 'identifier' and lhs is None:
                    lhs = child
                if child.type == '=' and equals_index is None:
                    equals_index = idx
            if lhs and equals_index is not None and equals_index + 1 < len(node.children):
                rhs = node.children[equals_index + 1]
                rhs_text = text_for_node(rhs).strip()
                if rhs.type in ('string_literal', 'number_literal'):
                    var_map[text_for_node(lhs)] = rhs_text
                elif rhs.type == 'identifier' and rhs_text in var_map:
                    var_map[text_for_node(lhs)] = var_map[rhs_text]
                    rewriter.replace(rhs.start_byte, rhs.end_byte, var_map[rhs_text])

        for child in node.children:
            visit(child, node)

    visit(tree.walk().node, None)
    return var_map


# -------------------------------------------------------------------
# PASS 2: Propagate literals into call-site arguments.
# -------------------------------------------------------------------
def propagate_to_call_sites(tree, source_bytes: bytes, var_map: dict, rewriter: Rewriter):
    """
    Walk every call_expression's argument_list, and replace any identifier
    that matches var_map with its literal value.
    """
    def text_for_node(node):
        return source_bytes[node.start_byte:node.end_byte].decode('utf8')

    def visit(node):
        if node.type == 'call_expression':
            for child in node.children:
                if child.type == 'argument_list':
                    def replace_in_arg(subnode):
                        if subnode.type == 'identifier':
                            name = text_for_node(subnode)
                            if name in var_map:
                                rewriter.replace(subnode.start_byte,
                                                 subnode.end_byte,
                                                 var_map[name])
                                # keep the mapping for cascaded propagation
                                var_map[name] = var_map[name]
                        for grandchild in subnode.children:
                            replace_in_arg(grandchild)
                    replace_in_arg(child)
        for child in node.children:
            visit(child)

    visit(tree.root_node)


# -------------------------------------------------------------------
# PASS 3: Collect call-site literals across all files.
# -------------------------------------------------------------------
def collect_call_site_literals(source_text: str) -> dict:
    """
    Parse a single file and return a mapping:
      function_name -> { argument_index: literal_text }
    for all string_literal and number_literal arguments at call sites.
    """
    source_bytes = source_text.encode('utf8')
    tree = parser.parse(source_bytes)
    call_site_map = defaultdict(dict)

    def text_for_node(node):
        return source_bytes[node.start_byte:node.end_byte].decode('utf8')

    def visit(node):
        if node.type == 'call_expression':
            # find the argument_list
            args_node = next((c for c in node.children if c.type == 'argument_list'), None)
            # extract function name (first identifier under the first child)
            fn_node = node.child_by_field_name('function') or node.children[0]
            function_name = None

            def find_identifier(candidate):
                if candidate.type == 'identifier':
                    return text_for_node(candidate)
                for child in candidate.children:
                    result = find_identifier(child)
                    if result:
                        return result
                return None

            function_name = find_identifier(fn_node)

            if function_name and args_node:
                for index, argument in enumerate(args_node.named_children):
                    if argument.type in ('string_literal', 'number_literal'):
                        call_site_map[function_name].setdefault(index,
                                                                 text_for_node(argument))
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return call_site_map


# -------------------------------------------------------------------
# PASS 4: Inline collected literals into function definitions.
# -------------------------------------------------------------------
def inline_into_definitions(source_text: str, call_site_map: dict) -> str:
    """
    For each function_definition in the file, if its name appears in call_site_map,
    replace parameter-name occurrences in the body with the collected literals.
    """
    source_bytes = source_text.encode('utf8')
    tree = parser.parse(source_bytes)
    rewriter = Rewriter(source_bytes)

    def text_for_node(node):
        return source_bytes[node.start_byte:node.end_byte].decode('utf8')

    def visit(node):
        if node.type == 'function_definition':
            # find the function_declarator and its parameter_list
            declarator = next((c for c in node.children if c.type == 'function_declarator'), None)
            if not declarator:
                return
            identifier_node = next((c for c in declarator.named_children if c.type == 'identifier'), None)
            if not identifier_node:
                return

            function_name = text_for_node(identifier_node)
            literal_map = call_site_map.get(function_name)
            if not literal_map:
                return

            parameter_list = next((c for c in declarator.named_children
                                   if c.type == 'parameter_list'), None)
            if not parameter_list:
                return

            # collect parameter names in order
            param_names = []
            for idx, param in enumerate(parameter_list.named_children):
                name_node = next((c for c in param.named_children if c.type == 'identifier'), None)
                if name_node:
                    param_names.append((idx, text_for_node(name_node)))

            # walk the function body and replace identifiers
            body_node = next((c for c in node.named_children if c.type == 'compound_statement'), None)
            if body_node:
                def replace_in_body(subnode):
                    if subnode.type == 'identifier':
                        name = text_for_node(subnode)
                        for idx, param_name in param_names:
                            if idx in literal_map and name == param_name:
                                rewriter.replace(subnode.start_byte,
                                                 subnode.end_byte,
                                                 literal_map[idx])
                    for child in subnode.children:
                        replace_in_body(child)
                replace_in_body(body_node)

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return rewriter.apply().decode('utf8')


# -------------------------------------------------------------------
# Rewrite a single file to fixed-point for argv and local literals.
# -------------------------------------------------------------------
def rewrite_file_with_argv_and_locals(source_text: str, argv_values: list) -> str:
    """
    Repeatedly apply passes 1 and 2 until the file stops changing.
    """
    previous = source_text
    while True:
        source_bytes = previous.encode('utf8')
        rewriter = Rewriter(source_bytes)
        tree = parser.parse(source_bytes)

        var_map = collect_initial_literals(tree, source_bytes, argv_values, rewriter)
        propagate_to_call_sites(tree, source_bytes, var_map, rewriter)

        updated_bytes = rewriter.apply()
        updated_text = updated_bytes.decode('utf8')
        if updated_text == previous:
            return updated_text
        previous = updated_text


# -------------------------------------------------------------------
# Driver: two-pass propagation across files.
# -------------------------------------------------------------------
def propagate_argv_in_combined(combined_text: str, argv_values: list) -> str:
    # Split combined input into individual files
    file_sources = parse_combined_sources(combined_text)

    # PASS A: per-file propagation of argv[] and local assignments
    for filename, contents in file_sources.items():
        file_sources[filename] = rewrite_file_with_argv_and_locals(contents, argv_values)

    # PASS B: collect all call-site literals across files
    global_literal_map = defaultdict(dict)
    for contents in file_sources.values():
        per_file_map = collect_call_site_literals(contents)
        for func_name, literal_map in per_file_map.items():
            for index, literal in literal_map.items():
                global_literal_map[func_name].setdefault(index, literal)

    # PASS C: inline into every file's function definitions
    for filename, contents in file_sources.items():
        file_sources[filename] = inline_into_definitions(contents, global_literal_map)

    # Re-emit the combined format
    return render_combined_sources(file_sources)

