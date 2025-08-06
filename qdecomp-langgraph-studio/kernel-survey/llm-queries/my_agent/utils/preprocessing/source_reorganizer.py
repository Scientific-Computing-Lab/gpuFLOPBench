from .preprocessing_common import *

# -------------------------------------------------------------------
# Extract #include directives of form:
#   #include "file"
#   # include <file>
# -------------------------------------------------------------------
def extract_includes(source: str) -> set:
    pattern = re.compile(r'#\s*include\s*["<]\s*([^">]+?)\s*[">]')
    return {m.group(1).strip() for m in pattern.finditer(source)}

# -------------------------------------------------------------------
# Traverse the AST to collect declared and defined symbols
# (functions and global variables)
# -------------------------------------------------------------------
def collect_symbols(code_bytes: bytes):
    """
    Returns two sets:
      declared_symbols: names declared (functions or globals)
      defined_symbols: names defined (functions or globals with initializer/body)
    """
    tree = parser.parse(code_bytes)
    root = tree.root_node

    declared = set()
    defined = set()

    def traverse(node, inside_function=False):
        # Entering a function definition
        if node.type == 'function_definition':
            # collect the function name as definition
            for child in node.children:
                if child.type in ('declarator', 'function_declarator'):
                    name_node = next((c for c in child.children if c.type == 'identifier'), None)
                    if name_node:
                        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf8')
                        defined.add(name)
            # traverse function body but mark inside_function
            for c in node.children:
                traverse(c, inside_function=True)
            return

        # Handle declaration nodes at top-level or in headers
        if node.type == 'declaration' and not inside_function:
            # detect function declarations
            func_decls = [c for c in node.children if c.type == 'function_declarator']
            if func_decls:
                for fd in func_decls:
                    name_node = next((c for c in fd.children if c.type == 'identifier'), None)
                    if name_node:
                        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf8')
                        declared.add(name)
            else:
                # variable declarations / definitions
                # look for init_declarator_list or init_declarator children
                init_lists = [c for c in node.children if c.type == 'init_declarator_list']
                inits = [c for c in node.children if c.type == 'init_declarator']
                for lst in init_lists:
                    for init_decl in [c for c in lst.children if c.type == 'init_declarator']:
                        idn = next((c for c in init_decl.children if c.type == 'identifier'), None)
                        if idn:
                            var = code_bytes[idn.start_byte:idn.end_byte].decode('utf8')
                            # if initializer present, count as definition
                            if any(grand.type == 'initializer' for grand in init_decl.children):
                                defined.add(var)
                            else:
                                declared.add(var)
                for init_decl in inits:
                    idn = next((c for c in init_decl.children if c.type == 'identifier'), None)
                    if idn:
                        var = code_bytes[idn.start_byte:idn.end_byte].decode('utf8')
                        if any(grand.type == 'initializer' for grand in init_decl.children):
                            defined.add(var)
                        else:
                            declared.add(var)

        # Recurse into children
        for c in node.children:
            traverse(c, inside_function)

    traverse(root, inside_function=False)
    return declared, defined

# -------------------------------------------------------------------
# Build dependency graph and topologically sort
# -------------------------------------------------------------------
def generate_order(sources: dict) -> list:
    """
    Given a dict mapping filename to source code, returns an ordering of
    filenames so that:
      - #include dependencies come first
      - headers declaring symbols come before sources defining them
      - variable and function declarations/definitions are respected
    """
    graph = defaultdict(set)
    indegree = defaultdict(int)

    # 1. Include-based dependencies
    for fname, code in sources.items():
        incs = extract_includes(code)
        for dep in (i for i in incs if i in sources):
            graph[dep].add(fname)

    # 2. Symbol-based dependencies (functions & globals)
    sym_declared_in = defaultdict(set)
    sym_defined_in = defaultdict(set)
    for fname, code in sources.items():
        decls, defs = collect_symbols(code.encode('utf8'))
        for s in decls:
            sym_declared_in[s].add(fname)
        for s in defs:
            sym_defined_in[s].add(fname)

    for sym, headers in sym_declared_in.items():
        defs = sym_defined_in.get(sym, set())
        for h in headers:
            for s in defs:
                if h != s:
                    graph[h].add(s)

    # 3. Compute indegree for Kahn's algorithm
    all_files = set(sources.keys())
    for f in all_files:
        indegree.setdefault(f, 0)
    for u, vs in graph.items():
        for v in vs:
            indegree[v] += 1

    # 4. Topological sort
    queue = deque([f for f, d in indegree.items() if d == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, []):
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(order) != len(all_files):
        raise RuntimeError("Cycle detected in file dependencies")

    return order

# -------------------------------------------------------------------
# Reorganize combined sources end-to-end
# -------------------------------------------------------------------
def reorganize_source(combined: str, verbose: bool = False) -> str:
    sources = parse_combined_sources(combined)
    if verbose:
        print(f"\nReorganizing {len(sources)} source files...")
        print("Files found:", sources.keys())
    order = generate_order(sources)
    ordered = {name: sources[name] for name in order}
    return render_combined_sources(ordered)



# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
# if __name__ == '__main__':
#     combined_code = '''
# ------------------
# main.cu
# ------------------
# 
# #include <iostream>
# #include <string>
# #include "helper.cuh"
# 
# int main(int argc, char** argv) {
#     auto a = argv[1];
#     std::string b = a;
#     helper::print_upper(b);
#     return 0;
# }
# 
# ------------------
# helper.cu
# ------------------
# 
# #include "helper.cuh"
# #include <algorithm>
# #include <iostream>
# 
# namespace helper {
#     void print_upper(const std::string &s) {
#         std::string tmp = s;
#         std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);
#         std::cout << tmp << std::endl;
#     }
# }
# 
# ------------------
# helper.cuh
# ------------------
# 
# #pragma once
# #include <string>
# namespace helper {
#     void print_upper(const std::string &s);
# }
# '''
# print(reorganize_source(combined_code))

