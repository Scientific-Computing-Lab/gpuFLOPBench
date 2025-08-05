from .TargetKernel import TargetKernel

# List of common NVIDIA CUDA math function names (and their single-precision counterparts)
# This list is not exhaustive but covers many common cases.
CUDA_MATH_FUNCTIONS = {
    # Standard functions (double precision)
    'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
    'cbrt', 'ceil', 'copysign', 'cos', 'cosh', 'cospi', 'cyl_bessel_i0', 'cyl_bessel_i1',
    'erf', 'erfc', 'erfcinv', 'erfcx', 'erfinv',
    'exp', 'exp10', 'exp2', 'expm1',
    'fabs', 'fdim', 'floor', 'fma', 'fmax', 'fmin', 'fmod',
    'frexp', 'hypot', 'ilogb',
    'j0', 'j1', 'jn',
    'ldexp', 'lgamma',
    'llrint', 'llround', 'log', 'log10', 'log1p', 'log2', 'logb', 'lrint', 'lround',
    'modf', 'nan', 'nearbyint', 'nextafter',
    'norm', 'norm3d', 'norm4d', 'normcdf', 'normcdfinv',
    'pow', 'rcbrt', 'remainder', 'remquo', 'rhypot', 'rint',
    'rnorm', 'rnorm3d', 'rnorm4d', 'round', 'rsqrt',
    'scalbln', 'scalbn', 'sin', 'sincos', 'sincospi', 'sinh', 'sinpi', 'sqrt',
    'tan', 'tanh', 'tgamma', 'trunc',
    'y0', 'y1', 'yn',

    # Standard functions (single precision from previous list, for completeness)
    'sinf', 'cosf', 'tanf', 'asinf', 'acosf', 'atanf', 'atan2f',
    'sinhf', 'coshf', 'tanhf', 'asinhf', 'acoshf', 'atanhf',
    'expf', 'exp2f', 'exp10f', 'expm1f',
    'logf', 'log2f', 'log10f', 'log1pf',
    'powf', 'sqrtf', 'rsqrtf', 'cbrtf', 'rcbrtf',
    'hypotf', 'rhypotf',
    'ceilf', 'floorf', 'truncf', 'roundf',
    'fmaxf', 'fminf',
    'fmaf',
    'fabsf',
    'copysignf',
    'fdimf',
    'fmodf', 'remainderf',
    'frexpf', 'ldexpf', 'modff',
    'ilogbf', 'logbf',
    'scalbnf', 'scalblnf',
    'tgammaf', 'lgammaf',
    'erff', 'erfcf', 'erfinvf', 'erfcinvf', 'erfcxf',
    'normcdff', 'normcdfinvf',
    'j0f', 'j1f', 'jnf', 'y0f', 'y1f', 'ynf',
    'cyl_bessel_i0f', 'cyl_bessel_i1f',
    'nextafterf',
    'remquof',
    'nanf',
    'rintf', 'lrintf', 'llrintf',
    'roundf', 'lroundf', 'llroundf',
    'nearbyintf',
    'fdimf',
    'fmaf',
    'norm3df', 'norm4df', 'normf',
    'rnorm3df', 'rnorm4df', 'rnormf',
    'sincosf', 'sincospif', 'sinpif', 'cospif',

    # Type-agnostic and other functions from original list
    'abs',
    'signbit',
    'isfinite', 'isinf', 'isnan',
    'max', 'min',

    # Intrinsics (often prefixed with __)
    '__sinf', '__cosf', '__tanf',
    '__expf', '__exp10f',
    '__logf', '__log2f', '__log10f',
    '__powf',
    '__fmaf_rn', '__fmaf_rz', '__fmaf_ru', '__fmaf_rd',
    '__fdividef',
    '__saturatef',

    # Single precision intrinsics from URL
    'fdividef',

    # Double precision intrinsics
    '__dadd_rd', '__dadd_rn', '__dadd_ru', '__dadd_rz',
    '__dsub_rd', '__dsub_rn', '__dsub_ru', '__dsub_rz',
    '__dmul_rd', '__dmul_rn', '__dmul_ru', '__dmul_rz',
    '__ddiv_rd', '__ddiv_rn', '__ddiv_ru', '__ddiv_rz',
    '__drcp_rd', '__drcp_rn', '__drcp_ru', '__drcp_rz',
    '__dsqrt_rd', '__dsqrt_rn', '__dsqrt_ru', '__dsqrt_rz',
    '__fma_rd', '__fma_rn', '__fma_ru', '__fma_rz',

    # Half precision arithmetic functions
    '__habs', '__hadd', '__hadd_rn', '__hadd_sat', '__hdiv',
    '__hfma', '__hfma_relu', '__hfma_sat',
    '__hmul', '__hmul_rn', '__hmul_sat',
    '__hneg', '__hsub', '__hsub_rn', '__hsub_sat',
    'atomicAdd',

    # Half precision math functions
    'hceil', 'hcos', 'hexp', 'hexp10', 'hexp2', 'hfloor',
    'hlog', 'hlog10', 'hlog2', 'hrcp', 'hrint', 'hrsqrt',
    'hsin', 'hsqrt', 'htanh', 'htanh_approx', 'htrunc',
}

def check_has_math_fnct_calls(input: TargetKernel):
    """
    Checks if the kernel source code contains calls to NVIDIA CUDA math functions.
    """
    
    # Walk the tree to find call expressions
    nodes_to_visit = [input.root_node]
    found_lines = set()

    while nodes_to_visit:
        node = nodes_to_visit.pop(0)

        if node.type == 'call_expression':
            function_node = node.child_by_field_name('function')
            if function_node:
                function_name = function_node.text.decode()
                if function_name in CUDA_MATH_FUNCTIONS:
                    input.has_special_math_function = True
                    found_lines.add(node.start_point[0] + 1)

        # Add children to the queue for visiting
        nodes_to_visit.extend(node.children)

    if found_lines:
        input.special_math_function_line_num = sorted(list(found_lines))
