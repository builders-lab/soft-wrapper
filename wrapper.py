"""
wrapper.py — ctypes bindings for libsoft_cuda_python.so
Maps every sc_* function exported by sc_bridge.cpp with precise
argtypes / restype declarations.

Opaque pointer convention:
  sc_pool_t*   → c_void_p  (aliased as sc_pool_t)
  sc_tensor_t* → c_void_p  (aliased as sc_tensor_t)
  sc_graph_t*  → c_void_p  (aliased as sc_graph_t)

All result-returning functions use c_void_p to prevent 64-bit pointer
truncation that would occur with c_int or c_uint32.
"""

import ctypes
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend") # Naya backend folder path

# Yeh dono lines missing thi!
core_lib_path = os.path.join(backend_dir, "libsoft_lib.so")
bridge_lib_path = os.path.join(backend_dir, "libsoft_cuda_python.so")

# 1. Load the core engine globally so the bridge can resolve its symbols
ctypes.CDLL(core_lib_path, mode=ctypes.RTLD_GLOBAL)

# 2. Load the C-API bridge
lib = ctypes.CDLL(bridge_lib_path)

###############################################################################
# OPAQUE POINTER TYPE ALIASES
# (all are c_void_p; kept as named aliases for readability)
###############################################################################

sc_pool_t   = ctypes.c_void_p   # sc_pool_t*
sc_tensor_t = ctypes.c_void_p   # sc_tensor_t*
sc_graph_t  = ctypes.c_void_p   # sc_graph_t*


###############################################################################
# DTYPE / BACKEND CONSTANTS  (mirrors sc_bridge.cpp enums)
###############################################################################

class SCDtype:
    """Mirrors tensor_dtype_t as integer constants."""
    UINT32  = 0
    INT32   = 1
    UINT64  = 2
    INT64   = 3
    FLOAT32 = 4
    FLOAT64 = 5


class SCBackend:
    """Backend mode integers for sc_build_graph / sc_assign_backend."""
    GPU    = 0
    CPU    = 1
    HYBRID = 2


###############################################################################
# HELPER: bind a symbol with a clear error message on failure
###############################################################################

def _bind(name: str, argtypes, restype) -> ctypes.CDLL.__class__:
    """Attach argtypes / restype to lib.<name> and return the callable."""
    try:
        fn = getattr(lib, name)
    except AttributeError:
        raise AttributeError(
            f"Symbol '{name}' not found in {_LIB_NAME}. "
            f"Run `nm -D {_LIB_NAME} | grep {name}` to verify the export."
        )
    fn.argtypes = argtypes
    fn.restype  = restype
    return fn


###############################################################################
# POOL WRAPPERS
#   sc_pool_t *sc_pool_create(size_t capacity_bytes, int on_device)
#   void       sc_pool_destroy(sc_pool_t *pool)
#   void       sc_pool_zero(sc_pool_t *pool)
#   void      *sc_pool_alloc(sc_pool_t *pool, size_t size, uint32_t *out_id)
#   size_t     sc_pool_size(sc_pool_t *pool)
#   size_t     sc_pool_used(sc_pool_t *pool)
###############################################################################

sc_pool_create = _bind(
    "sc_pool_create",
    argtypes=[ctypes.c_size_t, ctypes.c_int],
    restype=ctypes.c_void_p,       # returns sc_pool_t*
)

sc_pool_destroy = _bind(
    "sc_pool_destroy",
    argtypes=[sc_pool_t],
    restype=None,
)

sc_pool_zero = _bind(
    "sc_pool_zero",
    argtypes=[sc_pool_t],
    restype=None,
)

sc_pool_alloc = _bind(
    "sc_pool_alloc",
    argtypes=[sc_pool_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)],
    restype=ctypes.c_void_p,
)

sc_pool_size = _bind(
    "sc_pool_size",
    argtypes=[sc_pool_t],
    restype=ctypes.c_size_t,
)

sc_pool_used = _bind(
    "sc_pool_used",
    argtypes=[sc_pool_t],
    restype=ctypes.c_size_t,
)


###############################################################################
# TENSOR CORE WRAPPERS
#   sc_tensor_t *sc_tensor_create(pool, dtype, num_dims, dims*, elems*, grad)
#   uint32_t     sc_tensor_id(sc_tensor_t*)
#   void        *sc_tensor_get_data(sc_tensor_t*)
#   uint8_t      sc_tensor_get_ndims(sc_tensor_t*)
#   uint32_t    *sc_tensor_get_dims(sc_tensor_t*)
#   void         sc_tensor_print_data(sc_tensor_t*)
#   int          sc_tensor_fill_random_normal(sc_tensor_t*, float, float)
###############################################################################

sc_tensor_create = _bind(
    "sc_tensor_create",
    argtypes=[
        sc_pool_t,                          # pool
        ctypes.c_int,                       # dtype  (SCDtype constant)
        ctypes.c_uint32,                    # num_dims
        ctypes.POINTER(ctypes.c_uint32),    # dims[]
        ctypes.c_void_p,                    # elems  (raw data buffer, may be NULL)
        ctypes.c_int,                       # grad   (bool → int)
    ],
    restype=ctypes.c_void_p,               # sc_tensor_t* – must NOT be c_int
)

sc_tensor_id = _bind(
    "sc_tensor_id",
    argtypes=[sc_tensor_t],
    restype=ctypes.c_uint32,
)

sc_tensor_get_data = _bind(
    "sc_tensor_get_data",
    argtypes=[sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_get_ndims = _bind(
    "sc_tensor_get_ndims",
    argtypes=[sc_tensor_t],
    restype=ctypes.c_uint8,
)

sc_tensor_get_dims = _bind(
    "sc_tensor_get_dims",
    argtypes=[sc_tensor_t],
    restype=ctypes.POINTER(ctypes.c_uint32),
)

sc_tensor_print_data = _bind(
    "sc_tensor_print_data",
    argtypes=[sc_tensor_t],
    restype=None,
)

sc_tensor_fill_random_normal = _bind(
    "sc_tensor_fill_random_normal",
    argtypes=[sc_tensor_t, ctypes.c_float, ctypes.c_float],
    restype=ctypes.c_int,   # bool return → int
)


###############################################################################
# TENSOR OPERATION WRAPPERS
#   sc_tensor_t *sc_tensor_mul(pool, x, y)
#   sc_tensor_t *sc_tensor_mul_naive(pool, x, y)
#   sc_tensor_t *sc_tensor_add(pool, x, y)
#   sc_tensor_t *sc_tensor_add_bias(pool, xw, bias)
#   sc_tensor_t *sc_tensor_sub(pool, a, b)
#   sc_tensor_t *sc_tensor_relu(pool, a)
#   sc_tensor_t *sc_tensor_mean(pool, a)
#   sc_tensor_t *sc_tensor_mse_loss(pool, predictions, target)
#   sc_tensor_t *sc_tensor_square(pool, x)
#   sc_tensor_t *sc_tensor_transpose(pool, a)
#   int          sc_tensor_evaluate(pool, t)
#   int          sc_tensor_evaluate_gpu(pool, t, d_a, d_b, d_res)
###############################################################################

sc_tensor_mul = _bind(
    "sc_tensor_mul",
    argtypes=[sc_pool_t, sc_tensor_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_mul_naive = _bind(
    "sc_tensor_mul_naive",
    argtypes=[sc_pool_t, sc_tensor_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_add = _bind(
    "sc_tensor_add",
    argtypes=[sc_pool_t, sc_tensor_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_add_bias = _bind(
    "sc_tensor_add_bias",
    argtypes=[sc_pool_t, sc_tensor_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_sub = _bind(
    "sc_tensor_sub",
    argtypes=[sc_pool_t, sc_tensor_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_relu = _bind(
    "sc_tensor_relu",
    argtypes=[sc_pool_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_mean = _bind(
    "sc_tensor_mean",
    argtypes=[sc_pool_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_mse_loss = _bind(
    "sc_tensor_mse_loss",
    argtypes=[sc_pool_t, sc_tensor_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_square = _bind(
    "sc_tensor_square",
    argtypes=[sc_pool_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_transpose = _bind(
    "sc_tensor_transpose",
    argtypes=[sc_pool_t, sc_tensor_t],
    restype=ctypes.c_void_p,
)

sc_tensor_evaluate = _bind(
    "sc_tensor_evaluate",
    argtypes=[sc_pool_t, sc_tensor_t],
    restype=ctypes.c_int,   # bool → int
)

sc_tensor_evaluate_gpu = _bind(
    "sc_tensor_evaluate_gpu",
    argtypes=[
        sc_pool_t,
        sc_tensor_t,
        ctypes.POINTER(ctypes.c_float),   # d_a
        ctypes.POINTER(ctypes.c_float),   # d_b
        ctypes.POINTER(ctypes.c_float),   # d_res
    ],
    restype=ctypes.c_int,
)


###############################################################################
# COMPUTATION GRAPH — LAYER 1 (fine-grained)
#   sc_graph_t *sc_graph_create()
#   void        sc_graph_destroy(sc_graph_t*)
#   int         sc_verify_dag(meta_pool, tensor, graph)
#   void        sc_assign_backend(pool_gpu, graph, mode)
#   void        sc_assign_grad_memory(pool_grad_cpu, pool_grad_gpu, graph)
#   int         sc_graph_forward(pool_cpu, pool_gpu, graph)
#   void        sc_autograd_gpu_transfer(graph)
#   void        sc_grad_initializer(graph)
#   int         sc_backward(graph)
#   void        sc_sgd(graph, learning_rate)
#   int         sc_node_to_host(graph, node_idx)
###############################################################################

sc_graph_create = _bind(
    "sc_graph_create",
    argtypes=[],
    restype=ctypes.c_void_p,   # sc_graph_t*
)

sc_graph_destroy = _bind(
    "sc_graph_destroy",
    argtypes=[sc_graph_t],
    restype=None,
)

sc_verify_dag = _bind(
    "sc_verify_dag",
    argtypes=[sc_pool_t, sc_tensor_t, sc_graph_t],
    restype=ctypes.c_int,   # bool → int
)

sc_assign_backend = _bind(
    "sc_assign_backend",
    argtypes=[sc_pool_t, sc_graph_t, ctypes.c_int],
    restype=None,
)

sc_assign_grad_memory = _bind(
    "sc_assign_grad_memory",
    argtypes=[sc_pool_t, sc_pool_t, sc_graph_t],
    restype=None,
)

sc_graph_forward = _bind(
    "sc_graph_forward",
    argtypes=[sc_pool_t, sc_pool_t, sc_graph_t],
    restype=ctypes.c_int,
)

sc_autograd_gpu_transfer = _bind(
    "sc_autograd_gpu_transfer",
    argtypes=[sc_graph_t],
    restype=None,
)

sc_grad_initializer = _bind(
    "sc_grad_initializer",
    argtypes=[sc_graph_t],
    restype=None,
)

sc_backward = _bind(
    "sc_backward",
    argtypes=[sc_graph_t],
    restype=ctypes.c_int,
)

sc_sgd = _bind(
    "sc_sgd",
    argtypes=[sc_graph_t, ctypes.c_float],
    restype=None,
)

sc_node_to_host = _bind(
    "sc_node_to_host",
    argtypes=[sc_graph_t, ctypes.c_size_t],
    restype=ctypes.c_int,
)


###############################################################################
# COMPUTATION GRAPH — LAYER 2 (convenience high-level)
#   sc_graph_t *sc_build_graph(meta_pool, pool_gpu, pool_grad_cpu,
#                               pool_grad_gpu, loss_tensor, backend_mode)
#   void        sc_graph_step(pool_cpu, pool_gpu, graph, learning_rate)
#   float       sc_graph_get_loss(graph)
#   size_t      sc_graph_size(graph)
###############################################################################

sc_build_graph = _bind(
    "sc_build_graph",
    argtypes=[
        sc_pool_t,    # meta_pool
        sc_pool_t,    # pool_gpu
        sc_pool_t,    # pool_grad_cpu
        sc_pool_t,    # pool_grad_gpu
        sc_tensor_t,  # loss (root tensor)
        ctypes.c_int, # backend_mode (SCBackend constant)
    ],
    restype=ctypes.c_void_p,   # sc_graph_t*
)

sc_graph_step = _bind(
    "sc_graph_step",
    argtypes=[
        sc_pool_t,    # pool_cpu
        sc_pool_t,    # pool_gpu
        sc_graph_t,
        ctypes.c_float,  # learning_rate
    ],
    restype=None,
)

sc_graph_get_loss = _bind(
    "sc_graph_get_loss",
    argtypes=[sc_graph_t],
    restype=ctypes.c_float,
)

sc_graph_size = _bind(
    "sc_graph_size",
    argtypes=[sc_graph_t],
    restype=ctypes.c_size_t,
)


###############################################################################
# IO WRAPPERS
#   int sc_save_model(path, tensors**, count)
#   int sc_load_model(path, tensors**, count)
###############################################################################

sc_save_model = _bind(
    "sc_save_model",
    argtypes=[
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),   # sc_tensor_t*[]
        ctypes.c_size_t,
    ],
    restype=ctypes.c_int,
)

sc_load_model = _bind(
    "sc_load_model",
    argtypes=[
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),   # sc_tensor_t*[]
        ctypes.c_size_t,
    ],
    restype=ctypes.c_int,
)
