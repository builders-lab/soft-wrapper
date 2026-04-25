"""
tensor.py — High-level Python API for the soft-cuda Deep Learning backend.

Classes
-------
MemoryPool        – Wraps sc_pool_t*.  Manages a single memory arena.
Tensor            – Wraps sc_tensor_t*.  Supports eager math operators.
ComputationGraph  – Wraps sc_graph_t*.  Drives the full training loop.

Design decisions
----------------
* Eager execution: every math operator calls both the builder sc_tensor_*
  and sc_tensor_evaluate() before returning, so the result is immediately
  materialised on the CPU.
* .data returns a NumPy array shaped to .shape (F32 by default).
* All resource handles are freed in __del__ to prevent leaks; classes also
  expose explicit .destroy() for deterministic cleanup.
* Type errors are caught early with clear messages rather than letting
  ctypes silently truncate or segfault.
"""

from __future__ import annotations

import ctypes
import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import wrapper as _w
from wrapper import (
    SCDtype, SCBackend,
    sc_pool_create, sc_pool_destroy, sc_pool_zero,
    sc_pool_alloc, sc_pool_size, sc_pool_used,
    sc_tensor_create, sc_tensor_id,
    sc_tensor_get_data, sc_tensor_get_ndims, sc_tensor_get_dims,
    sc_tensor_print_data, sc_tensor_fill_random_normal,
    sc_tensor_mul, sc_tensor_mul_naive, sc_tensor_add,
    sc_tensor_add_bias, sc_tensor_sub, sc_tensor_relu,
    sc_tensor_mean, sc_tensor_mse_loss, sc_tensor_square,
    sc_tensor_transpose, sc_tensor_evaluate, sc_tensor_evaluate_gpu,
    sc_graph_create, sc_graph_destroy, sc_verify_dag,
    sc_assign_backend, sc_assign_grad_memory,
    sc_graph_forward, sc_autograd_gpu_transfer,
    sc_grad_initializer, sc_backward, sc_sgd, sc_node_to_host,
    sc_build_graph, sc_graph_step, sc_graph_get_loss, sc_graph_size,
    sc_save_model, sc_load_model,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _np_dtype_for(sc_dtype: int) -> np.dtype:
    """Map SCDtype constant → numpy dtype."""
    _map = {
        SCDtype.UINT32:  np.uint32,
        SCDtype.INT32:   np.int32,
        SCDtype.UINT64:  np.uint64,
        SCDtype.INT64:   np.int64,
        SCDtype.FLOAT32: np.float32,
        SCDtype.FLOAT64: np.float64,
    }
    return np.dtype(_map.get(sc_dtype, np.float32))


def _ctypes_dtype_for(sc_dtype: int) -> type:
    """Map SCDtype constant → ctypes scalar type for pointer casts."""
    _map = {
        SCDtype.UINT32:  ctypes.c_uint32,
        SCDtype.INT32:   ctypes.c_int32,
        SCDtype.UINT64:  ctypes.c_uint64,
        SCDtype.INT64:   ctypes.c_int64,
        SCDtype.FLOAT32: ctypes.c_float,
        SCDtype.FLOAT64: ctypes.c_double,
    }
    return _map.get(sc_dtype, ctypes.c_float)


def _dims_array(shape: Sequence[int]) -> ctypes.Array:
    """Convert a Python shape tuple into a c_uint32 ctypes array."""
    arr = (ctypes.c_uint32 * len(shape))(*shape)
    return arr


###############################################################################
# MemoryPool
###############################################################################

class MemoryPool:
    """
    Wraps a single sc_pool_t arena.

    Parameters
    ----------
    capacity_bytes : int
        Total byte capacity of the pool.
    on_device : bool
        True  → GPU (device) pool.
        False → CPU (host) pool.
    """

    def __init__(self, capacity_bytes: int, on_device: bool = False) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive.")
        self._handle: int = sc_pool_create(capacity_bytes, int(on_device))
        if not self._handle:
            raise RuntimeError("sc_pool_create returned NULL. Out of memory?")
        self._destroyed = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def handle(self) -> int:
        """Raw c_void_p integer for passing to wrapper functions."""
        self._check_alive()
        return self._handle

    @property
    def capacity(self) -> int:
        """Total capacity of the pool in bytes."""
        return sc_pool_size(self._handle)

    @property
    def used(self) -> int:
        """Number of bytes currently allocated from the pool."""
        return sc_pool_used(self._handle)

    @property
    def free(self) -> int:
        """Remaining free bytes."""
        return self.capacity - self.used

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def zero(self) -> None:
        """Zero-fill the entire pool memory."""
        self._check_alive()
        sc_pool_zero(self._handle)

    def alloc(self, size: int) -> Tuple[int, int]:
        """
        Allocate *size* bytes from the pool.

        Returns
        -------
        (ptr, alloc_id) : (int, int)
            ptr      – raw c_void_p integer to the allocated block.
            alloc_id – uint32 id assigned by the pool.
        """
        self._check_alive()
        out_id = ctypes.c_uint32(0)
        ptr = sc_pool_alloc(self._handle, size, ctypes.byref(out_id))
        if not ptr:
            raise MemoryError(f"Pool allocation of {size} bytes failed.")
        return ptr, out_id.value

    def destroy(self) -> None:
        """Explicitly release the pool. Safe to call multiple times."""
        if not self._destroyed and self._handle:
            sc_pool_destroy(self._handle)
            self._destroyed = True
            self._handle = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("This MemoryPool has already been destroyed.")

    def __del__(self) -> None:
        self.destroy()

    def __repr__(self) -> str:
        if self._destroyed:
            return "<MemoryPool [destroyed]>"
        return (
            f"<MemoryPool capacity={self.capacity} "
            f"used={self.used} free={self.free}>"
        )


###############################################################################
# Tensor
###############################################################################

class Tensor:
    """
    Wraps a single sc_tensor_t*.

    Construction
    ------------
    Use the class-methods ``from_numpy``, ``zeros``, ``ones``, or
    ``random_normal`` rather than the raw constructor when possible.

    Eager execution
    ---------------
    Every operator overload (+, -, *, .relu(), .transpose(), …) immediately
    calls sc_tensor_evaluate() so the result tensor is fully materialised
    on the CPU before it is returned.  This mirrors NumPy semantics and
    simplifies debugging.

    Parameters
    ----------
    pool : MemoryPool
        Arena that owns the tensor's memory.
    handle : int
        Raw c_void_p pointer returned by an sc_tensor_* call.
    dtype : int
        SCDtype constant describing the element type (default FLOAT32).
    requires_grad : bool
        Whether gradients should be tracked for this tensor.
    """

    def __init__(
        self,
        pool: MemoryPool,
        handle: int,
        dtype: int = SCDtype.FLOAT32,
        requires_grad: bool = False,
    ) -> None:
        if not handle:
            raise RuntimeError(
                "Tensor handle is NULL. The sc_tensor_create call failed."
            )
        self._pool = pool
        self._handle: int = handle
        self._dtype = dtype
        self._requires_grad = requires_grad

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        pool: MemoryPool,
        array: np.ndarray,
        dtype: int = SCDtype.FLOAT32,
        requires_grad: bool = False,
    ) -> "Tensor":
        """
        Create a Tensor from a NumPy array.

        The data is copied into the pool; the NumPy array is not kept alive.
        """
        np_dtype = _np_dtype_for(dtype)
        arr = np.ascontiguousarray(array, dtype=np_dtype)
        shape = list(arr.shape)
        dims_arr = _dims_array(shape)
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)

        handle = sc_tensor_create(
            pool.handle,
            dtype,
            len(shape),
            dims_arr,
            data_ptr,
            int(requires_grad),
        )
        return cls(pool, handle, dtype, requires_grad)

    @classmethod
    def zeros(
        cls,
        pool: MemoryPool,
        shape: Sequence[int],
        dtype: int = SCDtype.FLOAT32,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a zero-filled tensor of the given shape."""
        np_dtype = _np_dtype_for(dtype)
        arr = np.zeros(shape, dtype=np_dtype)
        return cls.from_numpy(pool, arr, dtype, requires_grad)

    @classmethod
    def ones(
        cls,
        pool: MemoryPool,
        shape: Sequence[int],
        dtype: int = SCDtype.FLOAT32,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a ones-filled tensor of the given shape."""
        np_dtype = _np_dtype_for(dtype)
        arr = np.ones(shape, dtype=np_dtype)
        return cls.from_numpy(pool, arr, dtype, requires_grad)

    @classmethod
    def random_normal(
        cls,
        pool: MemoryPool,
        shape: Sequence[int],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: int = SCDtype.FLOAT32,
        requires_grad: bool = False,
    ) -> "Tensor":
        """
        Create a tensor of the given shape filled with random normal values.
        Delegates to sc_tensor_fill_random_normal for RNG consistency.
        """
        t = cls.zeros(pool, shape, dtype, requires_grad)
        ok = sc_tensor_fill_random_normal(t.handle, float(mean), float(std))
        if not ok:
            raise RuntimeError("sc_tensor_fill_random_normal failed.")
        return t

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def handle(self) -> int:
        """Raw c_void_p integer pointer."""
        return self._handle

    @property
    def pool(self) -> MemoryPool:
        return self._pool

    @property
    def dtype(self) -> int:
        return self._dtype

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def id(self) -> int:
        """Unique uint32 id assigned by the pool allocator."""
        return sc_tensor_id(self._handle)

    @property
    def ndims(self) -> int:
        """Number of dimensions."""
        return int(sc_tensor_get_ndims(self._handle))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape tuple, e.g. (3, 4) for a 3×4 matrix."""
        n = self.ndims
        raw_dims = sc_tensor_get_dims(self._handle)
        return tuple(int(raw_dims[i]) for i in range(n))

    @property
    def numel(self) -> int:
        """Total number of elements."""
        s = self.shape
        result = 1
        for d in s:
            result *= d
        return result

    @property
    def data(self) -> np.ndarray:
        """
        Copy tensor data from C memory into a NumPy array shaped to .shape.

        The returned array is a fresh copy; mutations do not affect the
        underlying C tensor.
        """
        n = self.numel
        shape = self.shape
        np_dtype = _np_dtype_for(self._dtype)
        ct_type = _ctypes_dtype_for(self._dtype)

        raw_ptr = sc_tensor_get_data(self._handle)
        if not raw_ptr:
            raise RuntimeError("sc_tensor_get_data returned NULL.")

        # Cast void* to a typed pointer and copy into numpy
        typed_ptr = ctypes.cast(raw_ptr, ctypes.POINTER(ct_type))
        flat = np.ctypeslib.as_array(typed_ptr, shape=(n,)).copy()
        return flat.astype(np_dtype).reshape(shape)

    # ------------------------------------------------------------------
    # Eager evaluation helper
    # ------------------------------------------------------------------

    def _evaluate(self) -> None:
        """Trigger immediate computation for this tensor node (CPU path)."""
        ok = sc_tensor_evaluate(self._pool.handle, self._handle)
        if not ok:
            raise RuntimeError(
                f"sc_tensor_evaluate failed for tensor id={self.id}."
            )

    def _make_result(self, handle: int) -> "Tensor":
        """
        Wrap a raw handle, evaluate it eagerly, and return a new Tensor.
        Inherits pool and dtype from self.
        """
        t = Tensor(self._pool, handle, self._dtype, self._requires_grad)
        t._evaluate()
        return t

    # ------------------------------------------------------------------
    # Math operator overloads (eager)
    # ------------------------------------------------------------------

    def __add__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for +: {type(other)}")
        handle = sc_tensor_add(self._pool.handle, self._handle, other._handle)
        return self._make_result(handle)

    def __sub__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for -: {type(other)}")
        handle = sc_tensor_sub(self._pool.handle, self._handle, other._handle)
        return self._make_result(handle)

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiply (@ semantics for 2-D tensors)."""
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for *: {type(other)}")
        handle = sc_tensor_mul(self._pool.handle, self._handle, other._handle)
        return self._make_result(handle)

    # matmul operator alias
    __matmul__ = __mul__

    def add_bias(self, bias: "Tensor") -> "Tensor":
        """Element-wise bias addition: sc_tensor_add_bias(pool, self, bias)."""
        handle = sc_tensor_add_bias(
            self._pool.handle, self._handle, bias._handle
        )
        return self._make_result(handle)

    def relu(self) -> "Tensor":
        """Apply ReLU activation element-wise."""
        handle = sc_tensor_relu(self._pool.handle, self._handle)
        return self._make_result(handle)

    def mean(self) -> "Tensor":
        """Reduce to scalar mean."""
        handle = sc_tensor_mean(self._pool.handle, self._handle)
        return self._make_result(handle)

    def square(self) -> "Tensor":
        """Element-wise square."""
        handle = sc_tensor_square(self._pool.handle, self._handle)
        return self._make_result(handle)

    def transpose(self) -> "Tensor":
        """Return the transpose of this tensor."""
        handle = sc_tensor_transpose(self._pool.handle, self._handle)
        return self._make_result(handle)

    def mul_naive(self, other: "Tensor") -> "Tensor":
        """Naive (non-optimised) matrix multiply, useful for testing."""
        handle = sc_tensor_mul_naive(
            self._pool.handle, self._handle, other._handle
        )
        return self._make_result(handle)

    def mse_loss(self, target: "Tensor") -> "Tensor":
        """Compute MSE loss against *target*."""
        handle = sc_tensor_mse_loss(
            self._pool.handle, self._handle, target._handle
        )
        return self._make_result(handle)

    # ------------------------------------------------------------------
    # GPU evaluation
    # ------------------------------------------------------------------

    def evaluate_gpu(
        self,
        d_a: ctypes.POINTER(ctypes.c_float),
        d_b: ctypes.POINTER(ctypes.c_float),
        d_res: ctypes.POINTER(ctypes.c_float),
    ) -> bool:
        """
        Run this tensor node on the GPU.

        Parameters are raw device-buffer pointers; obtain them through your
        CUDA allocation helper (e.g. cudaMalloc wrappers).
        """
        ok = sc_tensor_evaluate_gpu(
            self._pool.handle, self._handle, d_a, d_b, d_res
        )
        return bool(ok)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def print_data(self) -> None:
        """Delegate to the C-side tensor_print_data for debugging."""
        sc_tensor_print_data(self._handle)

    def numpy(self) -> np.ndarray:
        """Alias for .data — returns a NumPy copy of the tensor contents."""
        return self.data

    def item(self) -> float:
        """Extract a scalar value from a 1-element tensor."""
        if self.numel != 1:
            raise ValueError(
                f".item() requires a scalar tensor; got shape {self.shape}."
            )
        return float(self.data.flat[0])

    def __repr__(self) -> str:
        return (
            f"<Tensor id={self.id} shape={self.shape} "
            f"dtype={self._dtype} requires_grad={self._requires_grad}>"
        )


###############################################################################
# ComputationGraph
###############################################################################

class ComputationGraph:
    """
    Wraps sc_graph_t* and orchestrates forward / backward passes and SGD.

    The preferred entry point is the class-method ``build``, which calls
    sc_build_graph() (the high-level Layer-2 convenience function) to
    walk the DAG from the loss tensor, assign backends, and allocate
    gradient memory in one shot.

    For fine-grained control use the Layer-1 class-methods
    ``create_empty`` + the individual step methods.

    Parameters
    ----------
    handle : int
        Raw sc_graph_t* pointer.
    pool_cpu : MemoryPool
        Host (CPU) memory pool.
    pool_gpu : MemoryPool | None
        Device (GPU) memory pool, or None for CPU-only training.
    """

    def __init__(
        self,
        handle: int,
        pool_cpu: MemoryPool,
        pool_gpu: Optional[MemoryPool] = None,
    ) -> None:
        if not handle:
            raise RuntimeError(
                "ComputationGraph handle is NULL. "
                "sc_build_graph failed (DAG verification error?)."
            )
        self._handle: int = handle
        self._pool_cpu = pool_cpu
        self._pool_gpu = pool_gpu
        self._destroyed = False

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        meta_pool: MemoryPool,
        loss: Tensor,
        pool_gpu: Optional[MemoryPool] = None,
        pool_grad_cpu: Optional[MemoryPool] = None,
        pool_grad_gpu: Optional[MemoryPool] = None,
        backend: int = SCBackend.CPU,
    ) -> "ComputationGraph":
        """
        High-level factory: verifies the DAG, assigns backend, allocates
        gradient memory, and returns a ready-to-train ComputationGraph.

        Parameters
        ----------
        meta_pool : MemoryPool
            Host pool used for graph metadata (node list).
        loss : Tensor
            Root (loss) tensor of the computation graph.
        pool_gpu : MemoryPool | None
            GPU pool.  Required when backend=SCBackend.GPU or HYBRID.
        pool_grad_cpu : MemoryPool | None
            Host pool for gradient storage.  Defaults to meta_pool.
        pool_grad_gpu : MemoryPool | None
            Device pool for gradient storage.  Defaults to pool_gpu.
        backend : int
            SCBackend.CPU / GPU / HYBRID constant.
        """
        gpu_handle  = pool_gpu.handle      if pool_gpu      else None
        gcpu_handle = pool_grad_cpu.handle if pool_grad_cpu else meta_pool.handle
        ggpu_handle = pool_grad_gpu.handle if pool_grad_gpu else (
            pool_gpu.handle if pool_gpu else None
        )

        handle = sc_build_graph(
            meta_pool.handle,
            gpu_handle,
            gcpu_handle,
            ggpu_handle,
            loss.handle,
            backend,
        )
        return cls(handle, meta_pool, pool_gpu)

    @classmethod
    def create_empty(
        cls,
        pool_cpu: MemoryPool,
        pool_gpu: Optional[MemoryPool] = None,
    ) -> "ComputationGraph":
        """
        Layer-1 factory: creates an empty sc_graph_t container.
        Call .verify_dag(), .assign_backend(), .assign_grad_memory()
        manually before training.
        """
        handle = sc_graph_create()
        if not handle:
            raise RuntimeError("sc_graph_create returned NULL.")
        return cls(handle, pool_cpu, pool_gpu)

    # ------------------------------------------------------------------
    # Layer-1 methods (fine-grained control)
    # ------------------------------------------------------------------

    def verify_dag(self, meta_pool: MemoryPool, loss: Tensor) -> bool:
        """Walk the graph from *loss* and populate the internal node list."""
        self._check_alive()
        return bool(sc_verify_dag(meta_pool.handle, loss.handle, self._handle))

    def assign_backend(self, backend: int = SCBackend.CPU) -> None:
        """Assign a backend (CPU/GPU/HYBRID) to each node."""
        self._check_alive()
        gpu_handle = self._pool_gpu.handle if self._pool_gpu else None
        sc_assign_backend(gpu_handle, self._handle, backend)

    def assign_grad_memory(
        self,
        pool_grad_cpu: MemoryPool,
        pool_grad_gpu: Optional[MemoryPool] = None,
    ) -> None:
        """Allocate gradient buffers for all trainable nodes."""
        self._check_alive()
        ggpu = pool_grad_gpu.handle if pool_grad_gpu else None
        sc_assign_grad_memory(pool_grad_cpu.handle, ggpu, self._handle)

    def forward(self) -> bool:
        """Run a forward pass over the graph."""
        self._check_alive()
        gpu_h = self._pool_gpu.handle if self._pool_gpu else None
        return bool(sc_graph_forward(self._pool_cpu.handle, gpu_h, self._handle))

    def gpu_transfer(self) -> None:
        """Transfer node outputs from GPU back to host (autograd step)."""
        self._check_alive()
        sc_autograd_gpu_transfer(self._handle)

    def zero_grad(self) -> None:
        """Initialise (zero) gradients on all nodes before backward."""
        self._check_alive()
        sc_grad_initializer(self._handle)

    def backward(self) -> bool:
        """Run the backward (gradient) pass."""
        self._check_alive()
        return bool(sc_backward(self._handle))

    def sgd_step(self, learning_rate: float) -> None:
        """Apply SGD parameter update with the given learning rate."""
        self._check_alive()
        sc_sgd(self._handle, float(learning_rate))

    def node_to_host(self, node_idx: int) -> bool:
        """Bring node *node_idx* from GPU to host memory."""
        self._check_alive()
        return bool(sc_node_to_host(self._handle, node_idx))

    # ------------------------------------------------------------------
    # Layer-2 methods (full training loop helpers)
    # ------------------------------------------------------------------

    def step(self, learning_rate: float) -> None:
        """
        Run one complete training iteration:
        forward → gpu_transfer → zero_grad → backward → sgd.

        Delegates to sc_graph_step (the C-side convenience wrapper).
        """
        self._check_alive()
        gpu_h = self._pool_gpu.handle if self._pool_gpu else None
        sc_graph_step(
            self._pool_cpu.handle,
            gpu_h,
            self._handle,
            float(learning_rate),
        )

    @property
    def loss(self) -> float:
        """
        Scalar loss value of the last forward pass.
        Returns math.nan if the graph is empty or data is unavailable.
        """
        self._check_alive()
        return sc_graph_get_loss(self._handle)

    @property
    def size(self) -> int:
        """Number of execution nodes in the graph."""
        self._check_alive()
        return sc_graph_size(self._handle)

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    def save(self, path: str, tensors: List[Tensor]) -> bool:
        """
        Save *tensors* to *path*.

        Parameters
        ----------
        path : str   – file path (bytes or str).
        tensors : list[Tensor]
        """
        self._check_alive()
        handles = (ctypes.c_void_p * len(tensors))(
            *(t.handle for t in tensors)
        )
        path_bytes = path.encode() if isinstance(path, str) else path
        return bool(sc_save_model(path_bytes, handles, len(tensors)))

    def load(self, path: str, tensors: List[Tensor]) -> bool:
        """
        Load weights from *path* into *tensors* (must be pre-allocated).

        Parameters
        ----------
        path : str   – file path (bytes or str).
        tensors : list[Tensor] – tensors to receive the loaded weights.
        """
        self._check_alive()
        handles = (ctypes.c_void_p * len(tensors))(
            *(t.handle for t in tensors)
        )
        path_bytes = path.encode() if isinstance(path, str) else path
        return bool(sc_load_model(path_bytes, handles, len(tensors)))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Explicitly free the graph. Safe to call multiple times."""
        if not self._destroyed and self._handle:
            sc_graph_destroy(self._handle)
            self._destroyed = True
            self._handle = None

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("This ComputationGraph has already been destroyed.")

    def __del__(self) -> None:
        self.destroy()

    def __repr__(self) -> str:
        if self._destroyed:
            return "<ComputationGraph [destroyed]>"
        return f"<ComputationGraph size={self.size}>"


###############################################################################
# Module-level convenience: save / load outside a graph object
###############################################################################

def save_model(path: str, tensors: List[Tensor]) -> bool:
    """Save a list of tensors to *path*."""
    handles = (ctypes.c_void_p * len(tensors))(*(t.handle for t in tensors))
    path_b = path.encode() if isinstance(path, str) else path
    return bool(sc_save_model(path_b, handles, len(tensors)))


def load_model(path: str, tensors: List[Tensor]) -> bool:
    """Load weights from *path* into pre-allocated tensors."""
    handles = (ctypes.c_void_p * len(tensors))(*(t.handle for t in tensors))
    path_b = path.encode() if isinstance(path, str) else path
    return bool(sc_load_model(path_b, handles, len(tensors)))
