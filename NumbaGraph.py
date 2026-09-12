import numpy as np
from numpy.typing import NDArray, DTypeLike
import numba as nb


class GraphOperator_Unsafe:
    
    num_nodes: int
    num_edges: int
    row_indices: NDArray
    col_indices: NDArray
    node_indices: NDArray
    _buf_degree: NDArray|None  # New: pre-allocated memory pool to avoid frequent malloc inside Numba

    # ================== Numba accelerated kernels ==================
    
    @staticmethod
    # @nb.njit(parallel=True, cache=True)
    def _numba_P_kernel(edges_2d, rows, cols, out_2d):
        n_batch = edges_2d.shape[0]
        
        for b in nb.prange(n_batch):
            # Optimization: use slice zeroing, Numba compiles this to efficient memset instructions
            out_2d[b, :] = 0.0
            
            for k in range(rows.shape[0]):
                i = rows[k]
                j = cols[k]
                w = edges_2d[b, k]
                out_2d[b, i] += w
                out_2d[b, j] += w
    
    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def _numba_Pstar_kernel(degrees, rows, cols, out_2d):
        n_batch = degrees.shape[0]
        for b in nb.prange(n_batch):
            for k in range(rows.shape[0]):
                out_2d[b, k] = degrees[b, rows[k]] + degrees[b, cols[k]]

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def _numba_A_combined_kernel(edges_2d, rows, cols, out_2d, temp_deg):
        """Modified to accept external buffer to avoid allocating memory on each call"""
        n_batch = edges_2d.shape[0]
        
        for b in nb.prange(n_batch):
            # 1. Zero out the buffer row for current batch (extremely fast)
            temp_deg[b, :] = 0.0
            
            # 2. Scatter-add (accumulate degrees)
            for k in range(rows.shape[0]):
                i = rows[k]
                j = cols[k]
                w = edges_2d[b, k]
                temp_deg[b, i] += w
                temp_deg[b, j] += w
            
            # 3. Gather (lookup output)
            for k in range(rows.shape[0]):
                out_2d[b, k] = temp_deg[b, rows[k]] + temp_deg[b, cols[k]]

    
    def __init__(
        self, 
        num_nodes: int) -> None:
        if num_nodes < 2:
            raise ValueError("NumbaGraph.__init__(): Number of nodes must be at least 2")
        
        self.num_nodes = num_nodes
        self.num_edges = num_nodes * (num_nodes - 1) // 2
        
        # Optimization: use NumPy built-in function to quickly generate upper-triangular indices, faster and cleaner than double loops
        self.row_indices, self.col_indices = np.triu_indices(self.num_nodes, k=1)
        self.row_indices = self.row_indices.astype(np.int32)
        self.col_indices = self.col_indices.astype(np.int32)
        
        self.node_indices = np.arange(self.num_nodes)
        
        # Initialize buffer as empty, delay allocation until first call based on input dtype
        self._buf_degree = None
    
    @property
    def TotalEdges(self) -> int:
        return self.num_edges
    
    @property
    def TotalNodes(self) -> int:
        return self.num_nodes
    
    def prepare_out(
        self, 
        input_tensor: NDArray, 
        out_shape: tuple[int, ...], 
        dtype: DTypeLike|None = None) -> NDArray:
        
        if dtype is None:
            dtype = input_tensor.dtype
        return np.empty(out_shape, dtype=dtype)
    
    def Q(
        self, 
        edges: NDArray, 
        out: NDArray|None=None) -> NDArray:
        
        batch_dims = edges.shape[:-1]
        if out is None:
            out = self.prepare_out(edges, batch_dims + (self.num_nodes, self.num_nodes))
        
        out[..., self.row_indices, self.col_indices] = edges
        out[..., self.col_indices, self.row_indices] = edges
        out[..., self.node_indices, self.node_indices] = 0
        return out
    
    def Qstar(
        self, 
        adj_matrix: NDArray, 
        out: NDArray|None=None) -> NDArray:
        
        batch_dims = adj_matrix.shape[:-2]
        
        if out is None:
            out = self.prepare_out(adj_matrix, batch_dims + (self.num_edges,))
        
        np.add(adj_matrix[..., self.row_indices, self.col_indices], adj_matrix[..., self.col_indices, self.row_indices], out=out)
        return out
    
    def P(
        self, 
        edges: NDArray, 
        out: NDArray|None=None) -> NDArray:
        
        batch_dims = edges.shape[:-1]
        
        if out is None:
            out = self.prepare_out(edges, batch_dims + (self.num_nodes,))
        
        original_shape = edges.shape
        # Optimization: force contiguous memory layout to avoid Numba slowdown or reshape copy
        edges_2d = np.ascontiguousarray(edges.reshape(-1, original_shape[-1]))
        out_2d = out.reshape(edges_2d.shape[0], self.num_nodes)
        
        self._numba_P_kernel(edges_2d, self.row_indices, self.col_indices, out_2d)
        
        out.shape = original_shape[:-1] + (self.num_nodes,)
        return out

    def Pstar(
        self, 
        degree: NDArray, 
        out: NDArray|None=None) -> NDArray:
        
        batch_dims = degree.shape[:-1]
        
        if out is None:
            out = self.prepare_out(degree, batch_dims + (self.num_edges,))
        
        edges_2d = degree.shape
        deg_2d = np.ascontiguousarray(degree.reshape(-1, edges_2d[-1]))
        out_2d = out.reshape(deg_2d.shape[0], self.num_edges)
        
        self._numba_Pstar_kernel(deg_2d, self.row_indices, self.col_indices, out_2d)
        
        out.shape = edges_2d[:-1] + (self.num_edges,)
        return out

    
    def R(
        self, 
        edges: NDArray, 
        out: NDArray|None=None) -> NDArray:
        
        batch_dims = edges.shape[:-1]
        
        if out is None:
            out = self.prepare_out(edges, batch_dims + (self.num_nodes, self.num_nodes))
        
        out[..., self.row_indices, self.col_indices] = -edges
        out[..., self.col_indices, self.row_indices] = -edges
        out[..., self.node_indices, self.node_indices] = -np.sum(out, axis=-1)
        return out
    
    def Rstar(
        self, 
        lap_matrix: NDArray, 
        out: NDArray|None=None) -> NDArray:
        
        batch_dims = lap_matrix.shape[:-2]
        
        if out is None:
            out = self.prepare_out(lap_matrix, batch_dims + (self.num_edges,))
        
        degree = np.diagonal(lap_matrix, axis1=-2, axis2=-1)
        np.add(degree[..., self.row_indices], degree[..., self.col_indices], out=out)
        np.subtract(out, lap_matrix[..., self.row_indices, self.col_indices], out=out)
        np.subtract(out, lap_matrix[..., self.col_indices, self.row_indices], out=out)
        return out
    
    def A(
        self, 
        edges: NDArray, 
        out: NDArray|None=None) -> NDArray:
        
        batch_dims = edges.shape[:-1]
        
        if out is None:
            out = self.prepare_out(edges, batch_dims + (self.num_edges,))

        original_shape = edges.shape
        edges_2d = np.ascontiguousarray(edges.reshape(-1, original_shape[-1]))
        out_2d = out.reshape(edges_2d.shape[0], self.num_edges)
        
        n_batch = edges_2d.shape[0]
        
        # --- Core optimization: reuse pre-allocated degree matrix buffer ---
        if self._buf_degree is None or self._buf_degree.shape[0] < n_batch or self._buf_degree.dtype != edges.dtype:
            # Reallocate on first call or when batch_size increases / dtype changes
            self._buf_degree = np.zeros((n_batch, self.num_nodes), dtype=edges.dtype)
        
        # Slice only the portion needed for current batch to avoid zeroing excess historical data
        active_buffer = self._buf_degree[:n_batch]
        
        # Pass in buffer
        self._numba_A_combined_kernel(edges_2d, self.row_indices, self.col_indices, out_2d, active_buffer)
        
        out.shape = original_shape[:-1] + (self.num_edges,)
        return out
    
    