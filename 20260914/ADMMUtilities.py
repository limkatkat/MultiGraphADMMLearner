from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from Proximities import ProximityProtocol

@runtime_checkable
class TransformProtocol(Protocol):
    def __call__(
        self, 
        input_tensor: NDArray, 
        out: NDArray|None=None) -> NDArray: ...


def identity_copy(
    input_tensor: NDArray, 
    out: NDArray|None=None) -> NDArray:
    
    if out is not None:
        np.copyto(dst=out, src=input_tensor)
        return out
    else:
        return input_tensor.copy()
    
# Parameter structure for ADMM
# AX-Z = 0
class SingleConstraint_Unsafe:
    
    z_prev: NDArray   # Z^{k-1}
    z_curr: NDArray   # Z^k
    u: NDArray   # dZ
    primal_residual: NDArray       # eZ = AX - Z
    
    prox_input: NDArray   # alpha*AX + beta*Z^{k-1}
    a_x: NDArray   # AX or alpha*AX
    
    relax_alpha: float    # coefficient for over-relaxation
    relax_beta: float   # beta=1-alpha
    
    a_op: TransformProtocol|None     # A
    at_op: TransformProtocol|None # A'
    prox_op: ProximityProtocol|None
    a_norm: float|None
    
    reduction_axis: tuple[int, ...]
    
    _buf_primal: NDArray   # Same shape as X
    _buf_dual: NDArray # Same shape as Z
    _buf_batch: NDArray # Same shape as batch dimensions
    
    rho: NDArray
    primal_error: NDArray
    dual_error: NDArray
    
    a_op_wrapper: TransformProtocol|Callable
    at_op_wrapper: TransformProtocol|Callable
    
    def __init__(self, 
                 input_tensor: NDArray, 
                 rho: NDArray|float,
                 a_op: TransformProtocol|None,
                 at_op: TransformProtocol|None,
                 prox_op: ProximityProtocol|None,
                 relax_alpha: float,
                 relax_beta: float,
                 reduction_axis: tuple[int, ...],
                 a_norm: float|None=None) -> None:
        
        self.a_x = a_op(input_tensor) if a_op else input_tensor.copy()
        self.z_prev = prox_op(self.a_x, rho) if prox_op else self.a_x.copy()
        self.u = self.a_x - self.z_prev
        self.primal_residual = np.zeros_like(self.z_prev, dtype=input_tensor.dtype)
        self.z_curr = np.zeros_like(self.z_prev, dtype=input_tensor.dtype)
        
        self.prox_input = np.zeros_like(self.a_x, dtype=input_tensor.dtype)
        self.relax_alpha = relax_alpha
        self.relax_beta = relax_beta
        
        self.a_op = a_op
        self.at_op = at_op
        self.prox_op = prox_op
        self.reduction_axis = reduction_axis
        self.a_norm = a_norm
        if isinstance(rho, np.ndarray):
            self.rho = rho
        else:
            batch_dims = input_tensor.shape[:input_tensor.ndim-len(reduction_axis)] + (1,)
            self.rho = rho*np.ones(batch_dims)
        
        self._buf_primal = np.zeros_like(input_tensor)
        self._buf_dual = np.zeros_like(self.a_x)
        self._buf_batch = np.zeros(shape=(input_tensor.shape[:input_tensor.ndim-len(reduction_axis)]))
        
        self.primal_error = np.zeros(shape=(input_tensor.shape[:input_tensor.ndim-len(reduction_axis)]))
        self.dual_error = np.zeros(shape=(input_tensor.shape[:input_tensor.ndim-len(reduction_axis)]))
        
        if self.a_op is None:
            self.a_op_wrapper = identity_copy
        else:
            self.a_op_wrapper = self.a_op
            
        if self.at_op is None:
            self.at_op_wrapper = identity_copy
        else:
            self.at_op_wrapper = self.at_op
            
        
    def swap_duals(self) -> None:
        self.z_prev, self.z_curr = self.z_curr, self.z_prev
        
    def compress_states(self, 
                        keep_indices: NDArray) -> 'SingleConstraint_Unsafe':
        
        self.a_x = compress_data(self.a_x, keep_indices)
        self.prox_input = compress_data(self.prox_input, keep_indices)
        self.z_prev = compress_data(self.z_prev, keep_indices)
        self.z_curr = compress_data(self.z_curr, keep_indices)
        self.u = compress_data(self.u, keep_indices)
        self.primal_residual = compress_data(self.primal_residual, keep_indices)
        self._buf_primal = compress_data(self._buf_primal, keep_indices)
        self._buf_dual = compress_data(self._buf_dual, keep_indices)
        self._buf_batch = compress_data(self._buf_batch, keep_indices)
        self.rho = compress_data(self.rho, keep_indices)
        self.primal_error = compress_data(self.primal_error, keep_indices)
        self.dual_error = compress_data(self.dual_error, keep_indices)
        
        return self
    
    def update(self, 
           input_tensor: NDArray) -> 'SingleConstraint_Unsafe':
        
        # Compute operator value
        self.a_op_wrapper(input_tensor, out=self.a_x)
        np.multiply(self.a_x, self.relax_alpha, self.a_x)
        np.multiply(self.z_prev, self.relax_beta, out=self.prox_input)
        np.add(self.prox_input, self.a_x, out=self.prox_input)
        
        if self.prox_op:
            np.add(self.prox_input, self.u, self._buf_dual)
            self.prox_op(self._buf_dual, self.rho, out=self.z_curr)
        else:
            np.add(self.prox_input, self.u, out=self.z_curr)
        
        np.subtract(self.prox_input, self.z_curr, out=self.primal_residual)
        self.u += self.primal_residual
        
        return self
    
    def adjust_rho(self, 
               enlarge_idx: NDArray, 
               shrink_idx: NDArray, 
               multiplier: float) -> 'SingleConstraint_Unsafe':
        
        self.rho[enlarge_idx] *= multiplier
        np.divide.at(self.u, enlarge_idx, multiplier)
        self.rho[shrink_idx] /= multiplier
        np.multiply.at(self.u, shrink_idx, multiplier)
        
        return self
        
    def update_errors(self) -> tuple[NDArray, NDArray]:
        
        squared_r = np.square(self.primal_residual, out=self._buf_dual)
        pri_err = np.sum(squared_r, axis=self.reduction_axis, out=self.primal_error)
        np.sqrt(pri_err, out=pri_err)
        
        diff_z = np.subtract(self.z_prev, self.z_curr, self._buf_dual)
        squared_diff = np.square(diff_z, out=self._buf_dual)
        d_err = np.sum(squared_diff, axis=self.reduction_axis, out=self.dual_error)
        np.sqrt(d_err, out=d_err)
        
        if self.a_norm:
            d_err = np.multiply(d_err, self.a_norm, out=d_err)
        
        np.multiply(d_err[..., np.newaxis], self.rho, out=d_err[..., np.newaxis])
        
        return pri_err, d_err
            
      
class MultiConstraint_Unsafe:
    z_prev_list: list[NDArray] # Z^{k-1}
    z_curr_list: list[NDArray] # Z^k
    u_list: list[NDArray] # dZ
    
    primal_residual_list: list[NDArray] # AX-Z
    prox_input: NDArray   # alpha*AX + (1-alpha)*Z^{k-1}
    a_x: NDArray   # AX or alpha*AX
    
    relax_alpha: float   # alpha
    relax_beta: float   # beta=1-alpha
    
    a_op: TransformProtocol|None
    at_op: TransformProtocol|None
    prox_ops: tuple[ProximityProtocol|None, ...]
    
    reduction_axis: tuple[int, ...]
    rho: NDArray
    a_norm: float|None
    
    _buf_primal: NDArray    # Same shape as X
    _buf_dual: NDArray      # Same shape as Z
    _buf_batch: NDArray     # Same shape as batch size
    
    primal_error: NDArray   
    dual_error: NDArray
    
    a_op_wrapper: TransformProtocol|Callable
    at_op_wrapper: TransformProtocol|Callable
    
    def __init__(self, 
                 input_tensor: NDArray,
                 rho: NDArray|float,
                 a_op: TransformProtocol|None,
                 at_op: TransformProtocol|None,
                 prox_ops: tuple[ProximityProtocol|None, ...],
                 relax_alpha: float,
                 relax_beta: float,
                 reduction_axis: tuple[int, ...],
                 a_norm: float|None=None) -> None:
        
        self.a_x = a_op(input_tensor) if a_op else input_tensor.copy()
        self.z_prev_list = [fn(self.a_x, rho) if fn else self.a_x.copy() for fn in prox_ops]
        self.z_curr_list = [np.zeros_like(W) for W in self.z_prev_list]
        self.u_list = [self.a_x - W for W in self.z_prev_list]
        self.primal_residual_list = [np.zeros_like(W) for W in self.z_prev_list]
        self.prox_input = np.zeros_like(self.a_x)
        self.relax_alpha = relax_alpha
        self.relax_beta = relax_beta
        self.a_op = a_op
        self.at_op = at_op
        self.prox_ops = prox_ops
        self.reduction_axis = reduction_axis
        if isinstance(rho, np.ndarray):
            self.rho = rho.copy()
        else:
            batch_dims = input_tensor.shape[:input_tensor.ndim-len(reduction_axis)] + (1,)
            self.rho = rho*np.ones(batch_dims)
            
        self.a_norm = a_norm
        
        self._buf_primal = np.zeros_like(input_tensor)
        self._buf_dual = np.zeros_like(self.a_x)
        self._buf_batch = np.zeros(shape=(input_tensor.shape[:input_tensor.ndim-len(reduction_axis)]))
        
        self.primal_error = np.zeros(shape=(input_tensor.shape[:input_tensor.ndim-len(reduction_axis)]))
        self.dual_error = np.zeros(shape=(input_tensor.shape[:input_tensor.ndim-len(reduction_axis)]))
        
        if self.a_op is None:
            self.a_op_wrapper = identity_copy
        else:
            self.a_op_wrapper = self.a_op
            
        if self.at_op is None:
            self.at_op_wrapper = identity_copy
        else:
            self.at_op_wrapper = self.at_op
            
    
    def evaluate(self, 
           x: np.ndarray) -> np.ndarray:
        
        val = self._buf_batch
        val[:] = 0
        for prox in self.prox_ops:
            if prox is not None:
                val += prox.evaluate(x)
        
        return val
        
    
    def swap_duals(self) -> None:
        self.z_prev_list, self.z_curr_list = self.z_curr_list, self.z_prev_list
        
    def compress_states(self, keep_indices: NDArray) -> 'MultiConstraint_Unsafe':
        
        self.a_x = compress_data(self.a_x, keep_indices)
        self._buf_primal = compress_data(self._buf_primal, keep_indices)
        self._buf_dual = compress_data(self._buf_dual, keep_indices)
        self._buf_batch = compress_data(self._buf_batch, keep_indices)
        self.prox_input = compress_data(self.prox_input, keep_indices)
        self.rho = compress_data(self.rho, keep_indices)
        self.primal_error = compress_data(self.primal_error, keep_indices)
        self.dual_error = compress_data(self.dual_error, keep_indices)
        
        self.z_prev_list = [
            compress_data(arr, keep_indices) for arr in self.z_prev_list
        ]
        
        self.z_curr_list = [
            compress_data(arr, keep_indices) for arr in self.z_curr_list
        ]
        
        self.u_list = [
            compress_data(arr, keep_indices) for arr in self.u_list
        ]
        
        self.primal_residual_list = [
            compress_data(arr, keep_indices) for arr in self.primal_residual_list
        ]
        
        return self
    
    def update(self,
           input_tensor: NDArray) -> 'MultiConstraint_Unsafe':
        
        self.a_op_wrapper(input_tensor, out=self.a_x)
        np.multiply(self.a_x, self.relax_alpha, self.a_x)
        
        # Update each parameter
        for z_cur, z_pre, u, resid, prox_fn in zip(self.z_curr_list, self.z_prev_list, self.u_list, self.primal_residual_list, self.prox_ops):
            
            np.multiply(z_pre, self.relax_beta, out=self.prox_input)
            np.add(self.prox_input, self.a_x, out=self.prox_input)
            
            if prox_fn:
                np.add(self.prox_input, u, out=self._buf_dual)
                prox_fn(self._buf_dual, self.rho, out=z_cur)
            else:
                np.add(self.prox_input, u, out=z_cur)
                
            np.subtract(self.prox_input, z_cur, out=resid)
            
            u += resid
        
        return self
            
    def adjust_rho(self, 
               enlarge_idx: NDArray, 
               shrink_idx: NDArray, 
               multiplier: float) -> 'MultiConstraint_Unsafe':
        
        self.rho[enlarge_idx] *= multiplier
        self.rho[shrink_idx] /= multiplier
        
        for u in self.u_list:
            np.divide.at(u, enlarge_idx, multiplier)
            np.multiply.at(u, shrink_idx, multiplier)
            
        return self
            
    def update_errors(self) -> tuple[NDArray, NDArray]:
        
        self.primal_error.fill(0)
        self.dual_error.fill(0)
        
        for z_cur, z_pre, resid in zip(self.z_curr_list, self.z_prev_list, self.primal_residual_list):
            
            squared_r = np.square(resid, out=self._buf_dual)
            pri_err = np.sum(squared_r, axis=self.reduction_axis, out=self._buf_batch)
            self.primal_error += pri_err
            
            diff_z = np.subtract(z_pre, z_cur, out=self._buf_dual)
            squared_diff = np.square(diff_z, out=self._buf_dual)
            d_err = np.sum(squared_diff, axis=self.reduction_axis, out=self._buf_batch)
            
            self.dual_error += d_err
        
        np.sqrt(self.primal_error, out=self.primal_error)
        np.sqrt(self.dual_error, out=self.dual_error)
        
        if self.a_norm:
            self.dual_error *= self.a_norm
        
        np.multiply(self.dual_error[..., np.newaxis], self.rho, out=self.dual_error[..., np.newaxis])
            
        return self.primal_error, self.dual_error

def compress_data(
    datalist: NDArray, 
    indices_to_conserve: NDArray) -> NDArray:
    if indices_to_conserve.dtype == bool:
        idx = np.nonzero(indices_to_conserve)[0]  # Only take non-zero indices of the first dimension
    else:
        idx = indices_to_conserve
    n_keep = len(idx)
    datalist[:n_keep] = datalist[idx]
    return datalist[:n_keep]
