import math
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

class ProximityProtocol(Protocol):
    """
    Protocol defining the interface for proximity operators.
    """
    def __call__(self, x: NDArray, sigma: NDArray | float, out: NDArray | None = None) -> NDArray:
        """Compute the proximal operator: prox_{f, sigma}(y)."""
        ...
    
    def evaluate(self, x: NDArray) -> NDArray:
        """Evaluate the cost function f(x)."""
        ...
        
    def set_value(self, x: NDArray|None) -> None:
        """Modifier inner state"""
        ...


class LogProx(ProximityProtocol):
    """
    Proximal operator for the negative log-barrier: -lambda * log(x).
    Proximal form: (y + sqrt(y^2 + 4*coeff)) / 2
    """
    
    def __init__(self, lambda_coeff: float | NDArray):
        self.lambda_coeff = lambda_coeff
        
    def __call__(self, y: NDArray, sigma: NDArray | float, out: NDArray | None = None) -> NDArray:
        # Effective coefficient: 2*lambda / sigma
        coeff = 2 * self.lambda_coeff / sigma
        
        # Ensure coeff is broadcastable to y
        if isinstance(coeff, np.ndarray):
            coeff = coeff.reshape([-1] + [1] * (y.ndim - 1))
        
        if out is None:
            out = np.empty_like(y, dtype=y.dtype)
        
        # Compute sqrt(y^2 + coeff)
        np.square(y, out=out)
        out += coeff
        np.sqrt(out, out)
        
        # (y + sqrt(...)) / 2
        out += y
        out /= 2.0
        
        return out
    
    def evaluate(self, x: NDArray) -> NDArray:
        # Cost: -lambda * sum(log(x))
        # Use eps to prevent log(0) -> -inf
        safe_x = np.maximum(x, 1e-10)
        return -self.lambda_coeff * np.sum(np.log(safe_x), axis=-1)
    
    def set_value(self, x: NDArray|None) -> None:
        pass


class L2Prox(ProximityProtocol):
    """
    Proximal operator for the quadratic regularization: lambda * ||x - target||^2.
    Result is a weighted average between y and target.
    """
    
    def __init__(self, lambda_weight: float, target: float | NDArray):
        self.target = target
        self.lambda_weight = lambda_weight
    
    def set_lambda(self, lambda_weight: float):
        self.lambda_weight = lambda_weight
        
    def set_value(self, target: float | NDArray):
        self.target = target
        
    def __call__(self, y: NDArray, sigma: NDArray | float, out: NDArray | None = None) -> NDArray:
        if out is None:
            out = np.empty_like(y, dtype=y.dtype)
        
        # ratio = lambda / (lambda + sigma)
        ratio = self.lambda_weight / sigma
        np.multiply(ratio, self.target, out=out)
        np.add(out, y, out=out)
        
        denom = ratio + 1.0 
        np.divide(out, denom, out=out)
        
        return out
    
    def evaluate(self, x: NDArray) -> NDArray:
        # Cost: lambda * sum((x - target)^2)
        return self.lambda_weight * np.sum(np.square(x - self.target), axis=-1)


class NNSoft(ProximityProtocol):
    
    value: None | float | NDArray
    
    def __init__(self, value: None|float|NDArray=None):
        self.value = value
        
    def set_value(self, value):
        self.value = value
        
    def __call__(self, y, _, out=None):
        if out is None:
            out = np.empty_like(y, dtype=y.dtype)
             
        np.subtract(y, self.value, out=out) # type: ignore
        np.clip(out, 0, None, out=out)
        return out
    
    def evaluate(self, x: NDArray) -> NDArray:
        # 约束条件: 返回0，形状为批次维 x.shape[:-1]
        return np.zeros(x.shape[:-1])


class ProjProx(ProximityProtocol):
    """
    Proximal operator for linear projection onto a vector r.
    Form: lambda * ||x - r'r*x||^2 + 0.5 * sigma * ||x - y||^2
    Projects y towards the subspace spanned by r.
    """
    r_vec: np.ndarray    
    lambda_weight: float | np.ndarray  

    def __init__(self, r_vec: np.ndarray, lambda_weight: float | np.ndarray):
        self.r_vec = r_vec.copy()
        self.r_vec /= np.linalg.norm(self.r_vec)
        self.lambda_weight = lambda_weight
        
    def __call__(self, y: np.ndarray, sigma: float | np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        
        if out is None:
            out = np.empty_like(y, dtype=y.dtype)
        
        # Mixing factor c = lambda / (lambda + sigma)
        c = self.lambda_weight / (self.lambda_weight + sigma)    
        
        if isinstance(c, np.ndarray) and c.ndim == 1:
            c = c[:, np.newaxis]

        # Projection of y onto r: (y' * r) * r
        rTy = np.sum(y * self.r_vec, axis=1, keepdims=True)
        np.multiply(rTy, self.r_vec, out=out)
        
        # Interpolate between y and its projection
        out -= y
        out *= c
        out += y
        
        return out

    def evaluate(self, x: NDArray) -> NDArray:
        return np.zeros(x.shape[:-1])
    
    def set_value(self, x: NDArray|None) -> None:
        pass


class EntropyProx(ProximityProtocol):
    """
    Proximal operator for Entropy regularization: sum(x * log(x)).
    Solved via Newton's method.
    """
    strength: float | NDArray
    
    def __init__(self, strength: float | NDArray):
        self.strength = strength
        self.eps_tolerance = 1e-10
        
    def __call__(self, y: np.ndarray, sigma: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.zeros_like(y)
        else:
            out.fill(0) 
            
        # Coefficient for Newton step
        coeff = self.strength / (2.0 * sigma)
        
        # RHS of the equation f(x)=0 approximation
        rhs = y - coeff
        
        # Initial guess: max(y, coeff)
        # Since the function is minimized around y for large y, but bounded below by 0
        np.maximum(y, coeff, out=out)
        
        # Newton Iteration buffers
        log_x = np.empty_like(out)
        f_val = np.empty_like(out)
        f_prime = np.empty_like(out)
        step_size = np.empty_like(out)
        
        itercount = 0
        while itercount < 20:
            # f(x) = c*log(x) + x - rhs
            np.log(out, out=log_x)
            np.multiply(coeff, log_x, out=f_val)
            np.add(f_val, out, out=f_val)
            np.subtract(f_val, rhs, out=f_val) 
            
            # f'(x) = c/x + 1
            np.divide(coeff, out, out=f_prime)
            np.add(f_prime, 1.0, out=f_prime)
            
            # x_new = x - f/f'
            np.divide(f_val, f_prime, out=step_size)
            np.subtract(out, step_size, out=out)
            
            # Enforce positivity
            np.clip(out, self.eps_tolerance, None, out=out)
            
            itercount += 1
            
            # Check convergence
            step_size = np.abs(step_size, out=step_size)
            if np.max(step_size) < self.eps_tolerance:
                break
            
        return out

    def evaluate(self, x: NDArray) -> NDArray:
        # Cost: strength * sum( x * log(x) )
        safe_x = np.where(x > 0, x, 1.0)
        xlogx = np.where(x > 0, x * np.log(safe_x), 0.0)
        return self.strength * np.sum(xlogx, axis=-1)
    
    def set_value(self, x: NDArray|None) -> None:
        pass

    
