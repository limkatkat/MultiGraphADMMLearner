from typing import Any, Tuple
import numpy as np
from numpy.typing import NDArray
from ADMMUtilities import SingleConstraint_Unsafe, MultiConstraint_Unsafe, compress_data
import NumbaGraph
import Proximities
from StateRecorder import StateRecorder


# min_w 2|w|'s + lam1*g1(Qw) + lam2*g2(Pw) + lam3*g3(Rw), such that w >= 0.
class MultiGraphADMMLearner:
    
    prox_nn: Proximities.ProximityProtocol
    prox_degree: Tuple[Proximities.ProximityProtocol, ...]|None
    prox_adjacency: Tuple[Proximities.ProximityProtocol, ...]|None
    prox_laplacian: Tuple[Proximities.ProximityProtocol, ...]|None
    
    graph_operator: NumbaGraph.GraphOperator_Unsafe
    num_nodes: int
    num_edges: int
    
    lam4: float
    lam5: float
    lam6: float
    
    over_relax_alpha: float
    over_relax_beta: float
    
    max_tolerance: float
    eps_tolerance: float
    max_iterations: int
    
    log_prefix: str|None
    verbose: bool
    
    rho_update_scalar: float
    residual_balance_threshold: float
    
    iteration_recorder: StateRecorder|None
        
    
    def __init__(self, 
                 graph_operator: NumbaGraph.GraphOperator_Unsafe,
                 lam4: float, 
                 lam5: float, 
                 lam6: float, 
                 relax_alpha: float, 
                 nn_prox: Proximities.ProximityProtocol,
                 adjmat_prox: tuple[Proximities.ProximityProtocol, ...]|Proximities.ProximityProtocol|None=None,
                 degree_prox: tuple[Proximities.ProximityProtocol, ...]|Proximities.ProximityProtocol|None=None, 
                 lapmat_prox: tuple[Proximities.ProximityProtocol, ...]|Proximities.ProximityProtocol|None=None,
                 max_iterations: int = 20000, 
                 max_tolerance: float = 1e-8, 
                 log_prefix: str|None=None,
                 verbose: bool=False) -> None:
        
        self.prox_degree = self._standardize_proximity_list(degree_prox)
        self.prox_adjacency = self._standardize_proximity_list(adjmat_prox)
        self.prox_laplacian = self._standardize_proximity_list(lapmat_prox)
        self.prox_nn = nn_prox
        
        # Graph operations, P, Q, R, A
        self.graph_operator = graph_operator
        self.num_nodes = graph_operator.TotalNodes
        self.num_edges = graph_operator.TotalEdges
        
        self.lam4 = lam4
        self.lam5 = lam5
        self.lam6 = lam6
        
        # ADMM parameters
        self.over_relax_alpha = relax_alpha
        self.over_relax_beta = 1 - relax_alpha
        
        self.max_tolerance = max_tolerance
        self.eps_tolerance = 1e-10
        self.max_iterations = max_iterations
        self.log_prefix = log_prefix
        self.verbose = verbose
        
        self.rho_update_scalar = 1.8
        self.residual_balance_threshold = 10.0
        
    @staticmethod 
    def _standardize_proximity_list(prox_param: list|tuple|Any|None):
        
        if prox_param is None:
            return None
        
        if isinstance(prox_param, (tuple, list)):
            return tuple(prox_param)
        
        return (prox_param,)
    
    def init_params(self, 
              edges: NDArray, 
              rho_nn: NDArray,
              rho_adjacent: NDArray|None,
              rho_degree: NDArray|None,
              rho_laplacian: NDArray|None) -> tuple[SingleConstraint_Unsafe, dict[str, MultiConstraint_Unsafe|None]]:
        
        constraint_nn = SingleConstraint_Unsafe(
            input_tensor=edges, 
            rho=rho_nn, 
            a_op=None, 
            at_op=None, 
            prox_op=self.prox_nn, 
            relax_alpha=self.over_relax_alpha, 
            relax_beta=self.over_relax_beta,
            reduction_axis=(-1,),
            a_norm=None)
        
        if self.prox_adjacency and rho_adjacent is not None:
            constraint_adj = MultiConstraint_Unsafe(
                input_tensor=edges, 
                rho=rho_adjacent, 
                a_op=self.graph_operator.Q,  # type: ignore
                at_op=self.graph_operator.Qstar, # type: ignore
                prox_ops=self.prox_adjacency, 
                relax_alpha=self.over_relax_alpha, 
                relax_beta=self.over_relax_beta,
                reduction_axis=(-1,-2),
                a_norm=np.sqrt(2))
        else:
            constraint_adj = None
        
        if self.prox_degree and rho_degree is not None:
            constraint_deg = MultiConstraint_Unsafe(
                input_tensor=edges, 
                rho=rho_degree, 
                a_op=self.graph_operator.P,  # type: ignore
                at_op=self.graph_operator.Pstar, # type: ignore
                prox_ops=self.prox_degree, 
                relax_alpha=self.over_relax_alpha, 
                relax_beta=self.over_relax_beta,
                reduction_axis=(-1,),
                a_norm=np.sqrt(self.num_nodes-2))
        else:
            constraint_deg = None
        
        if self.prox_laplacian and rho_laplacian is not None:
            constraint_lap = MultiConstraint_Unsafe(
                input_tensor=edges, 
                rho=rho_laplacian, 
                a_op=self.graph_operator.R,  # type: ignore
                at_op=self.graph_operator.Rstar, # type: ignore
                prox_ops=self.prox_laplacian, 
                relax_alpha=self.over_relax_alpha, 
                relax_beta=self.over_relax_beta,
                reduction_axis=(-1,-2),
                a_norm=np.sqrt(self.num_nodes))
        else:
            constraint_lap = None
        
        constraint_dict = {'adj': constraint_adj, 'deg': constraint_deg, 'lap': constraint_lap}
        return constraint_nn, constraint_dict
            
    def update_params(self, 
             edges: NDArray, 
             constraint_nn: SingleConstraint_Unsafe, 
             constraint_dict: dict[str, MultiConstraint_Unsafe|None]) -> tuple[SingleConstraint_Unsafe, dict[str, MultiConstraint_Unsafe|None]]:
        
        constraint_nn.update(edges)
        
        for constraint in constraint_dict.values():
            if constraint:
                constraint.update(edges)
        
        return constraint_nn, constraint_dict
    
    def calculate_total_error(self, 
              constraint_nn: SingleConstraint_Unsafe, 
              constraint_dict: dict[str, MultiConstraint_Unsafe|None],
              err_primal_accum: NDArray|None=None,
              err_dual_accum: NDArray|None=None) -> tuple[NDArray, NDArray]:
        
        err_p_nn, err_d_nn = constraint_nn.update_errors()
        
        if err_primal_accum is not None:
            np.copyto(dst=err_primal_accum, src=err_p_nn)
        else:
            err_primal_accum = err_p_nn.copy()
        
        if err_dual_accum is not None:
            np.copyto(dst=err_dual_accum, src=err_d_nn)
        else:
            err_dual_accum = err_d_nn.copy()
        
        for constraint in constraint_dict.values():
            if constraint:
                perr, derr = constraint.update_errors()
                err_primal_accum += perr
                err_dual_accum += derr
                
        return err_primal_accum, err_dual_accum # type: ignore
    
    def automatic_stepsize(self, 
                edges: NDArray, 
                signal_diff: NDArray) -> tuple[NDArray|None, NDArray|None, NDArray|None]:
        
        fidelity_scale = 2.0 * np.maximum(np.mean(np.abs(signal_diff), axis=-1), self.eps_tolerance)

        if self.prox_adjacency:
            op_mag = np.mean(np.abs(self.graph_operator.Q(edges)), axis=(-1, -2))*len(self.prox_adjacency)
            # op_mag = np.where(op_mag>0, op_mag, 0)
            np.clip(op_mag, 0, None, out=op_mag)
            rho_adj = fidelity_scale / np.maximum(op_mag, self.eps_tolerance)
            rho_adj = rho_adj[..., np.newaxis]
        else:
            rho_adj = None
            
        if self.prox_degree:
            op_mag = np.mean(np.abs(self.graph_operator.P(edges)), axis=-1)*len(self.prox_degree)
            np.clip(op_mag, 0, None, out=op_mag)
            rho_deg = fidelity_scale / np.maximum(op_mag, self.eps_tolerance)
            rho_deg = rho_deg[..., np.newaxis]
        else:
            rho_deg = None
            
        if self.prox_laplacian:
            op_mag = np.mean(np.abs(self.graph_operator.R(edges)), axis=(-1, -2))*len(self.prox_laplacian)
            np.clip(op_mag, 0, None, out=op_mag)
            rho_lap = fidelity_scale / np.maximum(op_mag, self.eps_tolerance)
            rho_lap = rho_lap[..., np.newaxis]
        else:
            rho_lap = None
        
        return rho_adj, rho_deg, rho_lap
    

    # Edge vector group: batch_size x num_graphs x num_edges
    # Signal difference vector group: batch_size x num_graphs x num_edges
    # NOTE: Because a relatively large dynamic adjustment interval (100) is used, 
    # the coarse-to-fine mechanism is not obvious between short iteration process.
    def __call__(self, 
                 edges: NDArray, 
                 signal_diff: NDArray,
                 expected_edge_ratio: NDArray|None=None,
                 iteration_record: StateRecorder|None=None,
                 show_error_interval=-1) -> Tuple[NDArray, NDArray, StateRecorder|None]:
        
        batch_shape = signal_diff.shape[:-1]
        num_graphs = np.prod(batch_shape).item()
        
        edges = edges.reshape((num_graphs, -1))
        signal_diff = signal_diff.reshape((num_graphs, -1))
        
        rho_nn = np.sqrt(self.graph_operator.num_nodes) * np.maximum(np.mean(np.abs(signal_diff), axis=-1), self.eps_tolerance)
        if expected_edge_ratio is not None:
            rho_nn *= 1-expected_edge_ratio
        rho_nn = rho_nn[..., np.newaxis]
        
        rho_adj, rho_deg, rho_lap = self.automatic_stepsize(edges, signal_diff)
        
        # constants settings
        count_adj = float(len(self.prox_adjacency)) if self.prox_adjacency else 0.0
        count_deg = float(len(self.prox_degree)) if self.prox_degree else 0.0
        count_lap = float(len(self.prox_laplacian)) if self.prox_laplacian else 0.0
        
        tau0 = rho_nn + 2*self.lam6
        mu1 = self.lam5
        if rho_adj is not None:
            tau0 += 2*count_adj*rho_adj
        if rho_deg is not None:
            mu1 += count_deg*rho_deg
        if rho_lap is not None:
            tau0 += 2*count_lap*rho_lap
            mu1 += count_lap*rho_lap
        
        np.divide(1.0, tau0, out=tau0)
        mu1 *= tau0
        tau1  = 1.0 / (self.num_nodes-2.0 + 1.0/mu1)
        tau2  = 1.0 / (self.num_nodes*2.0-2.0 + 1.0/mu1)
        # delta = 4.0 / self.num_nodes * (tau1-tau2)
        delta = 4.0*tau1*tau2
        
        signal_diff_scaled = (signal_diff + self.lam4) / rho_nn
        self.prox_nn.set_value(signal_diff_scaled) # type: ignore
        
        # Initialization
        final_weights = np.zeros_like(edges, dtype=edges.dtype)
        constraint_nn, constraint_dict = self.init_params(edges, rho_nn, rho_adj, rho_deg, rho_lap)
        
        edges = self._solve_edges(tau0, tau1, delta, constraint_nn, constraint_dict, edges) # type: ignore
        constraint_nn, constraint_dict = self.update_params(edges, constraint_nn, constraint_dict)
        
        err_primal_total, err_dual_total = self.calculate_total_error(constraint_nn, constraint_dict)
        
        unconverged_mask = np.logical_or(err_primal_total >= self.max_tolerance, err_dual_total >= self.max_tolerance)
        active_indices = np.arange(num_graphs, dtype=np.intp)
            
        current_iteration = 0
        while current_iteration < self.max_iterations and np.any(unconverged_mask):
            
            current_iteration += 1
            
            # Swap variables for next iteration
            constraint_nn.swap_duals()
            for constraint in constraint_dict.values():
                if constraint:
                    constraint.swap_duals()
            
            edges = self._solve_edges(tau0, tau1, delta, constraint_nn, constraint_dict, edges) # type: ignore
            
            constraint_nn, constraint_dict = self.update_params(edges, constraint_nn, constraint_dict)
            err_primal_total, err_dual_total = self.calculate_total_error(constraint_nn, constraint_dict, err_primal_accum=err_primal_total, err_dual_accum=err_dual_total)
            np.logical_or(err_primal_total >= self.max_tolerance, err_dual_total >= self.max_tolerance, out=unconverged_mask)
            if show_error_interval > 0 and current_iteration % show_error_interval == 0:
                print ("primal error [%d] = %s" % (current_iteration, err_primal_total))
                print ("dual error [%d] = %s" % (current_iteration, err_dual_total))
            
            if iteration_record and num_graphs == 1 and iteration_record.should_record(current_iteration):
                
                # Use the actual solution from current iteration, instead of all-zero non-negative constraint variable group
                
                current_edges = constraint_nn.z_curr
                
                # Note: add axis=-1 to ensure result is a vector of shape (total_graphs,) instead of scalar
                obj_value = 2*np.sum(current_edges*signal_diff, axis=-1)
                obj_value += 2*self.lam4*np.sum(current_edges, axis=-1)
                obj_value += self.lam5*np.sum(current_edges*self.graph_operator.A(current_edges), axis=-1)
                obj_value += 2*self.lam6*np.sum(current_edges*current_edges, axis=-1)
                
                # Non-negative constraint does not produce value, so no need to record
                for c_type, constraint in constraint_dict.items():
                    if constraint is not None:
                        if c_type == 'adj':
                            obj_value += constraint.evaluate(self.graph_operator.Q(current_edges))
                        elif c_type == 'deg':
                            obj_value += constraint.evaluate(self.graph_operator.P(current_edges))
                        elif c_type == 'lap':
                            obj_value += constraint.evaluate(self.graph_operator.R(current_edges))
                        
                iteration_record.record(
                    current_iteration, 
                    (constraint_nn.z_prev.copy(), err_primal_total.copy(), err_dual_total.copy(), obj_value.copy()))

            
            if not np.all(unconverged_mask):
                if self.verbose:
                    
                    idx_max_p = np.argmax(err_primal_total)
                    idx_max_d = np.argmax(err_dual_total)
                    print("Current iteration = %d, primal error = %f, %f, dual error = %f, %f, remaining unsolved graphs = %d" % (current_iteration, err_primal_total[idx_max_p], err_dual_total[idx_max_p], err_primal_total[idx_max_d], err_dual_total[idx_max_d], np.count_nonzero(unconverged_mask)))
                
                final_weights[active_indices[np.logical_not(unconverged_mask)]] = constraint_nn.z_curr[np.logical_not(unconverged_mask)]
                
                active_indices = active_indices[unconverged_mask]
                if len(active_indices) == 0:
                    break
                
                edges = compress_data(edges, unconverged_mask)
                signal_diff_scaled = compress_data(signal_diff_scaled, unconverged_mask)
                signal_diff = compress_data(signal_diff, unconverged_mask)
                tau0 = compress_data(tau0, unconverged_mask)
                mu1 = compress_data(mu1, unconverged_mask)
                tau1 = compress_data(tau1, unconverged_mask)
                tau2 = compress_data(tau2, unconverged_mask)
                delta = compress_data(delta, unconverged_mask)
                err_primal_total = compress_data(err_primal_total, unconverged_mask)
                err_dual_total = compress_data(err_dual_total, unconverged_mask)
                self.prox_nn.set_value(signal_diff_scaled) # type: ignore
                
                constraint_nn.compress_states(unconverged_mask)
                for constraint in constraint_dict.values():
                    if constraint:
                        constraint.compress_states(unconverged_mask)
                        
                unconverged_mask = np.ones(shape=(len(active_indices), ), dtype=bool)
        
            if current_iteration % 100 == 0:
                
                # Modifying the step size of non-negative constraint will cause severe oscillation
                # Strongly not recommended to modify!
                step_modified = False
                for constraint in constraint_dict.values():
                    if constraint is None:
                        continue
                    
                    p_err = constraint.primal_error
                    d_err = constraint.dual_error
                
                    enlarge_idx = np.where(p_err > self.residual_balance_threshold*d_err)[0]
                    shrink_idx = np.where(self.residual_balance_threshold*p_err < d_err)[0]
                
                    if len(enlarge_idx)==0 and len(shrink_idx)==0:
                        continue
                    
                    step_modified = True
                    constraint.adjust_rho(enlarge_idx, shrink_idx, self.rho_update_scalar)
                
                if step_modified:
                    tau0 = constraint_nn.rho + 2*self.lam6
                    mu1 = self.lam5
                    if constraint_dict['adj'] is not None:
                        tau0 += 2*count_adj*constraint_dict['adj'].rho
                    
                    if constraint_dict['deg'] is not None:
                        mu1 += count_deg*constraint_dict['deg'].rho
                        
                    if constraint_dict['lap'] is not None:
                        tau0 += 2*count_lap*constraint_dict['lap'].rho
                        mu1 += count_lap*constraint_dict['lap'].rho
                    
                    np.divide(1.0, tau0, out=tau0)
                    mu1 *= tau0
                    tau1  = 1.0 / (self.num_nodes-2.0 + 1.0/mu1)
                    tau2  = 1.0 / (self.num_nodes*2.0-2.0 + 1.0/mu1)
                    # delta = 4.0 / self.num_nodes * (tau1-tau2)
                    delta = 4.0*tau1*tau2
                    
                    signal_diff_scaled = (signal_diff + self.lam4) / constraint_nn.rho
                    self.prox_nn.set_value(signal_diff_scaled) # type: ignore
                        
        # Collect final results for remaining unconverged samples
        if active_indices.size > 0:
            final_weights[active_indices] = constraint_nn.z_curr
            
            if iteration_record is not None:
                if num_graphs == 1: 
                    if iteration_record.should_record(current_iteration):
                        # Similarly replace with actual current solution
                        current_edges = constraint_nn.z_curr
                        
                        obj_value = 2*np.sum(current_edges*signal_diff, axis=-1)
                        obj_value += 2*self.lam4*np.sum(current_edges, axis=-1)
                        obj_value += self.lam5*np.sum(current_edges*self.graph_operator.A(current_edges), axis=-1)
                        obj_value += 2*self.lam6*np.sum(current_edges*current_edges, axis=-1)
                        
                        for c_type, constraint in constraint_dict.items():
                            if constraint is not None:
                                if c_type == 'adj':
                                    obj_value += constraint.evaluate(self.graph_operator.Q(current_edges))
                                elif c_type == 'deg':
                                    obj_value += constraint.evaluate(self.graph_operator.P(current_edges))
                                elif c_type == 'lap':
                                    obj_value += constraint.evaluate(self.graph_operator.R(current_edges))
                        
                        print(f'Current iteration={current_iteration}, objective value={obj_value}')
                                
                        iteration_record.record(current_iteration, (constraint_nn.z_prev.copy(), err_primal_total.copy(), err_dual_total.copy(), obj_value.copy()))


        
        if active_indices.size == 0:
            print(f"All converged after {current_iteration} iterations")
        else:
            print(f"Reached max iterations {self.max_iterations}, {active_indices.size} samples remain unconverged")
        
        final_weights = final_weights.reshape(batch_shape+(-1,)) # type: ignore
        return final_weights, active_indices, iteration_record
    
    
    def _add_single_constraint(
        self, 
        constraint: SingleConstraint_Unsafe) -> NDArray:
        
        np.subtract(constraint.z_prev, constraint.u, out=constraint._buf_dual)
        hvec = constraint.at_op_wrapper(constraint._buf_dual, out=constraint._buf_primal)
        np.multiply(hvec, constraint.rho, out=hvec)
        return hvec
        
    def _add_multi_constraints(
        self, 
        constraint: MultiConstraint_Unsafe, 
        grad_term: NDArray) -> NDArray:
        
        for z_pre, u in zip(constraint.z_prev_list, constraint.u_list):
            
            np.subtract(z_pre, u, out=constraint._buf_dual)
            constraint.at_op_wrapper(constraint._buf_dual, out=constraint._buf_primal)
            np.multiply(constraint._buf_primal, constraint.rho, out=constraint._buf_primal)
            grad_term += constraint._buf_primal
                
        return grad_term
            
    def _solve_edges(self, 
               tau0: NDArray, 
               tau1: NDArray, 
               delta: NDArray, 
               constraint_nn: SingleConstraint_Unsafe, 
               constraint_dict: dict[str, MultiConstraint_Unsafe|None],
               edges: NDArray) -> NDArray:
        
        # Compute hvec
        # hvec = tau0*(Q'W + P'd + R'L + v)
        grad_term = self._add_single_constraint(constraint_nn)    # Occupies non-negative constraint parameter structure.primary variable buffer
        for constraint in constraint_dict.values():
            if constraint:
                grad_term = self._add_multi_constraints(constraint, grad_term=grad_term)
        grad_term *= tau0
        
        # Compute h - delta*(e'h)e
        sum_grad = np.sum(grad_term, axis=constraint_nn.reduction_axis, out=constraint_nn._buf_batch) # Occupies non-negative constraint parameter structure.batch buffer
        sum_grad = sum_grad[..., np.newaxis]
        sum_grad *= delta
        
        # Compute tau1*(Ah)
        self.graph_operator.A(grad_term, out=edges)
        np.multiply(edges, tau1, out=edges)
        
        grad_term += sum_grad
        np.subtract(grad_term, edges, out=edges)
        
        return edges
    
    