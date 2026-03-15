import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple, Dict, List
from functools import partial

from fractal_tree import FractalNode, FractalTree
from ellipsoid_core import Ellipsoid


class AdamMoments(NamedTuple):
    """Adam optimizer moments"""
    m: jnp.ndarray  # First moment (momentum)
    v: jnp.ndarray  # Second moment (variance)
    m_hat: Optional[jnp.ndarray] = None  # Bias-corrected first moment
    v_hat: Optional[jnp.ndarray] = None  # Bias-corrected second moment


class HierarchicalMoments(NamedTuple):
    """Hierarchical moments for fractal structure"""
    global_moments: AdamMoments
    node_moments: Dict[int, AdamMoments]  # Moments per node ID
    aggregation_weights: jnp.ndarray
    
    
@jax.jit
def update_adam_moments(
    gradient: jnp.ndarray,
    moments: AdamMoments,
    beta1: float = 0.9,
    beta2: float = 0.999,
    step: int = 1
) -> AdamMoments:
    """Update Adam moments with new gradient"""
    # Update biased moments
    m = beta1 * moments.m + (1 - beta1) * gradient
    v = beta2 * moments.v + (1 - beta2) * gradient**2
    
    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    
    return AdamMoments(m=m, v=v, m_hat=m_hat, v_hat=v_hat)


@jax.jit
def integrate_moments_with_ellipsoid(
    moments: AdamMoments,
    ellipsoid: Ellipsoid,
    integration_factor: float = 0.1
) -> Ellipsoid:
    """Integrate Adam second moments with ellipsoid shape"""
    # Use second moment to update ellipsoid shape
    # A_t = λA_{t-1} + (1-λ)diag(v_t)
    
    # Convert second moment to diagonal matrix
    v_diag = jnp.diag(jnp.sqrt(moments.v_hat + 1e-8))
    
    # Blend with current shape matrix
    new_shape = (1 - integration_factor) * ellipsoid.shape_matrix + integration_factor * v_diag
    
    # Ensure positive definiteness
    eigenvalues, eigenvectors = jnp.linalg.eigh(new_shape)
    eigenvalues = jnp.maximum(eigenvalues, 1e-6)
    new_shape = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    
    return ellipsoid._replace(
        shape_matrix=new_shape,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors
    )


def aggregate_child_moments(
    child_moments: List[AdamMoments],
    weights: Optional[jnp.ndarray] = None
) -> AdamMoments:
    """Aggregate moments from child nodes (bottom-up)"""
    if not child_moments:
        return None
    
    if weights is None:
        weights = jnp.ones(len(child_moments)) / len(child_moments)
    
    # Weighted average of moments
    m_aggregated = jnp.zeros_like(child_moments[0].m)
    v_aggregated = jnp.zeros_like(child_moments[0].v)
    
    for i, (child_moment, weight) in enumerate(zip(child_moments, weights)):
        m_aggregated += weight * child_moment.m
        v_aggregated += weight * child_moment.v
    
    # For second moments, use weighted RMS instead of average
    v_aggregated = jnp.sqrt(v_aggregated)
    
    return AdamMoments(m=m_aggregated, v=v_aggregated)


@jax.jit
def refine_moments_with_parent(
    local_moments: AdamMoments,
    parent_moments: AdamMoments,
    local_gradient: jnp.ndarray,
    alpha: float = 0.3
) -> AdamMoments:
    """Refine local moments using parent information (top-down)"""
    # Blend parent and local information
    m_refined = alpha * parent_moments.m + (1 - alpha) * local_moments.m
    v_refined = alpha * parent_moments.v + (1 - alpha) * local_moments.v
    
    # Ensure v_refined captures local gradient variance
    local_variance = local_gradient**2
    v_refined = jnp.maximum(v_refined, 0.1 * local_variance)
    
    return AdamMoments(m=m_refined, v=v_refined)


class HierarchicalAdamState(NamedTuple):
    """State for hierarchical Adam optimization"""
    tree: FractalTree
    moments_by_node: Dict[int, AdamMoments]
    global_step: int
    beta1: float
    beta2: float
    epsilon: float
    
    
def initialize_hierarchical_adam_state(
    tree: FractalTree,
    param_shape: Tuple[int, ...],
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
) -> HierarchicalAdamState:
    """Initialize hierarchical Adam state for fractal tree"""
    moments_by_node = {}
    
    def initialize_node_moments(node: FractalNode):
        # Initialize moments for this node's parameters
        node_size = node.parameter_group.size
        moments = AdamMoments(
            m=jnp.zeros(node_size),
            v=jnp.zeros(node_size)
        )
        moments_by_node[id(node)] = moments
        
        # Recursively initialize children
        if node.children:
            for child in node.children:
                initialize_node_moments(child)
    
    initialize_node_moments(tree.root)
    
    return HierarchicalAdamState(
        tree=tree,
        moments_by_node=moments_by_node,
        global_step=0,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )


def update_hierarchical_moments(
    state: HierarchicalAdamState,
    gradients: jnp.ndarray,
    phase: str = "fractal"
) -> Tuple[HierarchicalAdamState, jnp.ndarray]:
    """Update moments hierarchically and return preconditioned gradients"""
    new_moments = {}
    preconditioned_grads = jnp.zeros_like(gradients)
    
    if phase == "stochastic":
        # In stochastic phase, only update global moments
        global_moments = state.moments_by_node[id(state.tree.root)]
        updated_moments = update_adam_moments(
            gradients,
            global_moments,
            state.beta1,
            state.beta2,
            state.global_step
        )
        new_moments[id(state.tree.root)] = updated_moments
        
        # Apply global preconditioning
        preconditioned_grads = updated_moments.m_hat / (jnp.sqrt(updated_moments.v_hat) + state.epsilon)
        
    else:  # fractal or hybrid phase
        # Bottom-up: Update leaf nodes first
        for node in state.tree.get_leaf_nodes():
            node_indices = node.parameter_group.indices
            node_grads = gradients[node_indices]
            
            node_moments = state.moments_by_node[id(node)]
            updated_moments = update_adam_moments(
                node_grads,
                node_moments,
                state.beta1,
                state.beta2,
                state.global_step
            )
            new_moments[id(node)] = updated_moments
            
            # Apply local preconditioning
            preconditioned_grads = preconditioned_grads.at[node_indices].set(
                updated_moments.m_hat / (jnp.sqrt(updated_moments.v_hat) + state.epsilon)
            )
        
        # Bottom-up aggregation for internal nodes
        def aggregate_node(node: FractalNode):
            if node.is_leaf():
                return new_moments.get(id(node), state.moments_by_node[id(node)])
            
            # Aggregate from children
            child_moments = [aggregate_node(child) for child in node.children]
            aggregated = aggregate_child_moments(child_moments)
            
            # Update with node's gradient slice
            node_indices = node.parameter_group.indices
            node_grads = gradients[node_indices]
            
            updated_moments = update_adam_moments(
                node_grads,
                aggregated,
                state.beta1,
                state.beta2,
                state.global_step
            )
            new_moments[id(node)] = updated_moments
            
            return updated_moments
        
        # Start aggregation from root
        aggregate_node(state.tree.root)
        
        # Top-down refinement
        def refine_node(node: FractalNode, parent_moments: Optional[AdamMoments] = None):
            node_moments = new_moments[id(node)]
            
            if parent_moments is not None:
                node_indices = node.parameter_group.indices
                node_grads = gradients[node_indices]
                
                refined_moments = refine_moments_with_parent(
                    node_moments,
                    parent_moments,
                    node_grads,
                    alpha=0.3
                )
                new_moments[id(node)] = refined_moments
                
                # Update preconditioned gradients
                preconditioned_grads = preconditioned_grads.at[node_indices].set(
                    refined_moments.m_hat / (jnp.sqrt(refined_moments.v_hat) + state.epsilon)
                )
            
            # Recursively refine children
            if node.children:
                for child in node.children:
                    refine_node(child, node_moments)
        
        refine_node(state.tree.root)
    
    # Create new state
    new_state = state._replace(
        moments_by_node=new_moments,
        global_step=state.global_step + 1
    )
    
    return new_state, preconditioned_grads


@jax.jit
def compute_adaptive_learning_rate(
    base_lr: float,
    moments: AdamMoments,
    ellipsoid: Ellipsoid,
    phase: str
) -> jnp.ndarray:
    """Compute adaptive learning rate based on moments and ellipsoid geometry"""
    # Base adaptive rate from Adam
    adam_lr = base_lr / (jnp.sqrt(moments.v_hat) + 1e-8)
    
    # Geometric scaling from ellipsoid
    if ellipsoid.eigenvalues is not None:
        # Scale by inverse eigenvalues for better conditioning
        geometric_scaling = 1.0 / jnp.sqrt(ellipsoid.eigenvalues + 1e-8)
        geometric_scaling = geometric_scaling / jnp.mean(geometric_scaling)  # Normalize
    else:
        geometric_scaling = 1.0
    
    # Phase-dependent scaling
    if phase == "stochastic":
        phase_scale = 1.0
    elif phase == "fractal":
        phase_scale = 0.5  # More conservative in fractal mode
    else:  # hybrid
        phase_scale = 0.75
    
    return adam_lr * geometric_scaling * phase_scale