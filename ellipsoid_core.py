import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple
import numpy as np


class Ellipsoid(NamedTuple):
    """Represents an ellipsoid with shape matrix A and center a"""
    shape_matrix: jnp.ndarray  # Shape matrix A (symmetric positive definite)
    center: jnp.ndarray        # Center point a
    eigenvalues: Optional[jnp.ndarray] = None
    eigenvectors: Optional[jnp.ndarray] = None
    volume: Optional[float] = None
    
    
def create_ellipsoid(dim: int, center: Optional[jnp.ndarray] = None, 
                    scale: float = 1.0) -> Ellipsoid:
    """Create an initial spherical ellipsoid"""
    if center is None:
        center = jnp.zeros(dim)
    
    shape_matrix = scale * jnp.eye(dim)
    eigenvalues = scale * jnp.ones(dim)
    eigenvectors = jnp.eye(dim)
    volume = compute_ellipsoid_volume(shape_matrix)
    
    return Ellipsoid(shape_matrix, center, eigenvalues, eigenvectors, volume)


@jax.jit
def compute_ellipsoid_volume(shape_matrix: jnp.ndarray) -> float:
    """Compute the volume of an ellipsoid from its shape matrix"""
    det = jnp.linalg.det(shape_matrix)
    dim = shape_matrix.shape[0]
    
    # Volume = (π^(n/2) / Γ(n/2 + 1)) * sqrt(det(A))
    # For computational stability, we use log-space
    log_volume = (dim/2) * jnp.log(jnp.pi) - jax.scipy.special.gammaln(dim/2 + 1) + 0.5 * jnp.log(det)
    return jnp.exp(log_volume)


@jax.jit
def mahalanobis_distance(point: jnp.ndarray, ellipsoid: Ellipsoid) -> float:
    """Compute Mahalanobis distance from point to ellipsoid center"""
    diff = point - ellipsoid.center[:len(point)]
    # Use shape matrix inverse for distance computation
    # For numerical stability, use solve instead of explicit inverse
    dist_squared = jnp.dot(diff, jnp.linalg.solve(ellipsoid.shape_matrix[:len(point), :len(point)], diff))
    return jnp.sqrt(dist_squared)


@jax.jit
def update_ellipsoid_eigendecomposition(ellipsoid: Ellipsoid) -> Ellipsoid:
    """Update eigenvalues and eigenvectors of the ellipsoid"""
    eigenvalues, eigenvectors = jnp.linalg.eigh(ellipsoid.shape_matrix)
    volume = compute_ellipsoid_volume(ellipsoid.shape_matrix)
    
    return ellipsoid._replace(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        volume=volume
    )


@jax.jit
def ellipsoid_mahalanobis_distance(point: jnp.ndarray, ellipsoid: Ellipsoid) -> float:
    """Compute Mahalanobis distance from point to ellipsoid center"""
    diff = point - ellipsoid.center
    inv_shape = jnp.linalg.inv(ellipsoid.shape_matrix)
    return jnp.sqrt(diff.T @ inv_shape @ diff)


@jax.jit
def is_point_inside_ellipsoid(point: jnp.ndarray, ellipsoid: Ellipsoid) -> bool:
    """Check if a point is inside the ellipsoid"""
    return ellipsoid_mahalanobis_distance(point, ellipsoid) <= 1.0


@jax.jit
def project_gradient_to_ellipsoid(gradient: jnp.ndarray, ellipsoid: Ellipsoid) -> jnp.ndarray:
    """Project gradient according to ellipsoid geometry (Riemannian gradient)"""
    # grad_R = A * grad_E
    return ellipsoid.shape_matrix @ gradient


@jax.jit
def adapt_ellipsoid_shape(ellipsoid: Ellipsoid, gradient: jnp.ndarray, 
                         learning_rate: float = 0.01) -> Ellipsoid:
    """Adapt ellipsoid shape based on gradient information"""
    # Project gradient onto eigenvector basis
    if ellipsoid.eigenvectors is None:
        ellipsoid = update_ellipsoid_eigendecomposition(ellipsoid)
    
    grad_eigen = ellipsoid.eigenvectors.T @ gradient
    
    # Update eigenvalues based on gradient magnitude in each direction
    # λᵢ(t+1) = λᵢ(t) + η * (gᵢ² / λᵢ(t) - 1)
    new_eigenvalues = ellipsoid.eigenvalues + learning_rate * (
        grad_eigen**2 / (ellipsoid.eigenvalues + 1e-8) - 1
    )
    
    # Ensure positive eigenvalues
    new_eigenvalues = jnp.maximum(new_eigenvalues, 1e-6)
    
    # Reconstruct shape matrix
    new_shape_matrix = ellipsoid.eigenvectors @ jnp.diag(new_eigenvalues) @ ellipsoid.eigenvectors.T
    
    return Ellipsoid(
        shape_matrix=new_shape_matrix,
        center=ellipsoid.center,
        eigenvalues=new_eigenvalues,
        eigenvectors=ellipsoid.eigenvectors,
        volume=compute_ellipsoid_volume(new_shape_matrix)
    )


@jax.jit
def merge_ellipsoids(ellipsoid1: Ellipsoid, ellipsoid2: Ellipsoid, 
                    weight1: float = 0.5, weight2: float = 0.5) -> Ellipsoid:
    """Merge two ellipsoids with given weights"""
    # Weighted average of centers
    new_center = weight1 * ellipsoid1.center + weight2 * ellipsoid2.center
    
    # Weighted harmonic mean of shape matrices (preserves positive definiteness)
    inv1 = jnp.linalg.inv(ellipsoid1.shape_matrix)
    inv2 = jnp.linalg.inv(ellipsoid2.shape_matrix)
    new_inv = weight1 * inv1 + weight2 * inv2
    new_shape_matrix = jnp.linalg.inv(new_inv)
    
    return Ellipsoid(
        shape_matrix=new_shape_matrix,
        center=new_center,
        eigenvalues=None,
        eigenvectors=None,
        volume=None
    )


def decompose_ellipsoid(ellipsoid: Ellipsoid, num_children: int = 6) -> list[Ellipsoid]:
    """Decompose an ellipsoid into smaller child ellipsoids"""
    if ellipsoid.eigenvectors is None:
        ellipsoid = update_ellipsoid_eigendecomposition(ellipsoid)
    
    children = []
    scale_factor = 0.5
    offset_factor = 0.7
    
    # Create child ellipsoids along principal axes
    for i in range(min(num_children // 2, ellipsoid.eigenvalues.shape[0])):
        # Positive direction
        offset = offset_factor * jnp.sqrt(ellipsoid.eigenvalues[i]) * ellipsoid.eigenvectors[:, i]
        child_center = ellipsoid.center + offset
        child_shape = scale_factor * ellipsoid.shape_matrix
        children.append(Ellipsoid(child_shape, child_center))
        
        # Negative direction
        child_center = ellipsoid.center - offset
        children.append(Ellipsoid(child_shape, child_center))
    
    return children