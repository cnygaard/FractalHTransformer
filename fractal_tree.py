import jax
import jax.numpy as jnp
from typing import List, Optional, NamedTuple, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from ellipsoid_core import Ellipsoid, create_ellipsoid, decompose_ellipsoid


class ParameterGroup(NamedTuple):
    """Group of parameters with their indices"""
    indices: jnp.ndarray  # Indices into the flat parameter array
    size: int
    name: Optional[str] = None


@dataclass
class FractalNode:
    """Node in the fractal ellipsoid tree"""
    ellipsoid: Ellipsoid
    parameter_group: ParameterGroup
    children: Optional[List['FractalNode']] = None
    level: int = 0
    parent: Optional['FractalNode'] = None
    
    # Optimization state
    momentum: Optional[jnp.ndarray] = None
    second_moment: Optional[jnp.ndarray] = None
    gradient_history: Optional[List[jnp.ndarray]] = None
    
    def is_leaf(self) -> bool:
        return self.children is None or len(self.children) == 0
    
    def get_subtree_size(self) -> int:
        if self.is_leaf():
            return self.parameter_group.size
        return sum(child.get_subtree_size() for child in self.children)


class FractalTree:
    """Fractal tree structure for hierarchical parameter decomposition"""
    
    def __init__(self, total_params: int, min_group_size: int = 32, 
                 max_depth: int = 5, branching_factor: int = 6):
        self.total_params = total_params
        self.min_group_size = min_group_size
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.root = None
        self.nodes_by_level = {}
        
    def build_tree(self, param_correlations: Optional[jnp.ndarray] = None):
        """Build the fractal tree structure"""
        all_indices = jnp.arange(self.total_params)
        root_group = ParameterGroup(indices=all_indices, size=self.total_params, name="root")
        root_ellipsoid = create_ellipsoid(self.total_params)
        
        self.root = FractalNode(
            ellipsoid=root_ellipsoid,
            parameter_group=root_group,
            level=0
        )
        
        self._recursive_decompose(self.root, param_correlations)
        self._collect_nodes_by_level()
        
    def _recursive_decompose(self, node: FractalNode, 
                           param_correlations: Optional[jnp.ndarray] = None):
        """Recursively decompose nodes into children"""
        if node.level >= self.max_depth or node.parameter_group.size <= self.min_group_size:
            return
        
        # Partition parameters
        partitions = self._partition_parameters(
            node.parameter_group, 
            param_correlations,
            self.branching_factor
        )
        
        if len(partitions) <= 1:
            return
        
        # Create child nodes
        children = []
        child_ellipsoids = decompose_ellipsoid(node.ellipsoid, len(partitions))
        
        for i, (partition, child_ellipsoid) in enumerate(zip(partitions, child_ellipsoids)):
            child = FractalNode(
                ellipsoid=child_ellipsoid,
                parameter_group=partition,
                level=node.level + 1,
                parent=node
            )
            children.append(child)
            self._recursive_decompose(child, param_correlations)
        
        node.children = children
    
    def _partition_parameters(self, param_group: ParameterGroup,
                            param_correlations: Optional[jnp.ndarray],
                            num_partitions: int) -> List[ParameterGroup]:
        """Partition parameters based on correlations or uniformly"""
        indices = param_group.indices
        
        if param_correlations is not None and len(indices) > num_partitions:
            # Use spectral clustering based on correlations
            partitions = self._spectral_partition(indices, param_correlations, num_partitions)
        else:
            # Uniform partitioning
            partitions = self._uniform_partition(indices, num_partitions)
        
        return [
            ParameterGroup(
                indices=jnp.array(partition),
                size=len(partition),
                name=f"{param_group.name}_child{i}" if param_group.name else f"child{i}"
            )
            for i, partition in enumerate(partitions)
            if len(partition) > 0
        ]
    
    def _spectral_partition(self, indices: jnp.ndarray, 
                          correlations: jnp.ndarray,
                          num_partitions: int) -> List[List[int]]:
        """Partition using spectral clustering on correlation matrix"""
        # Extract submatrix for these indices
        sub_corr = correlations[indices][:, indices]
        
        # Compute affinity matrix
        affinity = jnp.abs(sub_corr)
        
        # Degree matrix
        degrees = jnp.sum(affinity, axis=1)
        D_sqrt_inv = jnp.diag(1.0 / jnp.sqrt(degrees + 1e-8))
        
        # Normalized Laplacian
        L_norm = jnp.eye(len(indices)) - D_sqrt_inv @ affinity @ D_sqrt_inv
        
        # Eigenvectors of Laplacian
        eigenvalues, eigenvectors = jnp.linalg.eigh(L_norm)
        
        # Use k smallest eigenvectors for clustering
        k = min(num_partitions, len(indices))
        embedding = eigenvectors[:, :k]
        
        # Simple k-means clustering on embedding
        partitions = self._kmeans_partition(embedding, k)
        
        # Map back to original indices
        return [[indices[i] for i in partition] for partition in partitions]
    
    def _kmeans_partition(self, points: jnp.ndarray, k: int, max_iters: int = 20) -> List[List[int]]:
        """Simple k-means clustering"""
        n = points.shape[0]
        
        # Initialize centroids
        centroid_indices = np.random.choice(n, k, replace=False)
        centroids = points[centroid_indices]
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = jnp.sum((points[:, None, :] - centroids[None, :, :])**2, axis=2)
            assignments = jnp.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = jnp.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if jnp.any(mask):
                    new_centroids = new_centroids.at[i].set(jnp.mean(points[mask], axis=0))
            
            if jnp.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Group indices by assignment
        partitions = [[] for _ in range(k)]
        for i, assignment in enumerate(assignments):
            partitions[int(assignment)].append(i)
        
        return partitions
    
    def _uniform_partition(self, indices: jnp.ndarray, num_partitions: int) -> List[List[int]]:
        """Uniformly partition indices"""
        n = len(indices)
        partition_size = n // num_partitions
        remainder = n % num_partitions
        
        partitions = []
        start = 0
        
        for i in range(num_partitions):
            size = partition_size + (1 if i < remainder else 0)
            if size > 0:
                partitions.append(indices[start:start + size].tolist())
                start += size
        
        return partitions
    
    def _collect_nodes_by_level(self):
        """Collect all nodes organized by level"""
        self.nodes_by_level = {}
        
        def traverse(node: FractalNode):
            level = node.level
            if level not in self.nodes_by_level:
                self.nodes_by_level[level] = []
            self.nodes_by_level[level].append(node)
            
            if node.children:
                for child in node.children:
                    traverse(child)
        
        if self.root:
            traverse(self.root)
    
    def get_leaf_nodes(self) -> List[FractalNode]:
        """Get all leaf nodes in the tree"""
        leaves = []
        
        def traverse(node: FractalNode):
            if node.is_leaf():
                leaves.append(node)
            elif node.children:
                for child in node.children:
                    traverse(child)
        
        if self.root:
            traverse(self.root)
        return leaves
    
    def get_node_at_path(self, path: List[int]) -> Optional[FractalNode]:
        """Get node at specified path (list of child indices)"""
        node = self.root
        for child_idx in path:
            if node.children and 0 <= child_idx < len(node.children):
                node = node.children[child_idx]
            else:
                return None
        return node
    
    def update_node_ellipsoid(self, node: FractalNode, new_ellipsoid: Ellipsoid):
        """Update a node's ellipsoid and propagate changes"""
        node.ellipsoid = new_ellipsoid
        
        # Update children if they exist
        if node.children:
            child_ellipsoids = decompose_ellipsoid(new_ellipsoid, len(node.children))
            for child, child_ellipsoid in zip(node.children, child_ellipsoids):
                self.update_node_ellipsoid(child, child_ellipsoid)