"""
Persistent Homology Computation for Attention Maps
Implements TDA analysis of PhoBERT attention patterns
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from ripser import ripser
import warnings

warnings.filterwarnings('ignore')


class PersistentHomologyComputer:
    """
    Computes persistent homology from attention maps
    
    Pipeline:
    1. Extract attention maps from PhoBERT layers
    2. Convert attention to distance matrix
    3. Compute Vietoris-Rips filtration
    4. Extract persistence diagrams (H₀, H₁)
    """
    
    def __init__(
        self,
        selected_layers: List[int] = [8, 9, 10, 11],
        homology_dims: List[int] = [0, 1],
        max_dimension: int = 1,
        distance_metric: str = 'precomputed'
    ):
        """
        Args:
            selected_layers: Which PhoBERT layers to analyze
            homology_dims: Which homology dimensions to compute (H₀, H₁, etc.)
            max_dimension: Maximum simplicial dimension
            distance_metric: Distance metric ('precomputed' for attention)
        """
        self.selected_layers = selected_layers
        self.homology_dims = homology_dims
        self.max_dimension = max_dimension
        self.distance_metric = distance_metric
    
    def compute_from_attention(
        self,
        attention_maps: torch.Tensor,
        aggregate_heads: bool = True
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Compute persistent homology from attention maps
        
        Args:
            attention_maps: Attention tensor [num_layers, num_heads, seq_len, seq_len]
            aggregate_heads: Whether to aggregate attention heads
        
        Returns:
            Dictionary: {layer_idx: {homology_dim: persistence_diagram}}
        """
        results = {}
        
        for layer_idx in self.selected_layers:
            # Get attention for this layer
            layer_attention = attention_maps[layer_idx]  # [num_heads, seq_len, seq_len]
            
            # Aggregate heads if needed
            if aggregate_heads:
                aggregated_attention = self._aggregate_attention_heads(layer_attention)
            else:
                aggregated_attention = layer_attention.mean(dim=0)  # Simple mean
            
            # Convert to distance matrix
            distance_matrix = self._attention_to_distance(aggregated_attention)
            
            # Compute persistent homology
            persistence_diagrams = self._compute_persistence(distance_matrix)
            
            results[layer_idx] = persistence_diagrams
        
        return results
    
    def _aggregate_attention_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multiple attention heads using mean pooling
        
        Args:
            attention: [num_heads, seq_len, seq_len]
        
        Returns:
            Aggregated attention: [seq_len, seq_len]
        """
        # Simple mean aggregation (can be extended to weighted average)
        return attention.mean(dim=0)
    
    def _attention_to_distance(self, attention: torch.Tensor) -> np.ndarray:
        """
        Convert attention matrix to distance matrix
        
        Distance = 1 - Attention (higher attention = smaller distance)
        
        Args:
            attention: [seq_len, seq_len]
        
        Returns:
            Distance matrix: [seq_len, seq_len]
        """
        # Convert to numpy
        attention_np = attention.detach().cpu().numpy()
        
        # Ensure symmetry (attention might not be perfectly symmetric)
        attention_symmetric = (attention_np + attention_np.T) / 2
        
        # Convert to distance
        distance_matrix = 1.0 - attention_symmetric
        
        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Ensure non-negative and within [0, 1]
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        
        return distance_matrix
    
    def _compute_persistence(
        self,
        distance_matrix: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Compute persistent homology using Ripser
        
        Args:
            distance_matrix: [seq_len, seq_len] distance matrix
        
        Returns:
            Dictionary: {homology_dim: persistence_diagram}
        """
        try:
            # Compute persistence using Ripser
            result = ripser(
                distance_matrix,
                maxdim=self.max_dimension,
                distance_matrix=True
            )
            
            # Extract diagrams for requested dimensions
            diagrams = {}
            for dim in self.homology_dims:
                if dim < len(result['dgms']):
                    diagrams[dim] = result['dgms'][dim]
                else:
                    # Empty diagram if dimension not computed
                    diagrams[dim] = np.array([]).reshape(0, 2)
            
            return diagrams
        
        except Exception as e:
            # Return empty diagrams on error
            print(f"Warning: PH computation failed: {e}")
            return {dim: np.array([]).reshape(0, 2) for dim in self.homology_dims}
    
    def filter_persistence(
        self,
        persistence_diagram: np.ndarray,
        threshold: float = 0.01
    ) -> np.ndarray:
        """
        Filter persistence diagram by persistence threshold
        
        Remove features with persistence < threshold (likely noise)
        
        Args:
            persistence_diagram: [n_features, 2] array of (birth, death)
            threshold: Minimum persistence to keep
        
        Returns:
            Filtered persistence diagram
        """
        if persistence_diagram.shape[0] == 0:
            return persistence_diagram
        
        # Compute persistence
        persistence = persistence_diagram[:, 1] - persistence_diagram[:, 0]
        
        # Filter
        mask = persistence >= threshold
        filtered = persistence_diagram[mask]
        
        return filtered
    
    def get_statistics(
        self,
        persistence_diagrams: Dict[int, Dict[int, np.ndarray]]
    ) -> Dict[str, any]:
        """
        Compute statistics of persistence diagrams
        
        Args:
            persistence_diagrams: {layer: {dim: diagram}}
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        for layer, diagrams in persistence_diagrams.items():
            layer_stats = {}
            
            for dim, diagram in diagrams.items():
                if diagram.shape[0] == 0:
                    layer_stats[f'H{dim}_num_features'] = 0
                    layer_stats[f'H{dim}_mean_persistence'] = 0.0
                    layer_stats[f'H{dim}_max_persistence'] = 0.0
                else:
                    # Compute persistence
                    persistence = diagram[:, 1] - diagram[:, 0]
                    
                    layer_stats[f'H{dim}_num_features'] = len(diagram)
                    layer_stats[f'H{dim}_mean_persistence'] = float(persistence.mean())
                    layer_stats[f'H{dim}_max_persistence'] = float(persistence.max())
                    layer_stats[f'H{dim}_total_persistence'] = float(persistence.sum())
            
            stats[f'layer_{layer}'] = layer_stats
        
        return stats
    
    def compute_bottleneck_distance(
        self,
        diagram1: np.ndarray,
        diagram2: np.ndarray
    ) -> float:
        """
        Compute bottleneck distance between two persistence diagrams
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
        
        Returns:
            Bottleneck distance
        """
        try:
            from persim import bottleneck
            return bottleneck(diagram1, diagram2)
        except ImportError:
            print("Warning: persim not installed, cannot compute bottleneck distance")
            return 0.0
    
    def visualize_persistence_diagram(
        self,
        persistence_diagram: np.ndarray,
        title: str = "Persistence Diagram",
        show_diagonal: bool = True
    ):
        """
        Visualize a persistence diagram
        
        Args:
            persistence_diagram: [n_features, 2] array
            title: Plot title
            show_diagonal: Whether to show the diagonal line
        """
        try:
            import matplotlib.pyplot as plt
            
            if persistence_diagram.shape[0] == 0:
                print(f"Empty persistence diagram for {title}")
                return
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot points
            ax.scatter(
                persistence_diagram[:, 0],
                persistence_diagram[:, 1],
                alpha=0.6,
                s=50
            )
            
            # Plot diagonal
            if show_diagonal:
                max_val = max(
                    persistence_diagram[:, 0].max(),
                    persistence_diagram[:, 1].max()
                )
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')
            
            ax.set_xlabel('Birth', fontsize=12)
            ax.set_ylabel('Death', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        except ImportError:
            print("Warning: matplotlib not available for visualization")


def extract_attention_maps(
    model_outputs,
    selected_layers: List[int] = [8, 9, 10, 11]
) -> torch.Tensor:
    """
    Extract attention maps from PhoBERT model outputs
    
    Args:
        model_outputs: HuggingFace model outputs with attentions
        selected_layers: Which layers to extract
    
    Returns:
        Attention tensor: [num_selected_layers, num_heads, seq_len, seq_len]
    """
    all_attentions = model_outputs.attentions  # Tuple of [batch, heads, seq, seq]
    
    # Extract selected layers
    selected_attentions = [all_attentions[i] for i in selected_layers]
    
    # Stack and take first batch item
    # Shape: [num_selected_layers, num_heads, seq_len, seq_len]
    attention_tensor = torch.stack(selected_attentions, dim=0).squeeze(1)
    
    return attention_tensor