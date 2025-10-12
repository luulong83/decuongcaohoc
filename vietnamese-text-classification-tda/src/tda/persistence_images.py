"""
Persistence Images - Vectorization of Persistence Diagrams
Converts topological features to fixed-size vectors for deep learning
"""

import numpy as np
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class PersistenceImagesVectorizer:
    """
    Converts persistence diagrams to Persistence Images (PI)
    
    Reference: Adams et al. (2017) - "Persistence Images: A stable vector 
    representation of persistent homology"
    
    Pipeline:
    1. Transform diagram: (birth, death) → (birth, persistence)
    2. Create 2D grid
    3. Apply weight function (persistence-based)
    4. Gaussian smoothing
    5. Normalize and flatten
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (20, 20),
        sigma: float = 0.1,
        weight_function: str = 'persistence',
        normalize: bool = True,
        birth_range: Optional[Tuple[float, float]] = None,
        persistence_range: Optional[Tuple[float, float]] = None
    ):
        """
        Args:
            resolution: Grid resolution (height, width)
            sigma: Gaussian smoothing parameter
            weight_function: 'persistence', 'uniform', or 'linear'
            normalize: Whether to normalize the image
            birth_range: Range for birth axis (auto if None)
            persistence_range: Range for persistence axis (auto if None)
        """
        self.resolution = resolution
        self.sigma = sigma
        self.weight_function = weight_function
        self.normalize = normalize
        self.birth_range = birth_range
        self.persistence_range = persistence_range
        
        # Grid will be created dynamically based on data
        self.x_grid = None
        self.y_grid = None
    
    def fit(self, persistence_diagrams: list):
        """
        Fit the vectorizer to determine grid ranges
        
        Args:
            persistence_diagrams: List of persistence diagrams
        """
        if len(persistence_diagrams) == 0:
            return
        
        # Collect all points
        all_points = []
        for diagram in persistence_diagrams:
            if diagram.shape[0] > 0:
                all_points.append(diagram)
        
        if len(all_points) == 0:
            return
        
        all_points = np.vstack(all_points)
        
        # Determine ranges
        if self.birth_range is None:
            birth_min = all_points[:, 0].min()
            birth_max = all_points[:, 0].max()
            margin = (birth_max - birth_min) * 0.1
            self.birth_range = (birth_min - margin, birth_max + margin)
        
        if self.persistence_range is None:
            persistence = all_points[:, 1] - all_points[:, 0]
            pers_min = 0.0  # Persistence is always >= 0
            pers_max = persistence.max()
            margin = pers_max * 0.1
            self.persistence_range = (pers_min, pers_max + margin)
    
    def transform(self, persistence_diagram: np.ndarray) -> np.ndarray:
        """
        Transform a persistence diagram to a Persistence Image
        
        Args:
            persistence_diagram: [n_features, 2] array of (birth, death)
        
        Returns:
            Persistence Image: [resolution[0] * resolution[1]] flattened vector
        """
        # Handle empty diagram
        if persistence_diagram.shape[0] == 0:
            return np.zeros(self.resolution[0] * self.resolution[1])
        
        # Transform to birth-persistence coordinates
        birth_pers = self._transform_coordinates(persistence_diagram)
        
        # Create grid if not exists
        if self.x_grid is None or self.y_grid is None:
            self._create_grid()
        
        # Compute weights
        weights = self._compute_weights(birth_pers)
        
        # Create image
        image = self._create_image(birth_pers, weights)
        
        # Normalize
        if self.normalize and image.sum() > 0:
            image = image / image.sum()
        
        # Flatten
        return image.flatten()
    
    def fit_transform(self, persistence_diagrams: list) -> np.ndarray:
        """
        Fit and transform multiple persistence diagrams
        
        Args:
            persistence_diagrams: List of persistence diagrams
        
        Returns:
            Array of shape [n_diagrams, resolution[0] * resolution[1]]
        """
        self.fit(persistence_diagrams)
        
        images = []
        for diagram in persistence_diagrams:
            images.append(self.transform(diagram))
        
        return np.array(images)
    
    def _transform_coordinates(self, diagram: np.ndarray) -> np.ndarray:
        """
        Transform (birth, death) to (birth, persistence)
        
        Args:
            diagram: [n, 2] array of (birth, death)
        
        Returns:
            [n, 2] array of (birth, persistence)
        """
        birth = diagram[:, 0]
        death = diagram[:, 1]
        persistence = death - birth
        
        return np.column_stack([birth, persistence])
    
    def _create_grid(self):
        """Create 2D grid for the image"""
        if self.birth_range is None:
            self.birth_range = (0.0, 1.0)
        if self.persistence_range is None:
            self.persistence_range = (0.0, 1.0)
        
        x_vals = np.linspace(
            self.birth_range[0],
            self.birth_range[1],
            self.resolution[1]
        )
        y_vals = np.linspace(
            self.persistence_range[0],
            self.persistence_range[1],
            self.resolution[0]
        )
        
        self.x_grid, self.y_grid = np.meshgrid(x_vals, y_vals)
    
    def _compute_weights(self, birth_pers: np.ndarray) -> np.ndarray:
        """
        Compute weights for each persistence point
        
        Args:
            birth_pers: [n, 2] array of (birth, persistence)
        
        Returns:
            [n] array of weights
        """
        persistence = birth_pers[:, 1]
        
        if self.weight_function == 'persistence':
            # Weight by persistence value
            weights = persistence
        elif self.weight_function == 'linear':
            # Linear weighting
            weights = persistence / (persistence.max() + 1e-8)
        elif self.weight_function == 'uniform':