from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
from typing import Any
import random

import numpy as np


class Environment(ABC):
    """
    Abstract base class defining the interface for an environment.
    """

    @abstractmethod
    def get_data(self) -> Any:
        """
        Retrieve the current environment data.

        :return: The environment data (e.g., image, audio).
        """

    @abstractmethod
    def update_data(self, new_data: Any) -> None:
        """
        Update the environment data.

        :param new_data: The new environment data.
        """

    @property
    @abstractmethod
    def boundaries(self) -> tuple:
        """
        Return the boundaries of this environment
        """


@dataclass
class Peak:
    x: float
    y: float
    height: float
    steepness: float
    method: str


class HillEnvironment(Environment):
    def __init__(self, width: int = 100, height: int = 100) -> None:
        self.width: int = width
        self.height: int = height
        self.complexity: int = 1  # Number of peaks
        # Each peak: (x, y, height, steepness, method)
        self.peaks: list[Peak] = []
        self._surface: np.ndarray | None = None
        self.generate_surface()

    def get_data(self) -> np.ndarray:
        """Returns the current hill surface image"""
        assert self._surface is not None
        return self._surface

    def update_data(self, new_data: np.ndarray) -> None:
        """Updates the surface with new data"""
        assert new_data.shape == (self.width, self.height)
        self._surface = new_data

    @property
    def boundaries(self) -> tuple[int, int]:
        """Returns (width, height) of the environment"""
        return (self.width, self.height)

    def generate_surface(self) -> None:
        """Generates a new surface based on current complexity"""
        base_surface: np.ndarray = np.zeros((self.width, self.height))
        self.peaks.clear()

        # Generate peaks
        for _ in range(self.complexity):
            # Random peak parameters
            peak_x: float = np.random.uniform() * self.width
            peak_y: float = np.random.uniform() * self.height
            height: float = np.random.uniform(0.5, 1.0)
            steepness: float = np.random.uniform(3, 7)
#            method: str = np.random.choice(['sigmoid', 'quadratic', 'inverse_quadratic'])
            method: str = "sigmoid"

            self.peaks.append(Peak(peak_x, peak_y, height, steepness, method))

            # Generate peak surface
            peak_surface: np.ndarray = self._create_hill_image(
                peak_x=peak_x,
                peak_y=peak_y,
                method=method,
                steepness=steepness
            )

            # Combine with existing surface
            base_surface = np.maximum(base_surface, peak_surface * height)

        # Normalize final surface
        if base_surface.max() > 0:  # Avoid division by zero
            base_surface = (base_surface - base_surface.min()) / (base_surface.max() - base_surface.min())

        self._surface = (base_surface * 255).astype(np.uint8)

    def _create_hill_image(self,
                           peak_x: float,
                           peak_y: float, 
                           method: str = 'sigmoid',
                           steepness: float = 5,) -> np.ndarray:
        """Creates a single hill peak"""
        # Create coordinates with correct shape (height, width)
        x: np.ndarray = np.linspace(0, self.width, self.width)
        y: np.ndarray = np.linspace(0, self.height, self.height)
        # Important: meshgrid needs to match the expected output shape
        x, y = np.meshgrid(x, y, indexing='ij')

        # Calculate distances from each point to the peak
        distances: np.ndarray = np.sqrt((x - peak_x)**2 + (y - peak_y)**2)

        # Normalize distances
        max_distance: float = np.sqrt(self.width**2 + self.height**2)
        normalized_distances: np.ndarray = distances / max_distance

        # Generate hill based on method
        if method == 'sigmoid':
            hill: np.ndarray = 1 / (1 + np.exp(steepness * (normalized_distances - 0.5)))
        elif method == 'quadratic':
            hill = 1 - (normalized_distances ** 2)
            hill = np.maximum(hill, 0)
        elif method == 'inverse_quadratic':
            hill = 1 / (1 + steepness * normalized_distances ** 2)
        else:
            raise ValueError(f"Invalid method: {method}")

        return hill

    def mutate(self) -> None:
        """Mutates the environment by potentially adding peaks or modifying existing ones"""
        # Chance to add new peak
        if random.random() < 0.2:  # 20% chance to add peak
            self.complexity += 1
            self.generate_surface()
            return

        # Otherwise modify existing peaks
        if self.peaks:  # If we have peaks to modify
            peak_idx: int = random.randrange(len(self.peaks))
            x, y, height, steepness, method = dataclasses.astuple(self.peaks[peak_idx])

            # Randomly modify one aspect
            mutation_type: str = random.choice(['position', 'height', 'steepness', 'method'])

            if mutation_type == 'position':
                x += random.gauss(0, self.width * 0.1)  # 10% of width
                y += random.gauss(0, self.height * 0.1)  # 10% of height
                x = np.clip(x, 0, self.width)
                y = np.clip(y, 0, self.height)
            elif mutation_type == 'height':
                height += random.gauss(0, 0.1)  # Adjust height by up to ±0.1
                height = np.clip(height, 0.1, 1.0)
            elif mutation_type == 'steepness':
                steepness += random.gauss(0, 1.0)
                steepness = np.clip(steepness, 2.0, 8.0)
            else:  # method
                method = random.choice(['sigmoid', 'quadratic', 'inverse_quadratic'])

            self.peaks[peak_idx] = Peak(x, y, height, steepness, method)
            self.generate_surface()

    def get_height(self, x: float, y: float) -> float:
        """Returns the height at given coordinates"""
        assert self._surface is not None

        # Convert to integer indices
        x_idx: int = int(x)
        y_idx: int = int(y)

        # Ensure within boundaries
        x_idx = np.clip(x_idx, 0, self.width - 1)
        y_idx = np.clip(y_idx, 0, self.height - 1)

        return self._surface[y_idx, x_idx] / 255.0  # Normalize to [0,1]

    def get_peak_positions(self) -> list[tuple[int, int]]:
        """Returns list of peak positions"""
        return [(int(p.x), int(p.y)) for p in self.peaks]
