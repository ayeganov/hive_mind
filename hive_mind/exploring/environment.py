from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
from typing import Any
import random
import uuid

from scipy.spatial import Voronoi
import numpy as np

from hive_mind.agent import Entity


class Environment(Entity, ABC):
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
        self._id = str(uuid.uuid4())
        self.generate_surface()

    @property
    def id(self) -> str:
        return self._id

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
        """Generates a new surface with guaranteed separated peaks"""
        base_surface: np.ndarray = np.zeros((self.width, self.height))
        self.peaks.clear()

        num_points = self.complexity
        points = np.random.rand(num_points, 2)
        points = points * [self.width, self.height]

        # Generate heights for each point
        heights = np.random.uniform(0.4, 1.0, size=len(points))

        # Create coordinate matrices
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # Falloff parameter for smoothness
        falloff = min(self.width, self.height) / 4

        # For each point/peak
        for i, (px, py) in enumerate(points):
            # Calculate distance to this point for all positions
            dist = np.sqrt((X - px)**2 + (Y - py)**2)

            # Use smooth falloff function
            height_contribution = heights[i] * np.exp(-(dist**2) / (2 * falloff**2))

            # Add to base surface
            base_surface = np.maximum(height_contribution, base_surface)

            # Store peak information
            self.peaks.append(Peak(px, py, heights[i], falloff, "gaussian"))

        # Normalize final surface
        if base_surface.max() > 0:
            base_surface = (base_surface - base_surface.min()) / (base_surface.max() - base_surface.min())

        self._surface = (base_surface * 255).astype(np.uint8)
        self.peaks = self.verify_peaks()

    def verify_peaks(self) -> list[Peak]:
        """Verify that each peak is still a local maximum"""
        verified_peaks = []
        window_size = 8
        if self._surface is None:
            return verified_peaks

        for peak in self.peaks:
            x_idx = int(peak.x)
            y_idx = int(peak.y)
 
            # Get local region around peak
            x_start = max(0, x_idx - window_size//2)
            x_end = min(self.width, x_idx + window_size//2 + 1)
            y_start = max(0, y_idx - window_size//2)
            y_end = min(self.height, y_idx + window_size//2 + 1)

            local_region = self._surface[y_start:y_end, x_start:x_end]

            # Check if peak is local maximum
            peak_height = self._surface[y_idx, x_idx]
            local_max = np.max(local_region)
            if peak_height == local_max or local_max == 255:
                verified_peaks.append(peak)

        return verified_peaks

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
