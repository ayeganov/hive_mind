from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override
import uuid

import cv2
import noise
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
    def __init__(self, width: int = 100, height: int = 100, complexity: int = 1) -> None:
        self.width: int = width
        self.height: int = height
        self._complexity: int = complexity
        # Each peak: (x, y, height, steepness, method)
        self._peaks: list[Peak] = []
        self._surface: np.ndarray | None = None
        self._id = str(uuid.uuid4())
        self.generate_surface()

    @property
    def id(self) -> str:
        return self._id

    @property
    def type(self) -> str:
        return "env"

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
        self._peaks.clear()

        num_points = self._complexity
        points = np.random.rand(num_points, 2)
        points = points * [self.width, self.height]

        # Generate heights for each point
        heights = np.random.uniform(0.5, 1.0, size=len(points))

        # Create coordinate matrices
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # Falloff parameter for smoothness
        falloff = min(self.width, self.height) / 2

        # For each point/peak
        for i, (px, py) in enumerate(points):
            # Calculate distance to this point for all positions
            dist = np.sqrt((X - px)**2 + (Y - py)**2)

            # Use smooth falloff function
            height_contribution = heights[i] * np.exp(-(dist**2) / (2 * falloff**2))

            # Add to base surface
            base_surface = np.maximum(height_contribution, base_surface)
            base_surface += np.random.uniform(-0.03, 0.03, base_surface.shape)

            # Store peak information
            self._peaks.append(Peak(px, py, heights[i], falloff, "gaussian"))

        # Normalize final surface
        if base_surface.max() > 0:
            base_surface = (base_surface - base_surface.min()) / (base_surface.max() - base_surface.min())

        self._surface = (base_surface * 255).astype(np.uint8)
        self._peaks = self.verify_peaks()

    def verify_peaks(self) -> list[Peak]:
        """Verify that each peak is still a local maximum"""
        verified_peaks = []
        window_size = 8
        if self._surface is None:
            return verified_peaks

        for peak in self._peaks:
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

        return sorted(verified_peaks, key=lambda p: p.height, reverse=True)

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
        return [(int(p.x), int(p.y)) for p in self._peaks]


class SlopedEnvironment(Environment):
    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        slope_x: float | None = None,
        slope_y: float | None = None,
        intercept: float = 0.0,
        complexity: int = 1,
    ) -> None:
        self.width = width
        self.height = height
        self.complexity = complexity
        self.intercept = intercept
        self._surface: np.ndarray | None = None
        self._id = str(uuid.uuid4())
        self.initialize_slopes(slope_x, slope_y)
        self.generate_surface()

    def initialize_slopes(self, slope_x: float | None, slope_y: float | None) -> None:
        """Initialize the slopes based on complexity."""
        max_slope = self.complexity * 0.1  # Adjust this factor to control steepness per complexity
        self.slope_x = slope_x if slope_x is not None else np.random.uniform(-max_slope, max_slope)
        self.slope_y = slope_y if slope_y is not None else np.random.uniform(-max_slope, max_slope)

    @property
    def id(self) -> str:
        return self._id

    @property
    def type(self) -> str:
        return "env"

    def get_data(self) -> np.ndarray:
        """Returns the current sloped surface image."""
        assert self._surface is not None
        return self._surface

    def update_data(self, new_data: np.ndarray) -> None:
        """Updates the surface with new data."""
        assert new_data.shape == (self.height, self.width)
        self._surface = new_data

    @property
    def boundaries(self) -> tuple[int, int]:
        """Returns (width, height) of the environment."""
        return (self.width, self.height)

    def generate_surface(self) -> None:
        """Generates a new sloped flat surface."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # Calculate the height at each point using the slopes.
        heights = self.slope_x * X + self.slope_y * Y + self.intercept

        heights += np.random.uniform(-0.05, 0.05, heights.shape)

        # Instead of normalizing the heights to [0, 255], we adjust the scaling to reflect the actual steepness.
        # We will translate the heights to positive values if necessary.
        min_height = heights.min()
        if min_height < 0:
            heights -= min_height  # Shift heights so that minimum is 0

        # Optionally scale the heights to prevent overflow when converting to uint8
        max_height = heights.max()
        scaling_factor = 1.0
        if max_height > 255:
            scaling_factor = 255 / max_height
            heights *= scaling_factor

        self._surface = heights.astype(np.uint8)

    def get_height(self, x: float, y: float) -> float:
        """Returns the actual height at given coordinates."""
        assert self._surface is not None

        # Convert to integer indices.
        x_idx: int = int(x)
        y_idx: int = int(y)

        # Ensure indices are within boundaries.
        x_idx = np.clip(x_idx, 0, self.width - 1)
        y_idx = np.clip(y_idx, 0, self.height - 1)

        return float(self._surface[y_idx, x_idx])

    def mutate(self) -> None:
        """Mutate the environment by possibly increasing complexity and adjusting slopes, driven by chance."""
        # Decide randomly whether to increase complexity
        if np.random.rand() < 0.5:
            # Increase complexity
            self.complexity += 1
            # Re-initialize slopes based on new complexity
            self.initialize_slopes(None, None)
        else:
            # Adjust slopes randomly within current complexity
            max_slope = self.complexity * 0.1
            self.slope_x = np.random.uniform(-max_slope, max_slope)
            self.slope_y = np.random.uniform(-max_slope, max_slope)

        # Re-generate the surface with updated parameters
        self.generate_surface()


class Terrain(Environment):
    def __init__(self, width: int, height: int, scale: float = 100.0):
        """
        Initialize the terrain with the given dimensions and noise scale.
        :param width: Width of the terrain grid.
        :param height: Height of the terrain grid.
        :param scale: Scale of the noise, influencing the level of detail.
        """
        self._width = width
        self._height = height
        self._scale = scale
        self._terrain_data = self._generate_perlin_noise()
        self._id = str(uuid.uuid4())
        self._peaks = self._find_peaks()

    @property
    @override
    def id(self) -> str:
        return self._id

    @property
    @override
    def type(self) -> str:
        return "env"

    @property
    def peaks(self) -> list:
        return self._peaks

    def _generate_perlin_noise(self) -> np.ndarray:
        """
        Generate terrain height values using Perlin noise.
        :return: A 2D numpy array representing terrain height values.
        """
        terrain = np.zeros((self._height, self._width))

        for y in range(self._height):
            for x in range(self._width):
                terrain[y][x] = noise.pnoise2(x / self._scale,
                                              y / self._scale,
                                              octaves=6,
                                              persistence=0.2,
                                              lacunarity=2.0,
                                              repeatx=self._width,
                                              repeaty=self._height,
                                              base=42)
        terrain_normalized = (terrain + 1.0) / 2.0
        terrain_scaled = terrain_normalized * 255.0

        min_value = terrain_scaled.min()
        terrain_scaled -= min_value

        return terrain_scaled

    def _find_peaks(self) -> list:
        """
        Find all peaks in the terrain by comparing each cell to its neighbors.
        :return: A list of tuples representing the coordinates of the peaks.
        """
        peaks = []

        for y in range(1, self._height - 1):
            for x in range(1, self._width - 1):
                # Get the value of the current cell
                current_value = self._terrain_data[y, x]

                # Get the values of the 8 neighboring cells
                neighbors = [
                    self._terrain_data[y - 1, x - 1], self._terrain_data[y - 1, x], self._terrain_data[y - 1, x + 1],
                    self._terrain_data[y, x - 1],                             self._terrain_data[y, x + 1],
                    self._terrain_data[y + 1, x - 1], self._terrain_data[y + 1, x], self._terrain_data[y + 1, x + 1]
                ]

                # Check if the current cell is greater than all its neighbors
                if all(current_value > neighbor for neighbor in neighbors):
                    peaks.append((x, y))

        return peaks

    @override
    def get_data(self) -> np.ndarray:
        """
        Retrieve the current terrain data.
        :return: A 2D numpy array representing the terrain height values.
        """
        return self._terrain_data

    @override
    def update_data(self, new_data: Any) -> None:
        pass

    @property
    @override
    def boundaries(self) -> tuple[int, int]:
        """
        Return the boundaries of this terrain.
        :return: A tuple containing width and height of the terrain.
        """
        return self._width, self._height
