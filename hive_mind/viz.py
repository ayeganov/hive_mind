import numpy as np
import pyvista as pv

import time


class AnimatedHillPlotter:
    def __init__(self, window_size: tuple[int, int] = (1024, 768)):
        """
        Initialize the plotter with a given window size.

        Args:
            window_size: Tuple of (width, height) for the window
        """
        self._plotter = pv.Plotter(window_size=window_size, off_screen=False)
        self._current_mesh = None
        self._current_grid = None
        self._peak_points = None
        self._continue_flag = False

    def _add_key_callback(self):
        """Add keyboard callback for continuing animation."""
        def key_callback():
            self._continue_flag = True

        self._plotter.add_key_event('n', key_callback)

    def _setup_plot(self, initial_shape: tuple[int, int]) -> None:
        """
        Set up the initial plot configuration.

        Args:
            initial_shape: Shape of the data grid (height, width)
        """
        self._plotter.show_axes()
        self._plotter.add_axes(
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
            line_width=2,
            labels_off=False,
            color='black',
        )
        self._plotter.camera_position = 'iso'

        flat_grid = self._create_grid(np.zeros(initial_shape))
        self._current_grid = flat_grid
        self._current_mesh = self._plotter.add_mesh(
            self._current_grid, 
            scalars="heights", 
            cmap="viridis",
            show_edges=False
        )
        self._plotter.reset_camera()

    def _create_grid(self, hill_image: np.ndarray) -> pv.StructuredGrid:
        """Create a structured grid from the image data."""
        x = np.arange(0, hill_image.shape[1], 1)
        y = np.arange(0, hill_image.shape[0], 1)
        x, y = np.meshgrid(x, y)
        grid = pv.StructuredGrid(x, y, hill_image)
        grid["heights"] = hill_image.flatten().astype(np.float32)
        return grid

    def update_plot(self, 
                   hill_image: np.ndarray, 
                   title: str, 
                   peaks: list[tuple[int, int]] | None = None,
                   smooth_factor: float = 0.1,
                   wait_for_key: bool = True) -> None:
        """
        Update the plot with new data.

        Args:
            hill_image: 2D numpy array containing height data
            title: Title for the plot
            peaks: List of peak coordinates (x, y)
            smooth_factor: Factor controlling transition smoothness (0-1)
            wait_for_key: If True, wait for 'n' key press to continue
        """
        if self._current_mesh is None or self._current_grid is None:
            raise RuntimeError("You must initialize the plotter first")

        new_grid = self._create_grid(hill_image)

        current_clim = self._current_mesh.mapper.scalar_range
        new_max = np.max(hill_image)
        if new_max > current_clim[1]:
            self._current_mesh.mapper.scalar_range = [0, new_max]

        old_heights = np.array(self._current_grid["heights"])
        new_heights = np.array(new_grid["heights"])

        steps = int(1 / smooth_factor)
        for i in range(steps):
            alpha = i / float(steps - 1)
            interpolated_heights = ((1 - alpha) * old_heights + alpha * new_heights).astype(np.float32)

            self._current_grid.points[:, 2] = interpolated_heights # type: ignore
            self._current_grid["heights"] = interpolated_heights
            self._current_grid.Modified()

            if not self._plotter.off_screen:
                self._plotter.render()
                time.sleep(0.05)

        self._current_grid.points[:, 2] = new_heights # type: ignore
        self._current_grid["heights"] = new_heights
        self._current_grid.Modified()

        if peaks:
            peak_points = []
            for peak_x, peak_y in peaks:
                peak_x_int = np.clip(peak_x, 0, hill_image.shape[1] - 1)
                peak_y_int = np.clip(peak_y, 0, hill_image.shape[0] - 1)
                peak_z = hill_image[peak_y_int, peak_x_int]
                marker_z = peak_z + 0.01 * np.max(hill_image)
                peak_points.append([peak_x, peak_y, marker_z])

            if peak_points:
                points = pv.PolyData(np.array(peak_points))
                if self._peak_points is not None:
                    self._plotter.remove_actor(self._peak_points)
                self._peak_points = self._plotter.add_mesh(
                    points, 
                    color="red", 
                    point_size=20
                )

        self._plotter.add_title(title)
        if not self._plotter.off_screen:
            self._plotter.render()

        if wait_for_key:
            self._continue_flag = False
            while not self._continue_flag and self._plotter.iren.initialized:
                self._plotter.iren.initialize()
                self._plotter.iren.start()
                self._plotter.iren.process_events()

    def initialize_window(self, initial_shape: tuple[int, int],) -> None:
        """Initialize the plotting window."""
        self._setup_plot(initial_shape)
        self._add_key_callback()
        # Don't use show(interactive=False) as it might block
        self._plotter.show(interactive_update=True, auto_close=False)

    def close(self) -> None:
        """Close the plotting window."""
        self._plotter.close()

