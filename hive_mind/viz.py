import numpy as np
import pyvista as pv


class AnimatedHillPlotter:
    def __init__(self,
                 window_size: tuple[int, int] = (1024, 768),
                 fps: int = 30,
                 smooth_factor: float = 0.1):
        """
        Initialize the plotter with a given window size.

        Args:
            window_size: Tuple of (width, height) for the window
            fps: Frames per second for animations
            smooth_factor: Factor controlling transition smoothness (0-1)
        """
        self._plotter = pv.Plotter(window_size=window_size, off_screen=False)
        self._current_mesh = None
        self._current_grid = None
        self._images: list[np.ndarray] = []
        self._titles: list[str] = []
        self._peaks: list[list[tuple[int, int]]] = []
        self._current_image_index = 0
        self._is_animating = False
        self._auto_play = False
        self._peak_actors: list[pv.Actor] = []

        # Animation parameters
        self._steps = int(1 / smooth_factor)
        self._duration = int(1000 * 1 / fps)  # Duration in ms
        self._current_step = 0
        self._old_heights = None
        self._new_heights = None

    def _update_peaks(self, image_index: int) -> None:
        """
        Update peak markers for the current image.

        Args:
            image_index: Index of the current image
        """
        # Remove existing peak markers
        for actor in self._peak_actors:
            self._plotter.remove_actor(actor) # type: ignore
        self._peak_actors.clear()

        # Add new peak markers
        if image_index < len(self._peaks):
            current_image = self._images[image_index]
            for peak_y, peak_x in self._peaks[image_index]:
                # Ensure peak coordinates are within bounds
                peak_x_int = np.clip(peak_x, 0, current_image.shape[1] - 1)
                peak_y_int = np.clip(peak_y, 0, current_image.shape[0] - 1)

                # Get height at peak location and add 5%
                peak_z = current_image[peak_x_int, peak_y_int] * 1.05

                # Create and add sphere at peak location
                sphere = pv.Sphere(
                    center=(peak_x, peak_y, peak_z),
                    radius=1.5
                )
                actor = self._plotter.add_mesh(
                    sphere,
                    color="red",
                    ambient=0.3,
                    diffuse=0.7,
                    specular=0.5,
                )
                self._peak_actors.append(actor)

    def _setup_plot(self, initial_shape: tuple[int, int]) -> None:
        """
        Set up the initial plot configuration.

        Args:
            initial_shape: Shape of the data grid (height, width)
        """
        self._plotter.show_axes() # type: ignore
        self._plotter.add_axes( # type: ignore
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
        if self._titles:
            self._plotter.add_title(self._titles[0])
        if self._peaks:
            self._update_peaks(0)
        self._plotter.reset_camera() # type: ignore

    def _create_grid(self, hill_image: np.ndarray) -> pv.StructuredGrid:
        """Create a structured grid from the image data."""
        x = np.arange(0, hill_image.shape[1], 1)
        y = np.arange(0, hill_image.shape[0], 1)
        x, y = np.meshgrid(x, y)
        grid = pv.StructuredGrid(x, y, hill_image)
        grid["heights"] = hill_image.flatten().astype(np.float32)
        return grid

    def _animation_callback(self, step: int) -> None:
        """
        Callback function for the timer event.

        Args:
            step: Current animation step
        """
        assert self._current_grid is not None, "WTF"
        if self._old_heights is None or self._new_heights is None:
            return

        alpha = step / float(self._steps - 1)
        interpolated_heights = ((1 - alpha) * self._old_heights + 
                              alpha * self._new_heights).astype(np.float32)

        self._current_grid.points[:, 2] = interpolated_heights # type: ignore
        self._current_grid["heights"] = interpolated_heights
        self._current_grid.Modified()

        if step >= self._steps - 1:
            self._is_animating = False
            self._current_image_index += 1
            self._old_heights = None
            self._new_heights = None

            if self._auto_play and self._current_image_index < len(self._images):
                self._start_animation_to_next()
        else:
            self._plotter.add_timer_event( # type: ignore
                max_steps=1,
                duration=self._duration,
                callback=lambda _: self._animation_callback(step + 1),
            )

    def _start_animation_to_next(self) -> None:
        """Start animation transition to the next image."""
        if self._current_image_index >= len(self._images):
            return

        if self._current_mesh is None or self._current_grid is None:
            raise RuntimeError("You must initialize the plotter first")

        next_image = self._images[self._current_image_index]
        new_grid = self._create_grid(next_image)

        current_clim = self._current_mesh.mapper.scalar_range
        new_max = np.max(next_image)
        if new_max > current_clim[1]:
            self._current_mesh.mapper.scalar_range = [0, new_max]

        self._old_heights = np.array(self._current_grid["heights"])
        self._new_heights = np.array(new_grid["heights"])
        self._is_animating = True

        if self._current_image_index < len(self._titles):
            self._plotter.add_title(self._titles[self._current_image_index])
            self._update_peaks(self._current_image_index)

        self._plotter.add_timer_event( # type: ignore
            max_steps=1,
            duration=self._duration,
            callback=self._animation_callback
        )

    def _handle_space(self) -> None:
        """Handle space key press event."""
        has_more_images = self._current_image_index < len(self._images)
        if not self._is_animating and has_more_images:
            self._start_animation_to_next()

    def initialize_window(self, 
                         images: list[np.ndarray], 
                         titles: list[str] | None = None,
                         peaks: list[list[tuple[int, int]]] | None = None,
                         auto_play: bool = False) -> None:
        """
        Initialize the plotting window with a list of images and titles.

        Args:
            images: List of 2D numpy arrays containing height data
            titles: Optional list of titles matching the length of images
            peaks: Optional list of peak coordinates for each image
            auto_play: If True, automatically play through all images
        """
        if not images:
            raise ValueError("Must provide at least one image")

        if titles is not None and len(titles) != len(images):
            raise ValueError("Number of titles must match number of images")

        if peaks is not None and len(peaks) != len(images):
            raise ValueError("Number of peak lists must match number of images")

        self._images = images
        self._titles = titles if titles is not None else [f"Image {i}" for i in range(len(images))]
        self._peaks = peaks if peaks is not None else [[] for _ in range(len(images))]
        self._auto_play = auto_play

        self._setup_plot(images[0].shape)
        self._plotter.add_key_event('space', self._handle_space) # type: ignore
        self._plotter.add_key_event('q', self._plotter.close) # type: ignore

        if auto_play:
            self._plotter.add_timer_event(  # type: ignore
                max_steps=1,
                duration=100,  # Small delay before starting
                callback=lambda _: self._start_animation_to_next()
            )

        self._plotter.show(interactive_update=True, auto_close=False)
        self._plotter.iren.initialize()
        self._plotter.iren.start()

    def close(self) -> None:
        """Close the plotting window."""
        self._plotter.close()
