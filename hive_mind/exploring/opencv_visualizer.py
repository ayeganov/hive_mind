from dataclasses import dataclass
from typing import Iterable
import cv2
import numpy as np

from .environment import Environment, Peak
from .visualizer import Visualizer
from hive_mind.agent import Agent


def draw_text(img,
              text,
              position,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1,
              color=(255, 255, 255),
              thickness=1):
    """
    Draw text on an image.

    Args:
        img (numpy.ndarray): The input image.
        text (str): The text to be drawn.
        position (tuple): The top-left starting position of the text (x, y).
        font (int, optional): The font style. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): The font size. Defaults to 1.
        color (tuple, optional): The text color in BGR format. Defaults to (255, 255, 255).
        thickness (int, optional): The thickness of the text lines. Defaults to 2.
    """
    cv2.putText(img, text, position, font, font_scale, color, thickness)


@dataclass
class RenderAgents:
    agents: Iterable[Agent]
    dist: float = float("inf")
    closest_id: str | None = None
    peaks: list[Peak] | None = None


class OpenCVHillClimberVisualizer(Visualizer[RenderAgents]):
    """
    Concrete implementation of the Visualizer interface using OpenCV.
    Visualizes an environment image and overlays multiple agents on it.
    """

    def __init__(self, window_name: str = "Visualizer"):
        """
        Initialize the OpenCVVisualizer.

        :param window_name: Name of the OpenCV window.
        """
        self._environment = None
        self._window_name = window_name
        self.is_rendering = False

    def set_environment(self, environment: Environment) -> None:
        """
        Set or update the environment to be visualized.

        :param environment: An instance of Environment.
        """
        self._environment = environment

    def clear(self):
        self.environment = None
        self.is_rendering = False

    def render(self, ctx: RenderAgents) -> None:
        """
        Render the current state of the environment and agents using OpenCV.
        """
        if self._environment is None:
            print("No environment set for visualization.")
            return

        env_data = self._environment.get_data()

        if not isinstance(env_data, np.ndarray):
            print("Environment data is not a valid image array.")
            return

        if ctx.peaks is None:
            print("No peaks passed in the render context")
            return

        goal = ctx.peaks[0]

        goal_xy = (int(goal.x), int(goal.y))
        display_image = cv2.cvtColor(env_data, cv2.COLOR_GRAY2BGR)
        cv2.circle(display_image, goal_xy, radius=4, color=(0, 0, 255), thickness=-1)
        cv2.circle(display_image, goal_xy, radius=20, color=(0, 0, 255), thickness=1)

        for agent in ctx.agents:
            loc = agent.location
            x, y = int(loc.get('x', 0)), int(loc.get('y', 0))
            body_direction = np.array(agent.body_direction)

            height, width = display_image.shape[:2]
            is_within_bounds = 0 <= x < width and 0 <= y < height

            if is_within_bounds:
                cv2.circle(display_image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

                arrow_length = 20
                body_end_x = int(x + arrow_length * body_direction[0])
                body_end_y = int(y + arrow_length * body_direction[1])

                if ctx.closest_id is not None and agent.id == ctx.closest_id:
                    cv2.arrowedLine(display_image, (x, y), (body_end_x, body_end_y), color=(205, 20, 100), thickness=2, tipLength=0.3)
                    draw_text(display_image, f"Dist: {ctx.dist:.0f}", (0, 10), font_scale=0.3)
                else:
                    cv2.arrowedLine(display_image, (x, y), (body_end_x, body_end_y), color=(0, 255, 0), thickness=2, tipLength=0.3)

                self._draw_vision_area(agent, display_image)

        display_image = cv2.resize(display_image, (800, 800))
        cv2.imshow(self._window_name, display_image)
        cv2.waitKey(1)

    def _draw_vision_area(self, agent: Agent, disp_img: np.ndarray) -> None:
        body_direction = np.array(agent.body_direction)
        gaze_direction = np.array(agent.gaze_direction)
        view_radius = agent._view_radius
        focus = agent.focus

        final_direction = body_direction + gaze_direction
        magnitude = np.linalg.norm(final_direction)
        if magnitude != 0:
            final_direction /= magnitude

        area = view_radius * view_radius
        view_width = int(np.sqrt(2 * area * focus))
        view_width = max(view_width, 2)

        view_height = int(np.ceil(area / view_width))
        view_height = max(view_height, 2)

        rotation_angle = np.degrees(np.arctan2(-final_direction[1], final_direction[0]))

        loc = agent.location
        x, y = int(loc.get('x', 0)), int(loc.get('y', 0))
        rect_points = cv2.boxPoints((
            (x, y),
            (view_width, view_height),
            -rotation_angle
        ))
        rect_points = np.int32(rect_points)

        cv2.drawContours(disp_img, [rect_points], 0, (255, 0, 0), 1)


    def close(self) -> None:
        """
        Close the OpenCV window.
        """
        cv2.destroyWindow(self._window_name)
        self.is_rendering = False

    def start_rendering_loop(self, delay: int = 1) -> None:
        """
        Start a continuous rendering loop. This method is blocking.

        :param delay: Delay in milliseconds between frames.
        """
        self.is_rendering = True
        while self.is_rendering:
            # TODO: must fix the render call
            # self.render()
            key = cv2.waitKey(delay)
            if key == ord('q'):  # Press 'q' to quit the rendering loop
                self.is_rendering = False
        self.close()

    def stop_rendering_loop(self) -> None:
        """
        Stop the continuous rendering loop.
        """
        self.is_rendering = False
        self.close()

