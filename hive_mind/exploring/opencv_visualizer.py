import cv2
import numpy as np

from .environment import Environment
from .visualizer import Visualizer
from hive_mind.agent import Agent


class OpenCVVisualizer(Visualizer):
    """
    Concrete implementation of the Visualizer interface using OpenCV.
    Visualizes an environment image and overlays multiple agents on it.
    """

    def __init__(self, window_name: str = "Visualizer"):
        """
        Initialize the OpenCVVisualizer.

        :param window_name: Name of the OpenCV window.
        """
        self.environment = None
        self.agents: dict[str, Agent] = {}
        self.window_name = window_name
        self.is_rendering = False

    def set_environment(self, environment: Environment) -> None:
        """
        Set or update the environment to be visualized.

        :param environment: An instance of Environment.
        """
        self.environment = environment

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the visualization.

        :param agent: An instance of Agent.
        """
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the visualization based on its ID.

        :param agent_id: The unique identifier of the agent to remove.
        """
        if agent_id in self.agents:
            del self.agents[agent_id]

    def clear(self):
        self.environment = None
        self.agents.clear()
        self.is_rendering = False

    def get_agents(self) -> list[Agent]:
        """
        Retrieve the list of agents currently in the visualization.

        :return: A list of Agent instances.
        """
        return list(self.agents.values())

    def render(self, goal: tuple[int, int], closest_agent: Agent) -> None:
        """
        Render the current state of the environment and agents using OpenCV.
        """
        if self.environment is None:
            print("No environment set for visualization.")
            return

        env_data = self.environment.get_data()

        if not isinstance(env_data, np.ndarray):
            print("Environment data is not a valid image array.")
            return

        display_image = cv2.cvtColor(env_data, cv2.COLOR_GRAY2BGR)
        cv2.circle(display_image, goal, radius=4, color=(0, 0, 255), thickness=-1)

        for agent in self.agents.values():
            loc = agent.location
            x, y = int(loc.get('x', 0)), int(loc.get('y', 0))
            body_direction = np.array(agent.body_direction)
            gaze_direction = np.array(agent.gaze_direction)
            view_radius = agent._view_radius
            focus = agent.focus

            height, width = display_image.shape[:2]
            is_within_bounds = 0 <= x < width and 0 <= y < height

            if is_within_bounds:
                cv2.circle(display_image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

                arrow_length = 20
                body_end_x = int(x + arrow_length * body_direction[0])
                body_end_y = int(y + arrow_length * body_direction[1])
                if agent.id == closest_agent.id:
                    cv2.arrowedLine(display_image, (x, y), (body_end_x, body_end_y), color=(128, 255, 0), thickness=2, tipLength=0.3)
                else:
                    cv2.arrowedLine(display_image, (x, y), (body_end_x, body_end_y), color=(0, 255, 0), thickness=2, tipLength=0.3)

                final_direction = body_direction + gaze_direction
                magnitude = np.linalg.norm(final_direction)
                if magnitude != 0:
                    final_direction /= magnitude

                area = view_radius * view_radius
                view_width = int(np.sqrt(2 * area * focus))
                view_width = max(view_width, 2)

                view_height = int(np.ceil(area / view_width))
                view_height = max(view_height, 2)

                rotation_angle = np.degrees(np.arctan2(final_direction[1], final_direction[0]))

                rect_points = cv2.boxPoints((
                    (x, y),
                    (view_width, view_height),
                    -rotation_angle
                ))
                rect_points = np.int32(rect_points)

                cv2.drawContours(display_image, [rect_points], 0, (255, 0, 0), 2)

        cv2.imshow(self.window_name, display_image)
        cv2.waitKey(1)

    def close(self) -> None:
        """
        Close the OpenCV window.
        """
        cv2.destroyWindow(self.window_name)
        self.is_rendering = False

    def start_rendering_loop(self, delay: int = 1) -> None:
        """
        Start a continuous rendering loop. This method is blocking.

        :param delay: Delay in milliseconds between frames.
        """
        self.is_rendering = True
        while self.is_rendering:
            self.render()
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

