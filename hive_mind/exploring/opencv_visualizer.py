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
        self.environment = None  # Instance of Environment
        self.agents: dict[str, Agent] = {}  # Maps agent IDs to Agent instances
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

    def get_agents(self) -> list[Agent]:
        """
        Retrieve the list of agents currently in the visualization.

        :return: A list of Agent instances.
        """
        return list(self.agents.values())

    def render(self) -> None:
        """
        Render the current state of the environment and agents using OpenCV.
        """
        if self.environment is None:
            print("No environment set for visualization.")
            return

        # Retrieve environment data
        env_data = self.environment.get_data()

        if not isinstance(env_data, np.ndarray):
            print("Environment data is not a valid image array.")
            return

        # Create a copy to draw on
        display_image = env_data.copy()

        # Overlay each agent
        for agent in self.agents.values():
            loc = agent.location
            x, y = int(loc.get('x', 0)), int(loc.get('y', 0))

            # Check if the agent is within the image boundaries
            height, width = display_image.shape[:2]
            if 0 <= x < width and 0 <= y < height:
                # Draw a circle for the agent
                cv2.circle(display_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

                # Put the agent's ID near the agent
                cv2.putText(
                    display_image,
                    agent.id,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )
            else:
                # Optionally, handle agents outside the boundaries
                print(f"Agent {agent.id} is outside the image boundaries.")

        # Display the image
        cv2.imshow(self.window_name, display_image)
        cv2.waitKey(1)  # Required to update the OpenCV window

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

