import uuid
from typing import Any

from neat.config import os
from neat.population import Population
import numpy as np
import cv2
import neat

from hier_neat.exploring.environment import Environment
from hier_neat.image_agent import ImageAgent
from hier_neat.exploring.opencv_visualizer import OpenCVVisualizer


class SimpleEnvironment(Environment):
    """
    A simple environment that provides a static image.
    """

    def __init__(self, image_path: str):
        """
        Initialize the environment with an image.

        :param image_path: Path to the image file.
        """
        self.image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    def get_data(self) -> Any:
        """
        Retrieve the current environment image.

        :return: The environment image as a NumPy array.
        """
        return self.image

    def update_data(self, new_data: Any) -> None:
        """
        Update the environment image.

        :param new_data: The new image data as a NumPy array.
        """
        if isinstance(new_data, np.ndarray):
            self.image = new_data
        else:
            raise TypeError("Environment data must be a NumPy array representing an image.")


def create_dummy_agents(num_agents: int, neat_config: neat.Config) -> list[ImageAgent]:
    """
    Create a list of dummy ImageAgents with random locations.

    :param num_agents: Number of agents to create.
    :param neat_config: NEAT configuration.
    :return: A list of ImageAgent instances.
    """
    population = Population(neat_config)
    agents = []
    for genome in population.population.values():
        # Create a dummy genome (this should be replaced with actual genomes during evolution)
        agent_id = str(uuid.uuid4())
        agent = ImageAgent(config={'genome': genome, 'config': neat_config}, agent_id=agent_id)
        # Assign random initial locations within the image boundaries (e.g., 200x200)
        agent.location = {'x': np.random.randint(0, 200), 'y': np.random.randint(0, 200)}
        agents.append(agent)
    return agents


def main():
    image_path = "assets/sky_cloud.jpeg"

    # Initialize the environment
    environment = SimpleEnvironment(image_path=image_path)

    # Initialize the visualizer
    visualizer = OpenCVVisualizer(window_name="OpenCV Visualizer")
    visualizer.set_environment(environment)

    # Load NEAT configuration (ensure the config file exists and is correctly set up)
    config_path = os.path.abspath("config")  # Replace with your NEAT config path
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create and add agents to the visualizer
    num_agents = 5  # Number of agents to visualize
    agents = create_dummy_agents(num_agents, neat_config)
    for agent in agents:
        visualizer.add_agent(agent)

    # Start the rendering loop (this will block until 'q' is pressed)
    visualizer.start_rendering_loop(delay=30)  # Adjust delay as needed (milliseconds)

    # Cleanup after rendering loop ends
    visualizer.close()


if __name__ == "__main__":
    main()

