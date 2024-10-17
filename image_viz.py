import time
import uuid
from typing import Any, override

from neat.config import os
from neat.genome import DefaultGenome
from neat.population import Population
import numpy as np
import cv2
import matplotlib.pyplot as plt
import neat

from hive_mind.agent import Agent
from hive_mind.exploring.environment import Environment
from hive_mind.image_agent import ImageAgent
from hive_mind.exploring.opencv_visualizer import OpenCVVisualizer


class SimpleEnvironment(Environment):
    """
    A simple environment that provides a static image.
    """

    def __init__(self, image_path: str | np.ndarray | None = None) -> None:
        """
        Initialize the environment with an image.

        :param image_path: Path to the image file.
        """
        if image_path is None:
            self._image: np.ndarray = np.array([])
        elif isinstance(image_path, str):
            self._image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
            if self._image is None:
                raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        else:
            self._image = image_path

    @override
    def get_data(self) -> Any:
        """
        Retrieve the current environment image.

        :return: The environment image as a NumPy array.
        """
        return self._image

    @override
    def update_data(self, new_data: Any) -> None:
        """
        Update the environment image.

        :param new_data: The new image data as a NumPy array.
        """
        if isinstance(new_data, np.ndarray):
            self._image = new_data
        else:
            raise TypeError("Environment data must be a NumPy array representing an image.")

    @property
    @override
    def boundaries(self) -> tuple:
        return self._image.shape


def create_agents(goal: tuple[int, int], genomes: list[tuple[int, DefaultGenome]], neat_config: neat.Config, env: Environment) -> dict[ImageAgent, DefaultGenome]:
    """
    Create a list of dummy ImageAgents with random locations.

    :param num_agents: Number of agents to create.
    :param neat_config: NEAT configuration.
    :return: A list of ImageAgent instances.
    """
    agents = {}
    for _, genome in genomes:
        agent_id = str(uuid.uuid4())
        agent = ImageAgent(genome, neat_config, agent_id=agent_id, env=env)

        x, y = env.boundaries
        agent_x = x - goal[0]
        agent_y = y - goal[1]
        agent.location = {'x': agent_x, 'y': agent_y}
        agents[agent] = genome
    return agents


class ImageCrawlersSim:
    def __init__(self, area: tuple[int, int], epoch_sec: int, config: neat.Config) -> None:
        self._area = np.array(area)
        self._epoch_sec = epoch_sec
        self._environment = SimpleEnvironment()
        self._population = Population(config)
        self._goal = area
        self._stats = neat.StatisticsReporter()
        self._population.add_reporter(self._stats)
        self._population.add_reporter(neat.StdOutReporter(True))

    def start_sim(self) -> DefaultGenome | None:
        best_genome = self._population.run(self._fitness_eval, 50)
        return best_genome

    def _calc_fitness(self, agent: Agent) -> float:
        x_y = np.array((agent.location['x'], agent.location['y']))
        dist = np.sum((x_y - self._goal)**2)
        max_fitness = 100
        steepness = 0.01
        bonus = 1000

        fitness = max_fitness * np.exp(-steepness * dist)
        if dist < 5:
            fitness += bonus

        return fitness

    def _fitness_eval(self, genomes: list[tuple[int, DefaultGenome]], config: neat.Config) -> None:
        visualizer = OpenCVVisualizer(window_name="OpenCV Visualizer")
        visualizer.set_environment(self._environment)

        width, height = self._area
        goal_x, goal_y = np.random.uniform(0, width), np.random.uniform(0, height)
        self._goal = (int(goal_x), int(goal_y))
        hill_image = create_hill_image(width, height, goal_x, goal_y, method="inverse_quadratic", steepness=5)
        self._environment.update_data(hill_image)

        try:
            agents = create_agents(self._goal, genomes, config, self._environment)

            for agent, genome in agents.items():
                visualizer.add_agent(agent)
                genome.fitness = 0.0

            start = time.time()
            delta = time.time() - start

            while delta < self._epoch_sec:
                for agent, genome in agents.items():
                    agent.observe(self._environment.get_data())
                    agent.process()

                    genome.fitness += self._calc_fitness(agent)

                visualizer.render(self._goal)
                delta = time.time() - start
        finally:
            visualizer.close()


def create_hill_image(width, height, peak_x=None, peak_y=None, method='sigmoid', steepness=5):
    """
    Create a 2D image of a hill that spans the whole image.
    
    :param width: Width of the image
    :param height: Height of the image
    :param peak_x: X-coordinate of the peak (random if None)
    :param peak_y: Y-coordinate of the peak (random if None)
    :param method: 'sigmoid', 'quadratic', or 'inverse_quadratic'
    :param steepness: Controls the steepness of the hill
    :return: 2D numpy array representing the hill image
    """
    if peak_x is None:
        peak_x = np.random.uniform() * width
    if peak_y is None:
        peak_y = np.random.uniform() * height
    
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)
    
    # Calculate distances from each point to the peak
    distances = np.sqrt((x - peak_x)**2 + (y - peak_y)**2)
    
    # Normalize distances to [0, 1] range
    max_distance = np.sqrt(width**2 + height**2)
    normalized_distances = distances / max_distance
    
    if method == 'sigmoid':
        hill = 1 / (1 + np.exp(steepness * (normalized_distances - 0.5)))
    elif method == 'quadratic':
        hill = 1 - (normalized_distances ** 2)
        hill = np.maximum(hill, 0)  # Clip negative values to 0
    elif method == 'inverse_quadratic':
        hill = 1 / (1 + steepness * normalized_distances ** 2)
    else:
        raise ValueError("Invalid method. Choose 'sigmoid', 'quadratic', or 'inverse_quadratic'.")
    
    # Normalize to [0, 1] range
    hill = (hill - hill.min()) / (hill.max() - hill.min())

    hill_image = (hill * 255).astype(np.uint8)
    
    return hill_image

def plot_3d_hill(hill_image, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(0, hill_image.shape[1], 1)
    y = np.arange(0, hill_image.shape[0], 1)
    x, y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(x, y, hill_image, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()


def main():
#    width, height = 100, 80
#    hill_image = create_hill_image(width, height)
#
#    for method in ['sigmoid', 'quadratic', 'inverse_quadratic']:
#        hill_image = create_hill_image(width, height, method=method)
#        plot_3d_hill(hill_image, f'3D Visualization of Hill Image ({method})')
#
#    return
    area = 400, 400

    config_path = os.path.abspath("config")  # Replace with your NEAT config path
    neat_config = neat.Config(
        DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    sim = ImageCrawlersSim(area, 25, neat_config)
    best = sim.start_sim()


#    plt.imshow(hill_image, cmap='viridis')
#    plt.colorbar()
#    plt.title('Hill Image')
#    plt.show()

#    print('\nBest genome:\n{!s}'.format(best))


if __name__ == "__main__":
    main()
