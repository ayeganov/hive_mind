import time
import uuid
from collections import defaultdict
from typing import Any, override

from neat.config import os
from neat.genome import DefaultGenome
from neat.population import Population
import numpy as np
import cv2
import matplotlib.pyplot as plt
import neat
from numpy.typing import NDArray

from hive_mind.agent import Agent
from hive_mind.exploring.environment import Environment, HillEnvironment
from hive_mind.exploring.novelty import DomainAdapter, EvaluationResult, NoveltySearch
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


class HillClimbingAdapter(DomainAdapter[Agent, tuple[OpenCVVisualizer, tuple[int, int]]]):
    """Adapter for hill climbing domain"""

    def __init__(self, area: tuple[int, int], epoch_sec: int) -> None:
        self._area = area
        self._epoch_sec = epoch_sec
        self._environment = HillEnvironment(*area)

    def setup_evaluation(self) -> tuple[OpenCVVisualizer, tuple[int, int]]:
        """Setup visualization and environment"""
        visualizer = OpenCVVisualizer(window_name="OpenCV Visualizer")
        self._environment.generate_surface()
        visualizer.set_environment(self._environment)

        peak_pos = self._environment.get_peak_positions()[0]
        return visualizer, (int(peak_pos[0]), int(peak_pos[1]))

    def evaluate_agents(self,
                        genomes: list[tuple[int, DefaultGenome]],
                        config: neat.Config,
                        domain_data: tuple[OpenCVVisualizer, tuple[int, int]]) -> list[EvaluationResult[Agent]]:
        """Run agent evaluation and return behavior characterization"""
        visualizer, goal = domain_data

        agents = create_agents(goal, genomes, config, self._environment)
        for agent in agents:
            visualizer.add_agent(agent)

        start = time.time()
        delta = 0
        while delta < self._epoch_sec:
            for agent in agents:
                agent.observe(self._environment.get_data())
                agent.process()
            visualizer.render(goal)
            delta = time.time() - start

        results = []
        for agent in agents:
            position = np.array([agent.location['x'], agent.location['y']], dtype=np.float32)
            direction = np.array(agent.body_direction, dtype=np.float32)
            behavior = np.concatenate([position, direction])
            results.append(EvaluationResult(agent=agent, behavior=behavior, additional_data={'goal': goal}))

        return results

    def cleanup_evaluation(self, domain_data: tuple[OpenCVVisualizer, tuple[int, int]]) -> None:
        """Cleanup visualization"""
        visualizer, _ = domain_data
        visualizer.close()


class NoveltyHillClimber:
    """Main class for hill climbing with novelty search"""

    def __init__(self, area: tuple[int, int], epoch_sec: int, config: neat.Config) -> None:
        self._adapter = HillClimbingAdapter(area, epoch_sec)
        self._novelty_search = NoveltySearch(
            k_nearest=10,
            archive_prob=0.02,
            min_novelty_score=0.01
        )
        self._population = neat.Population(config)
        self._stats = neat.StatisticsReporter()
        self._population.add_reporter(self._stats)
        self._population.add_reporter(neat.StdOutReporter(True))

    def start_sim(self) -> DefaultGenome | None:
        """Start the simulation"""
        best_genome = self._population.run(self._evaluate, 50)
        return best_genome

    def _evaluate(self, genomes: list[tuple[int, DefaultGenome]], config: neat.Config) -> None:
        """Evaluation function that bridges domain adapter and novelty search"""
        domain_data = self._adapter.setup_evaluation()
        try:
            results = self._adapter.evaluate_agents(genomes, config, domain_data)
            behaviors: list[NDArray[np.float32]] = [r.behavior for r in results]

            self._novelty_search.assign_novelty_scores(genomes, behaviors)

        finally:
            self._adapter.cleanup_evaluation(domain_data)


def create_agents(goal: tuple[int, int],
                  genomes: list[tuple[int, DefaultGenome]],
                  neat_config: neat.Config,
                  env: Environment,) -> dict[ImageAgent, DefaultGenome]:
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


class FitnessCalculator:
    def __init__(self):
        self.goal: tuple[int, int] | None = None
        self._direction_history = defaultdict(list)
        self._history_size = 30  # Number of frames to consider (adjust based on your needs)
        self._spin_penalty_threshold = 0.5  # Threshold for detecting spinning behavior

    def _calc_fitness(self, agent: Agent) -> tuple[float, float, float]:
        # Distance fitness calculation (unchanged)
        x_y = np.array((agent.location['x'], agent.location['y']))
        dist = np.sum((x_y - self.goal)**2)
        max_fitness = 100
        steepness = 0.01

        distance_fitness = max_fitness * np.exp(-steepness * dist)

        # Direction fitness calculation
        agent_dir = np.array(agent.body_direction)
        target_dir = self.goal - x_y
        target_dir = target_dir / np.linalg.norm(target_dir)

        dot_product = np.dot(agent_dir, target_dir)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle)

        # Calculate instantaneous direction fitness
        direction_fitness = 0.0
        if angle_deg <= 180:
            direction_fitness = max_fitness * (np.cos(angle) + 1) / 2

        # Update direction history
        direction_history = self._direction_history[agent.id]
        direction_history.append(direction_fitness)
        if len(direction_history) > self._history_size:
            direction_history.pop(0)

        consistency_fitness = 0.0
        if len(direction_history) >= 3:

            variance = np.var(direction_history)
            max_possible_variance = max_fitness * max_fitness / 4  # Maximum theoretical variance
            normalized_variance = np.clip(variance / max_possible_variance, 0, 1)

            recent_avg = np.mean(direction_history[-10:]) if len(direction_history) >= 10 else np.mean(direction_history)
            normalized_avg = recent_avg / max_fitness  # 0 to 1 range

            # Combine into consistency metric
            stability = 1 - normalized_variance  # High when variance is low
            consistency_fitness = max_fitness * stability * normalized_avg

            # Optional: Add bonus for very stable good performance
#            if stability > 0.9 and normalized_avg > 0.9:
#                consistency_fitness += bonus * 0.5  # Half bonus for excellent consistency

        return distance_fitness, consistency_fitness


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
        self._fitness_calc = FitnessCalculator()

    def start_sim(self) -> DefaultGenome | None:
        best_genome = self._population.run(self._fitness_eval, 50)
        return best_genome

    def _calc_fitness(self, agent: Agent) -> tuple[float, float]:
        # Distance fitness calculation
        x_y = np.array((agent.location['x'], agent.location['y']))
        dist = np.sum((x_y - self._goal)**2)
        max_fitness = 100
        steepness = 0.01
#    bonus = 1000

        distance_fitness = max_fitness * np.exp(-steepness * dist)
#    if dist < 5:
#        distance_fitness += bonus

        # Direction fitness calculation
        agent_dir = np.array(agent.body_direction)
        target_dir = self._goal - x_y
        target_dir = target_dir / np.linalg.norm(target_dir)  # normalize

        # Calculate angle between vectors using dot product
        dot_product = np.dot(agent_dir, target_dir)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle)

        # Only award points if agent is facing within 180 degrees of target
        direction_fitness = 0
        if angle_deg <= 180:
            # Use cosine function to smoothly scale fitness from 1 (perfect alignment) to 0 (180 degrees)
            direction_fitness = max_fitness * (np.cos(angle) + 1) / 2

        return distance_fitness, direction_fitness

    def _fitness_eval(self, genomes: list[tuple[int, DefaultGenome]], config: neat.Config) -> None:
        visualizer = OpenCVVisualizer(window_name="OpenCV Visualizer")
        visualizer.set_environment(self._environment)

        width, height = self._area
        goal_x, goal_y = np.random.uniform(0, width), np.random.uniform(0, height)
        self._goal = (int(goal_x), int(goal_y))
        self._fitness_calc.goal = self._goal
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

                    genome.fitness += sum(self._fitness_calc._calc_fitness(agent))

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


def plot_3d_hill(hill_image: np.ndarray, title: str, peaks: list[tuple[int, int]],) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, hill_image.shape[1], 1)
    y = np.arange(0, hill_image.shape[0], 1)
    x, y = np.meshgrid(x, y)

    surf = ax.plot_surface(x, y, hill_image, cmap='viridis', edgecolor='none')

    for peak_x, peak_y in peaks:
        # Get the height at the peak location
        peak_z = hill_image[peak_y, peak_x]  # Note: y,x order for array indexing
        marker_z = peak_z + 0.01 * np.max(hill_image)
        ax.scatter([peak_x], [peak_y], [marker_z],
                   color='red',
                   marker='o',
                   zorder=1000,
                   label='Peak',)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()


def main():

    width, height = 200, 200
#    config_path = os.path.abspath("config")  # Replace with your NEAT config path
#    neat_config = neat.Config(
#        DefaultGenome,
#        neat.DefaultReproduction,
#        neat.DefaultSpeciesSet,
#        neat.DefaultStagnation,
#        config_path
#    )
#    nov_hill_climber = NoveltyHillClimber((width, height), 8, neat_config)
#    winner = nov_hill_climber.start_sim()
#
#    print(f"{winner=}")
#    return

    hill_env = HillEnvironment(width, height)
    hill_env.complexity = 1

    while True:
        peaks = hill_env.get_peak_positions()
        hill_image = hill_env.get_data()

        plot_3d_hill(hill_image, f'3D Visualization of Hill Image', peaks)
        hill_env.generate_surface()

    return

    area = 400, 400

    sim = ImageCrawlersSim(area, 8, neat_config)
    best = sim.start_sim()


#    plt.imshow(hill_image, cmap='viridis')
#    plt.colorbar()
#    plt.title('Hill Image')
#    plt.show()

#    print('\nBest genome:\n{!s}'.format(best))


if __name__ == "__main__":
    main()
