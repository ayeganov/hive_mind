import copy
import time
import uuid
from collections import defaultdict
from typing import Any, override

from neat.config import os
from neat.genome import DefaultGenome
from neat.population import Population
import numpy as np
import cv2
import neat
import pyvista as pv
from numpy.typing import NDArray

from hive_mind.agent import Agent
from hive_mind.exploring.environment import Environment, HillEnvironment
from hive_mind.exploring.mcc import ResourceTracker
from hive_mind.exploring.novelty import DomainAdapter, EvaluationResult, NoveltySearch
from hive_mind.image_agent import ImageAgent
from hive_mind.exploring.opencv_visualizer import OpenCVHillClimberVisualizer, RenderAgents


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


class HillClimbingAdapter(DomainAdapter[Agent, OpenCVHillClimberVisualizer]):
    """Adapter for hill climbing domain"""

    def __init__(self,
                 area: tuple[int, int],
                 epoch_sec: int,
                 num_landscapes: int,) -> None:
        self._area = area
        self._epoch_sec = epoch_sec
        self._num_landscapes = num_landscapes
        self._environments = [HillEnvironment(*area, complexity=2) for _ in range(num_landscapes)]
        self._seed_agents: set[DefaultGenome] = set()
        self._seed_agent_ids: set[str] = set()
        self._seed_landscapes: set[Environment] = set()
        self._resource_trackers = [ResourceTracker(5) for _ in range(num_landscapes)]

    def setup_evaluation(self) -> OpenCVHillClimberVisualizer:
        """Setup visualization and environment"""
        visualizer = OpenCVHillClimberVisualizer(window_name="OpenCV Visualizer")
        return visualizer

    def _is_at_peak(self, goal: tuple[int, int], agent: Agent) -> tuple[bool, float]:
        x_y = np.array((agent.location['x'], agent.location['y']))
        dist = float(np.sqrt(np.sum((x_y - goal)**2)))
        return (bool(dist <= 20), dist)

    def is_search_completed(self) -> bool:
        num_seed_agents = len(self._seed_agents)
        num_seed_landscapes = len(self._seed_landscapes)
        print(f"Num agents: {num_seed_agents}, num solved landscapes: {num_seed_landscapes}")
        return num_seed_agents >= 20 and num_seed_landscapes == len(self._environments)

    def evaluate_agents(self,
                        genomes: list[tuple[int, DefaultGenome]],
                        config: neat.Config,
                        domain_data: OpenCVHillClimberVisualizer) -> list[EvaluationResult[Agent]]:
        """Run agent evaluation and return behavior characterization"""
        visualizer = domain_data

        agents = create_agents(genomes, config, self._environments[0])
        render_ctx = RenderAgents(agents=agents.keys())

        agent_behaviors: dict[Agent, NDArray[np.float32]] = {}
        for env, tracker in zip(self._environments, self._resource_trackers):

            render_ctx.peaks = env.get_peak_positions()

            visualizer.set_environment(env)
            goal = env.get_peak_positions()[0]

            set_agent_locations(agents, goal, env)

            start = time.time()
            delta = 0
            round_is_over = False
            while delta < self._epoch_sec and not round_is_over:
                render_ctx.dist = float("inf")
                for agent, genome in agents.items():
                    agent.observe(env.get_data())
                    agent.process()

                    if tracker.is_at_limit():
                        continue

                    has_reached_peak, dist = self._is_at_peak(goal, agent)
                    if dist < render_ctx.dist:
                        render_ctx.closest_id = agent.id
                        render_ctx.dist = dist

                    if has_reached_peak:
                        if agent.id not in self._seed_agent_ids:
                            self._seed_agents.add(copy.deepcopy(genome))
                            self._seed_agent_ids.add(agent.id)
                            self._seed_landscapes.add(env)

                            print(f"Agent {agent.id} has been added to seeds")
                            print(f"Landscape added to seeds")

                            tracker.record_usage(agent.id)
                            if tracker.is_at_limit():
                                print(f"The landscape has been solved 5 times, moving on to the next one")
                                round_is_over = True
                                break

                visualizer.render(render_ctx)
                delta = time.time() - start

            for agent in agents:
                position = np.array([agent.location['x'], agent.location['y']], dtype=np.float32) / 255.
                height = np.array([env.get_height(position[0], position[1])], dtype=np.float32)
                current_behavior = np.concatenate([position, height])
                all_behaviors = agent_behaviors.get(agent, np.array([], dtype=np.float32))
                all_behaviors = np.concatenate([all_behaviors, current_behavior])
                agent_behaviors[agent] = all_behaviors

        results = []
        for agent, behavior in agent_behaviors.items():
            assert len(behavior) == self._num_landscapes * 3, f"Behavior of invalid size: {len(behavior)=}"
            results.append(EvaluationResult(agent=agent, behavior=behavior))

        return results

    def cleanup_evaluation(self,
                           domain_data: OpenCVHillClimberVisualizer,) -> None:
        """Cleanup visualization"""
        visualizer = domain_data
        visualizer.close()


class NoveltyHillClimber:
    """Main class for hill climbing with novelty search"""

    def __init__(self, area: tuple[int, int], epoch_sec: int, config: neat.Config) -> None:
        self._adapter = HillClimbingAdapter(area, epoch_sec, 10)
        self._novelty_search = NoveltySearch(
            k_nearest=10,
            archive_prob=0.08,
            min_novelty_score=0.01
        )
        self._population = neat.Population(config)
        self._stats = neat.StatisticsReporter()
        self._population.add_reporter(self._stats)
        self._population.add_reporter(neat.StdOutReporter(True))

    def start_sim(self) -> DefaultGenome | None:
        """Start the simulation"""
        while not self._adapter.is_search_completed():
            self._population.run(self._evaluate, 1)

    def _evaluate(self, genomes: list[tuple[int, DefaultGenome]], config: neat.Config) -> None:
        """Evaluation function that bridges domain adapter and novelty search"""
        domain_data = self._adapter.setup_evaluation()
        try:
            results = self._adapter.evaluate_agents(genomes, config, domain_data)
            behaviors: list[NDArray[np.float32]] = [r.behavior for r in results]

            self._novelty_search.assign_novelty_scores(genomes, behaviors)

        finally:
            self._adapter.cleanup_evaluation(domain_data)


def create_agents(genomes: list[tuple[int, DefaultGenome]],
                  neat_config: neat.Config,
                  env: Environment,) -> dict[Agent, DefaultGenome]: 
    """
    Create a list of dummy ImageAgents with location 0, 0

    :param genomes: 
    :param neat_config: NEAT configuration.
    :return: A list of ImageAgent instances.
    """
    agents = {}
    for _, genome in genomes:
        agent_id = str(uuid.uuid4())
        agent = ImageAgent(genome, neat_config, agent_id=agent_id, env=env)
        agents[agent] = genome

    return agents


def set_agent_locations(agents: dict[Agent, DefaultGenome],
                        goal: tuple[int, int],
                        env: Environment,) -> None:
    for agent in agents:
        x, y = env.boundaries
        agent_x = x - goal[0]
        agent_y = y - goal[1]
        agent.location = {'x': agent_x, 'y': agent_y}


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
        visualizer = OpenCVHillClimberVisualizer(window_name="OpenCV Visualizer")
        visualizer.set_environment(self._environment)

        width, height = self._area
        goal_x, goal_y = np.random.uniform(0, width), np.random.uniform(0, height)
        self._goal = (int(goal_x), int(goal_y))
        self._fitness_calc.goal = self._goal
        hill_image = create_hill_image(width, height, goal_x, goal_y, method="inverse_quadratic", steepness=5)
        self._environment.update_data(hill_image)

        try:
            agents = create_agents(genomes, config, self._environment)

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



def plot_3d_hill(hill_image: np.ndarray, title: str, peaks: list[tuple[int, int]]) -> None:
    x = np.arange(0, hill_image.shape[1], 1)
    y = np.arange(0, hill_image.shape[0], 1)
    x, y = np.meshgrid(x, y)

    grid = pv.StructuredGrid(x, y, hill_image)
    grid["heights"] = hill_image.flatten()

    plotter = pv.Plotter()

    plotter.add_mesh(grid, scalars="heights", cmap="viridis",)

    peak_points = []
    for peak_x, peak_y in peaks:

        peak_x_int = np.clip(peak_x, 0, hill_image.shape[1] - 1)
        peak_y_int = np.clip(peak_y, 0, hill_image.shape[0] - 1)

        peak_z = hill_image[peak_y_int, peak_x_int]
        marker_z = peak_z + 0.01 * np.max(hill_image)

        peak_points.append([peak_x, peak_y, marker_z])

    if peak_points:
        points = pv.PolyData(np.array(peak_points))
        plotter.add_mesh(points, color="red", point_size=20)

    plotter.show_axes()
    plotter.add_axes(
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=2,
        labels_off=False,
        color='black',
    )

    plotter.camera_position = 'iso'

    plotter.add_title(title)

    plotter.show()


def main():

    width, height = 200, 200
    config_path = os.path.abspath("config")  # Replace with your NEAT config path
    neat_config = neat.Config(
        DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    nov_hill_climber = NoveltyHillClimber((width, height), 8, neat_config)
    winner = nov_hill_climber.start_sim()

    print(f"{winner=}")
    return

#    hill_env = HillEnvironment(width, height)
#    hill_env.complexity = 4
#
#    while True:
#        hill_env.generate_surface()
#        peaks = hill_env.get_peak_positions()
#        hill_image = hill_env.get_data()
#
#        plot_3d_hill(hill_image, f'3D Visualization of Hill Image', peaks)
#
#    return

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
