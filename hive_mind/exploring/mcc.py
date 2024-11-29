from abc import abstractmethod
from collections import defaultdict
from typing import Iterable, Iterator, TypeVar, cast, override
import hashlib
import itertools
import random
import time
import uuid

from dataclasses import dataclass
from neat.config import Config
from neat.genome import DefaultGenome
from neat.nn import FeedForwardNetwork
import numpy as np

from hive_mind.agent import Agent, Entity
from hive_mind.exploring.environment import Environment, Peak, Terrain
from hive_mind.exploring.gene import TerrainGene
from hive_mind.image_agent import ImageAgent


class EvolvingEntity(Entity):

    @property
    @abstractmethod
    def age(self) -> int:
        """
        Return number of generations this entity has been around
        """

    @property
    @abstractmethod
    def parents(self) -> tuple[str, str]:
        """
        Return ids of the parents of this entity
        """

    @abstractmethod
    def grow_older(self) -> None:
        """
        Update time related properties of this entity
        """

    @property
    @abstractmethod
    def needed(self) -> list[str]:
        """
        Return the list of other entities that this entity need to attempt a solution
        """


@dataclass
class MCCConfig:
    """Configuration parameters for MCC"""
    pop_queue_size: int = 250
    resource_limit: int = 5  # How many times an env can be used
    batch_size: int = 40  # Number of agents evaluated per batch
    epoch_time: float = 8

 
class ResourceTracker:
    """Tracks resource usage for environments"""
    def __init__(self, limit: int):
        self._limit = limit
        self._usage: dict[str, int] = {}  # ent_id -> times_used

    def can_use(self, ent_id: str) -> bool:
        return self._usage.get(ent_id, 0) < self._limit

    def is_at_limit(self) -> bool:
        return sum(self._usage.values()) >= self._limit

    def record_usage(self, ent_id: str):
        self._usage[ent_id] = self._usage.get(ent_id, 0) + 1

    def clear(self):
        self._usage.clear()


class PopulationQueue:
    """Manages a queue of entities with fixed capacity"""
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._queue: list[EvolvingEntity] = list()
        self._entities_by_type = defaultdict(list)

    def add(self, entity: EvolvingEntity) -> None:
        self._entities_by_type[entity.type].append(entity)
        self._queue.append(entity)

    @property
    def current_population(self) -> list[EvolvingEntity]:
        return self._queue

    def get_sub_population(self, ent_type: str) -> list[EvolvingEntity]:
        return self._entities_by_type[ent_type]

    def add_batch(self, entities: Iterable[EvolvingEntity]) -> None:
        for ent in entities:
            self._entities_by_type[ent.type].append(ent)
            self._queue.append(ent)

    def count_generation(self) -> None:
        print("Increasing age of every member of the population by 1...")
        for ent in self._queue:
            ent.grow_older()

    def remove_oldest(self) -> list[EvolvingEntity]:
        """
        Removes the oldest entities in the population. Updates the internal
        queue, and returned the list of removed.
        """
        num_to_remove = len(self._queue) - self._capacity
        sorted_ents = sorted(self._queue, reverse=True, key=lambda it: it.age)
        removed, self._queue = sorted_ents[:num_to_remove], sorted_ents[num_to_remove:]

        for ent in removed:
            if ent.type in self._entities_by_type:
                ents = self._entities_by_type[ent.type]
                ents.remove(ent)

        return removed

    def get_batch(self, size: int) -> list[EvolvingEntity]:
        batch, rest = self._queue[:size], self._queue[size:]
        self._queue = rest
        return batch

    def __len__(self):
        return len(self._queue)


def create_uuid_from_string_hashlib(input_string: str) -> uuid.UUID:
    hash_object = hashlib.sha1(input_string.encode())
    hex_dig = hash_object.hexdigest()
    return uuid.UUID(hex=hex_dig[:32])


class AgentGenomeEntity(EvolvingEntity):
    def __init__(self, genome: DefaultGenome, parent_ids: tuple[str, str]) -> None:
        self._genome = genome
        self._id: str = str(create_uuid_from_string_hashlib(str(self._genome.key)))
        self._age: int = 0
        self._parents = parent_ids

    @property
    @override
    def id(self) -> str:
        return self._id

    @property
    def type(self) -> str:
        return "agent"

    @property
    @override
    def age(self) -> int:
        return self._age

    @property
    @override
    def parents(self) -> tuple[str, str]:
        return self._parents

    @override
    def grow_older(self) -> None:
        self._age += 1

    @property
    @override
    def needed(self) -> list[str]:
        return ["env"]

    @property
    def genome(self) -> DefaultGenome:
        return self._genome


class EnvEntity(EvolvingEntity):
    def __init__(self, genome: TerrainGene, parent_ids: tuple[str, str]) -> None:
        self._genome = genome
        self._parents = parent_ids
        self._age: int = 0

    @property
    @override
    def id(self) -> str:
        return self._genome.id

    @property
    def type(self) -> str:
        return "env"

    @property
    @override
    def age(self) -> int:
        return self._age

    @property
    @override
    def parents(self) -> tuple[str, str]:
        return self._parents

    @override
    def grow_older(self) -> None:
        self._age += 1

    @property
    @override
    def needed(self) -> list[str]:
        return ["agent"]

    @property
    def genome(self) -> TerrainGene:
        return self._genome


class MCCEvolution:
    """Core MCC implementation managing coevolution process"""

    def __init__(self,
                 config: MCCConfig,
                 neat_config: Config,
                 seed_agents: set[DefaultGenome],
                 seed_envs: set[TerrainGene]) -> None:
        self._config = config
        self._neat_config = neat_config
        self._viable_population = PopulationQueue(config.pop_queue_size)
        self._genome_indexer = itertools.count(len(seed_agents))
        self._generation = 1

        evolving_agents = (AgentGenomeEntity(g, tuple()) for g in seed_agents)
        evolving_envs = (EnvEntity(g, tuple()) for g in seed_envs)

        self._viable_population.add_batch(itertools.chain(evolving_agents, evolving_envs))

        self._resource_tracker = ResourceTracker(config.resource_limit)

    def run_evolution(self) -> None:
        """
        Main loop running the evolution algorithm
        """
        while True:
            print(f"Starting generation {self._generation}...")
            parents = self._viable_population.get_batch(self._config.batch_size)
            children = self._reproduce(parents)
            self._viable_population.add_batch(parents)

            # TODO: create pairs of child and env and then run them in parallel
            # using multiproc or 3.13 multithread?
            eval_agent = None
            eval_env = None
            for idx, child in enumerate(children):
                if child.type == "agent":
                    eval_agent = child
                    avaialble_envs = self._viable_population.get_sub_population("env")
                    for env in avaialble_envs:
                        if self._resource_tracker.can_use(env.id):
                            eval_env = env
                            break
                else: # child is environment
                    eval_env = child
                    available_agents = self._viable_population.get_sub_population("agent")
                    eval_agent = self._next(available_agents, child.needed, idx)

                if eval_agent is None or eval_env is None:
                    print("Warning: failed to match a pair of agent-env")
                    continue

                mc_satisfied = self._evaluate_agent(cast(AgentGenomeEntity, eval_agent), cast(EnvEntity, eval_env))

                if mc_satisfied:
                    self._viable_population.add(child)
                eval_agent = None
                eval_env = None
            self._viable_population.count_generation()
            removed = self._viable_population.remove_oldest()
            print(f"After generation {self._generation} removed {len(removed)} members of the population")
            self._generation += 1

    def _next(self, entities: list[EvolvingEntity], needed: list[str], idx: int) -> EvolvingEntity:
        """
        TODO: This is a build pipeline of the hierarchy being tested. Need a
        way to express preferences to what `next` entity should be.
        """
        need_type = needed[0]
        for i in range(idx + 1, len(entities)):
            ent = entities[i]
            if ent.type == need_type:
                return ent
        raise RuntimeError("Unexpected error, must be able to find next match")

    def _reproduce(self, parents: list[EvolvingEntity]) -> list[EvolvingEntity]:
        print(f"Reprodcuing from {len(parents)} parents")
        children = []
        sorted_parents = sorted(parents, key=lambda it: it.type)
        for ent_type, group in itertools.groupby(sorted_parents, key=lambda it: it.type):
            if ent_type == "env":
                children.extend(self._reproduce_envs(list(group), 10))
            else:
                children.extend(self._reproduce_agents(list(group), 40))
        return children

    def _reproduce_agents(self, parents: list[EvolvingEntity], num_children: int) -> list[EvolvingEntity]:
        genomes: list[AgentGenomeEntity] = cast(list[AgentGenomeEntity], parents)
        children: list[EvolvingEntity] = []

        config = self._neat_config

        while len(children) < num_children:
            parent1, parent2 = random.sample(genomes, 2)
            parent1_genome = parent1.genome
            parent2_genome = parent2.genome

            child_genome = DefaultGenome(key=next(self._genome_indexer))

            child_genome.configure_crossover(
                parent1_genome,
                parent2_genome,
                config.genome_config,
            )

            child_genome.mutate(config.genome_config)

            child = AgentGenomeEntity(
                genome=child_genome,
                parent_ids=(parent1.id, parent2.id)
            )

            children.append(child)

        return children

    def _reproduce_envs(self, parents: list[EvolvingEntity], num_children: int) -> list[EvolvingEntity]:
        genomes: list[EnvEntity] = cast(list[EnvEntity], parents)
        children: list[EvolvingEntity] = []

        while len(children) < num_children:
            parent1, parent2 = random.sample(genomes, 2)

            child_genome = parent1.genome.crossover(parent2.genome)
            child_genome.mutate(0.05)

            child = EnvEntity(genome=child_genome, parent_ids=(parent1.id, parent2.id),)

            children.append(child)

        return children

    def _is_at_peak(self, goal: Peak, agent: Agent) -> tuple[bool, float]:
        x_y = np.array((agent.location['x'], agent.location['y']))
        goal_xy = np.array((goal.x, goal.y))
        dist = float(np.sqrt(np.sum((x_y - goal_xy)**2)))
        return (bool(dist <= 20), dist)

    def _evaluate_agent(self, 
                        agent_ent: AgentGenomeEntity, 
                        env_ent: EnvEntity) -> bool:
        """Evaluate agent against available environments"""
        agent_genome = agent_ent.genome
        env_genome = env_ent.genome
        env = Terrain.from_genes(env_genome)
        agent = ImageAgent(agent_genome, self._neat_config, env, agent_ent.id)
        min_idx = np.unravel_index(np.argmin(env.get_data("float32")), env.get_data().shape)
        y_min, x_min, *_ = min_idx
        agent.location = {"x": int(x_min), "y": int(y_min)}

        goal = env.peaks[0]
        start = time.time()
        delta = 0
        round_is_over = False
        peak_reached = False
        while delta < self._config.epoch_time and not round_is_over:
            agent.observe(env.get_data())
            agent.process()

            if not peak_reached:
                peak_reached, dist = self._is_at_peak(goal, agent)

            delta = time.time() - start

        return peak_reached
