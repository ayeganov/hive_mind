from abc import abstractmethod
from collections import defaultdict
from typing import TypeVar, override
import itertools
import hashlib
import uuid

from dataclasses import dataclass
from neat.genome import DefaultGenome
from neat.nn import FeedForwardNetwork

from hive_mind.agent import Entity
from hive_mind.exploring.environment import Environment


class EvolvingEntity(Entity):

    @property
    @abstractmethod
    def age(self) -> int:
        """
        Return number of generations this entity has been around
        """

    @property
    @abstractmethod
    def parents(self) -> tuple[str]:
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
    agent_queue_size: int = 250
    env_queue_size: int = 50
    resource_limit: int = 5  # How many times an env can be used
    batch_size: int = 40  # Number of agents evaluated per batch
    env_batch_size: int = 10  # Number of envs evaluated per batch

 
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

    def add_batch(self, entities: list[EvolvingEntity]) -> None:
        for ent in entities:
            self._entities_by_type[ent.type].append(ent)
            self._queue.append(ent)

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


class GenomeEntity(EvolvingEntity):
    def __init__(self, genome: DefaultGenome, parent_ids: tuple[str]) -> None:
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
    def parents(self) -> tuple[str]:
        return self._parents

    @override
    def grow_older(self) -> None:
        self._age += 1

    @property
    @override
    def needed(self) -> list[str]:
        return ["env"]


class MCCEvolution:
    """Core MCC implementation managing coevolution process"""

    def __init__(self,
                 config: MCCConfig,
                 seed_agents: set[DefaultGenome],
                 seed_envs: set[Environment]) -> None:
        self._config = config
        self._viable_population = PopulationQueue(config.agent_queue_size)

        for entity in itertools.chain(seed_agents, seed_envs):
            self._viable_population.add(entity)

        self._resource_tracker = ResourceTracker(config.resource_limit)

    def run_evolution(self) -> None:
        """
        Main loop running the evolution algorithm
        """
        parents = self._viable_population.get_batch(self._config.batch_size)
        children = self._reproduce(parents)

        # TODO: create pairs of child and env and then run them in parallel
        for idx, child in enumerate(children):
            pass

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

        raise RuntimeError("Unexpected error, must be able to always find next match")

    def _reproduce(self, parents: list[Entity]) -> list[Entity]:
        return []

    def evaluate_agent(self, 
                      agent_genome: DefaultGenome, 
                      envs: list[tuple[int, DefaultGenome]]) -> bool:
        """Evaluate agent against available environments"""
        agent_net = FeedForwardNetwork.create(agent_genome, self.agent_config)
        
        for env_id, env_genome in envs:
            if not self.resource_tracker.can_use(env_id):
                continue
                
            # Try to solve environment
            success = self._attempt_solve(agent_net, env_genome)
            if success:
                self.resource_tracker.record_usage(env_id)
                return True
                
        return False
    
    def _attempt_solve(self, 
                      agent_net: FeedForwardNetwork, 
                      env_genome: DefaultGenome) -> bool:
        """Run single evaluation of agent in environment"""
        # Implementation depends on specific environment type
        raise NotImplementedError
    
    def step(self) -> tuple[int, int]:
        """Run one batch of evaluations"""
        # Get batch of agents and environments
        agent_batch = self.agent_queue.get_batch(self.config.batch_size)
        env_batch = self.env_queue.get_batch(self.config.env_batch_size)
        
        if not agent_batch or not env_batch:
            return 0, 0
            
        # Clear resource usage for this batch
        self.resource_tracker.clear()
        
        # Evaluate and add successful offspring
        agents_added = 0
        envs_added = 0
        
        # Evaluate agents
        for _, agent_genome in agent_batch:
            # Create offspring through mutation
            child = agent_genome.copy()
            child.mutate(self.agent_config)
            
            # Check if meets minimal criterion
            if self.evaluate_agent(child, env_batch):
                self.agent_queue.add(child)
                agents_added += 1
                
        # Evaluate environments (similar process)
        for _, env_genome in env_batch:
            child = env_genome.copy()
            child.mutate(self.env_config)
            # Environment evaluation would go here
            # Add if meets minimal criterion
            
        return agents_added, envs_added
