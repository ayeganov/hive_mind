from dataclasses import dataclass
import numpy as np
from neat.genome import DefaultGenome
from neat.nn import FeedForwardNetwork


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
    """Manages a queue of genomes with fixed capacity"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue: list[tuple[int, DefaultGenome]] = []  # (id, genome)
        self.next_id = 0

    def add(self, genome: DefaultGenome) -> int:
        genome_id = self.next_id
        self.next_id += 1

        self.queue.append((genome_id, genome))
        if len(self.queue) > self.capacity:
            self.queue.pop(0)  # Remove oldest

        return genome_id

    def get_batch(self, size: int) -> list[tuple[int, DefaultGenome]]:
        return self.queue[:size]

    def __len__(self):
        return len(self.queue)


class MCCEvolution:
    """Core MCC implementation managing coevolution process"""
    
    def __init__(self, 
                 config: MCCConfig,
                 agent_config: dict,  # NEAT config for agents
                 env_config: dict):   # Config for environments
        self.config = config
        self.agent_config = agent_config
        self.env_config = env_config
        
        # Population queues
        self.agent_queue = PopulationQueue(config.agent_queue_size)
        self.env_queue = PopulationQueue(config.env_queue_size)
        
        # Resource tracking
        self.resource_tracker = ResourceTracker(config.resource_limit)
        
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
