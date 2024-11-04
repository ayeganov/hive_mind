from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar
import random

from neat import DefaultGenome
from numpy.typing import NDArray
import neat
import numpy as np


GenAgent = TypeVar('GenAgent')
GenContext = TypeVar('GenContext')


@dataclass
class EvaluationResult[GenAgent]:
    """Container for evaluation results"""
    agent: GenAgent
    behavior: NDArray[np.float32]
    additional_data: dict[str, Any]


class NoveltySearch:
    """Generic novelty search implementation"""

    def __init__(self,
                 k_nearest: int = 10,
                 archive_prob: float = 0.02,
                 min_novelty_score: float = 0.01,) -> None:
        self.k_nearest = k_nearest
        self.archive_prob = archive_prob
        self.min_novelty_score = min_novelty_score
        self.archive: list[NDArray[np.float32]] = []
        self.generation: int = 0

    def compute_novelty_score(self,
                              behavior: NDArray[np.float32],
                              behaviors: list[NDArray[np.float32]],) -> float:
        """Compute novelty score for a behavior"""
        if len(behaviors) < 2:
            return float('inf')

        distances = [float(np.linalg.norm(behavior - other)) for other in behaviors]
        distances.sort()
        k = min(self.k_nearest, len(distances)-1)
        return float(np.mean(distances[1:k+1]))

    def assign_novelty_scores(self,
                              genomes: list[tuple[int, DefaultGenome]],
                              behaviors: list[NDArray[np.float32]],) -> None:
        """Assign novelty scores to genomes and update archive"""
        all_behaviors = behaviors + self.archive

        for (_, genome), behavior in zip(genomes, behaviors):
            novelty_score = self.compute_novelty_score(behavior, all_behaviors)
            genome.fitness = novelty_score  # type: ignore

            is_archive_worthy = (random.random() < self.archive_prob
                                 and novelty_score > self.min_novelty_score)

            if is_archive_worthy:
                self.archive.append(behavior)

        self.generation += 1
        print(f"Generation {self.generation}: Archive size {len(self.archive)}")


class DomainAdapter[GenAgent, GenContext](ABC):
    """Abstract adapter for domain-specific logic"""

    @abstractmethod
    def setup_evaluation(self) -> GenContext:
        """Setup for evaluation, returns domain data needed for evaluation"""

    @abstractmethod
    def evaluate_agents(self,
                        genomes: list[tuple[int, DefaultGenome]],
                        config: neat.Config, 
                        domain_data: GenContext) -> list[EvaluationResult[GenAgent]]:
        """Evaluate a single genome and return its behavior characterization"""

    @abstractmethod
    def cleanup_evaluation(self, domain_data: GenContext) -> None:
        """Cleanup after evaluation"""

    @abstractmethod
    def is_search_completed(self) -> bool:
        """Have the search conditions been met"""
