from abc import ABC, abstractmethod
from typing import TypeVar

from hive_mind.agent import Agent
from .environment import Environment


RenderContext = TypeVar('RenderContext')


class Visualizer[RenderContext](ABC):
    """
    Abstract base class defining the interface for a visualizer.
    """

    @abstractmethod
    def set_environment(self, environment: Environment) -> None:
        """
        Set or update the environment to be visualized.

        :param environment: An instance of Environment.
        """

    @abstractmethod
    def render(self, ctx: RenderContext) -> None:
        """
        Render the current state of the environment and agents.
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Fully clear the visualizer state
        """
