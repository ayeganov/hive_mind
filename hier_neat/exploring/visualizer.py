from abc import ABC, abstractmethod

from hier_neat.agent import Agent
from .environment import Environment


class Visualizer(ABC):
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
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the visualization.

        :param agent: An instance of Agent.
        """

    @abstractmethod
    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the visualization based on its ID.

        :param agent_id: The unique identifier of the agent to remove.
        """

    @abstractmethod
    def render(self) -> None:
        """
        Render the current state of the environment and agents.
        """

    @abstractmethod
    def get_agents(self) -> list[Agent]:
        """
        Retrieve the list of agents currently in the visualization.

        :return: A list of Agent instances.
        """
