from abc import ABC, abstractmethod
from typing import Any, Iterable

from torch import Tensor


class Entity(ABC):
    """
    Something that can exist in space and can be identified
    """
    @property
    @abstractmethod
    def id(self) -> str:
        """
        Unique id for this particular entity
        """

    @property
    @abstractmethod
    def type(self) -> str:
        """
        Type of this entity - agent, environment, genome etc
        """


class Agent(Entity, ABC):
    """
    Abstract base class defining the simplified interface for an AI agent.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any], agent_id: str):
        """
        Initialize the agent with a given configuration and unique identifier.

        :param config: A dictionary of configuration parameters.
        :param agent_id: A unique identifier for the agent.
        """

    @abstractmethod
    def observe(self, input_data: Any) -> None:
        """
        Consume a portion of the incoming input.

        :param input_data: The data received from the environment.
        """

    @abstractmethod
    def process(self) -> Tensor:
        """
        Process the observed data and output a tensor representing the agent's
        decision or internal state.

        :return: A tensor representing the agent's output.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the agent's internal state to its initial configuration.
        """

    @property
    @abstractmethod
    def state(self) -> dict[str, Any]:
        """
        Get the current internal state of the agent.

        :return: A dictionary representing the agent's state.
        """

    @state.setter
    @abstractmethod
    def state(self, new_state: dict[str, Any]) -> None:
        """
        Set the agent's internal state.

        :param new_state: A dictionary representing the new state.
        """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Get the unique identifier of the agent.

        :return: The agent's unique ID.
        """

    @property
    @abstractmethod
    def location(self) -> dict[str, float]:
        """
        Get the current location of the agent.

        :return: A dictionary representing the agent's location (e.g., {'x': 0.0, 'y': 0.0}).
        """

    @location.setter
    @abstractmethod
    def location(self, new_location: dict[str, float]) -> None:
        """
        Set the agent's current location.

        :param new_location: A dictionary representing the new location.
        """

    @property
    @abstractmethod
    def body_direction(self) -> Iterable[float]:
        """
        Get the agent's current body direction
        """

    @property
    @abstractmethod
    def gaze_direction(self) -> Iterable[float]:
        """
        Get the agents gaze direction
        """

    @property
    @abstractmethod
    def focus(self) -> float:
        """
        Get the agents current attention focus
        """
