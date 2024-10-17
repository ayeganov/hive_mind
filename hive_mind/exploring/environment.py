from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    """
    Abstract base class defining the interface for an environment.
    """

    @abstractmethod
    def get_data(self) -> Any:
        """
        Retrieve the current environment data.

        :return: The environment data (e.g., image, audio).
        """

    @abstractmethod
    def update_data(self, new_data: Any) -> None:
        """
        Update the environment data.

        :param new_data: The new environment data.
        """

    @property
    @abstractmethod
    def boundaries(self) -> tuple:
        """
        Return the boundaries of this environment
        """
