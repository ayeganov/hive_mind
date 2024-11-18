from typing import Any, Iterable, override
import uuid

import cv2
import neat
import numpy as np
import torch

from hive_mind.exploring.environment import Environment

from .agent import Agent


class ImageAgent(Agent):
    """
    An agent that observes a portion of an image and moves based on NEAT network outputs.
    """

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, env: Environment, agent_id: str | None = None,) -> None:
        """
        Initialize the ImageAgent with a configuration and unique identifier.

        :param config: A dictionary containing NEAT genome and configuration.
        :param agent_id: Optional unique identifier for the agent. If not provided, a UUID is generated.
        """
        self.config = config
        self._env = env
        self._internal_state = {}
        self._rotation = 0.0
        self._body_direction = np.array([np.random.random(), np.random.random()]) * np.pi * 2
        self._gaze_direction = 0 #np.array([np.random.random(), np.random.random()]) * np.pi
        self._focus = 0.5

        self._id = agent_id if agent_id is not None else str(uuid.uuid4())

        self._network = neat.nn.FeedForwardNetwork.create(genome, config)

        self._location = {'x': 0.0, 'y': 0.0}
        self._view_radius = int(config.genome_config.num_inputs**0.5)
        self._add_noise = False

    def observe(self, input_data: np.ndarray) -> None:
        """
        Observe a rectangular portion of the image based on current location, body direction, gaze direction, and focus.

        :param input_data: The full image as a NumPy array.
        """
        image = input_data
        height, width = image.shape[:2]
        x, y = self._location['x'], self._location['y']
        r = self._view_radius

        final_direction = self._body_direction + self._gaze_direction
        magnitude = np.linalg.norm(final_direction)
        if magnitude != 0:
            final_direction = final_direction / magnitude

        rotation_angle = np.degrees(np.arctan2(-final_direction[1], final_direction[0]))

        area = r * r
        view_width = int(np.sqrt(2 * area * self._focus))
        view_width = max(view_width, 2)

        view_height = int(np.ceil(area / view_width))
        view_height = max(view_height, 2)

        M = cv2.getRotationMatrix2D((x, y), -rotation_angle, 1.0)

        rotated_image = cv2.warpAffine(image, M, (width, height))

        cropped_image = cv2.getRectSubPix(rotated_image, (view_width, view_height), (x, y))

        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        normalized_image = cropped_image / 255.0  # Normalize pixel values to [0, 1]

        tensor_input = torch.tensor(normalized_image, dtype=torch.float32).flatten()[:area]
        if len(tensor_input) != area:
            print(f"{view_width=} {view_height=}")

        if self._add_noise:
            confusion = torch.randn_like(tensor_input)
            tensor_input += confusion

        self._internal_state['last_observation'] = tensor_input

    def process(self) -> torch.Tensor:
        """
        Process the observed data through the NEAT network and update location.

        :return: A tensor containing the NEAT network's outputs.
        """
        input_tensor = self._internal_state.get('last_observation')
        if input_tensor is None:
            return torch.tensor([])  # Return empty tensor if no observation

        # Ensure the input tensor has the correct shape
        network_output = self._network.activate(input_tensor.numpy())

        # Convert output to torch tensor
        output_tensor = torch.tensor(network_output, dtype=torch.float32)

        # Store the output
        self._internal_state['last_output'] = output_tensor

        # Extract left and right speeds from the first two outputs
        if len(output_tensor) < 2:
            raise ValueError("NEAT network must output at least two values for speeds.")

        left_speed = output_tensor[0].item()
        right_speed = output_tensor[1].item()
#        self._gaze_direction = np.array([np.cos(output_tensor[2].item()), np.sin(output_tensor[2].item())])
#        self._focus = output_tensor[3].item()

        self._derive_direction(left_speed, right_speed)

        self._update_location(left_speed, right_speed)

        return output_tensor

    def _derive_direction(self, left_speed: float, right_speed: float) -> None:
        """
        Derive the agent's orientation based on left and right speeds.

        :param left_speed: Speed of the left wheel.
        :param right_speed: Speed of the right wheel.
        """
        rotation_force = np.clip(left_speed - right_speed, -0.2, 0.2)
        self._rotation += rotation_force
        self._body_direction = np.array([np.cos(self._rotation), np.sin(self._rotation)])

    def _update_location(self, left_speed: float, right_speed: float) -> None:
        """
        Update the agent's location based on speeds and body direction.

        :param left_speed: Speed of the left wheel.
        :param right_speed: Speed of the right wheel.
        """
        speed = left_speed + right_speed
        movement_offset = np.array(self.body_direction, dtype=float) * speed

        new_position = np.array([self._location['x'], self._location['y']]) + movement_offset

        x_max, y_max = self._env.boundaries
        view_limit = self._view_radius / 2
        x_min, y_min = view_limit, view_limit
        x_max -= view_limit
        y_max -= view_limit

        new_position[0] = np.clip(new_position[0], x_min, x_max)
        new_position[1] = np.clip(new_position[1], y_min, y_max)

        x_updated = self._location['x'] != new_position[0]
        y_updated = self._location['y'] != new_position[1]

        self._location['x'], self._location['y'] = new_position

        self._add_noise = not (x_updated or y_updated)

    def reset(self) -> None:
        """
        Reset the agent's internal state and location.
        """
        self._internal_state = {}
        self._location = {'x': 0.0, 'y': 0.0}

    @property
    def state(self) -> dict[str, Any]:
        """
        Get the current internal state of the agent.

        :return: A dictionary representing the agent's state.
        """
        return self._internal_state

    @state.setter
    def state(self, new_state: dict[str, Any]) -> None:
        """
        Set the agent's internal state.

        :param new_state: A dictionary representing the new state.
        """
        self._internal_state = new_state

    @property
    def id(self) -> str:
        """
        Get the unique identifier of the agent.

        :return: The agent's unique ID.
        """
        return self._id

    @property
    def type(self) -> str:
        return "agent"

    @property
    def location(self) -> dict[str, float]:
        """
        Get the current location of the agent.

        :return: A dictionary representing the agent's location (e.g., {'x': 0.0, 'y': 0.0}).
        """
        return self._location

    @location.setter
    def location(self, new_location: dict[str, float]) -> None:
        """
        Set the agent's current location.

        :param new_location: A dictionary representing the new location.
        """
        self._location = new_location

    @property
    @override
    def body_direction(self) -> Iterable[float]:
        return self._body_direction

    @property
    @override
    def gaze_direction(self) -> Iterable[float]:
        return self._gaze_direction

    @property
    @override
    def focus(self) -> float:
        return self._focus
