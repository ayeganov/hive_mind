from typing import Any
import uuid

import cv2
import neat
import numpy as np
import torch

from .agent import Agent


class ImageAgent(Agent):
    """
    An agent that observes a portion of an image and moves based on NEAT network outputs.
    """

    def __init__(self, config: dict[str, Any], agent_id: str | None = None):
        """
        Initialize the ImageAgent with a configuration and unique identifier.

        :param config: A dictionary containing NEAT genome and configuration.
        :param agent_id: Optional unique identifier for the agent. If not provided, a UUID is generated.
        """
        self.config = config
        self.internal_state = {}

        self._id = agent_id if agent_id is not None else str(uuid.uuid4())

        self.network = neat.nn.FeedForwardNetwork.create(config['genome'], config['config'])

        self._location = {'x': 0.0, 'y': 0.0}
        self.view_radius = config.get('view_radius', 20)  # Default view radius if not specified

    def observe(self, input_data: Any) -> None:
        """
        Observe a rectangular portion of the image based on current location and view radius.

        :param input_data: The full image as a NumPy array.
        """
        image: np.ndarray = input_data  # Assuming input_data is a NumPy array representing the image
        height, width, _ = image.shape
        x, y = self._location['x'], self._location['y']
        r = self.view_radius

        # Define the rectangular region
        left = int(max(x - r, 0))
        right = int(min(x + r, width))
        top = int(max(y - r, 0))
        bottom = int(min(y + r, height))

        # Crop the image
        cropped_image = image[top:bottom, left:right]

        # If the cropped image is smaller than expected (e.g., at the edges), pad it
        expected_size = (2 * r, 2 * r, image.shape[2])
        if cropped_image.shape[0] != expected_size[0] or cropped_image.shape[1] != expected_size[1]:
            padded_image = np.zeros(expected_size, dtype=image.dtype)
            padded_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
            cropped_image = padded_image

        # Convert the image to grayscale if it's not already
        if cropped_image.shape[2] == 3:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Normalize the image
        normalized_image = cropped_image / 255.0  # Normalize pixel values to [0, 1]

        # Flatten the image and convert to torch tensor
        tensor_input = torch.tensor(normalized_image, dtype=torch.float32).flatten()
        self.internal_state['last_observation'] = tensor_input

    def process(self) -> torch.Tensor:
        """
        Process the observed data through the NEAT network and update location.

        :return: A tensor containing the NEAT network's outputs.
        """
        input_tensor = self.internal_state.get('last_observation')
        if input_tensor is None:
            return torch.tensor([])  # Return empty tensor if no observation

        # Ensure the input tensor has the correct shape
        network_output = self.network.activate(input_tensor.numpy())

        # Convert output to torch tensor
        output_tensor = torch.tensor(network_output, dtype=torch.float32)

        # Store the output
        self.internal_state['last_output'] = output_tensor

        # Extract left and right speeds from the first two outputs
        if len(output_tensor) < 2:
            raise ValueError("NEAT network must output at least two values for speeds.")

        left_speed = output_tensor[0].item()
        right_speed = output_tensor[1].item()

        self._derive_orientation(left_speed, right_speed)

        # Update location based on speeds and orientation
        self._update_location(left_speed, right_speed)

        return output_tensor

    def _derive_orientation(self, left_speed: float, right_speed: float) -> None:
        """
        Derive the agent's orientation based on left and right speeds.

        :param left_speed: Speed of the left wheel.
        :param right_speed: Speed of the right wheel.
        """
        # Simple differential steering model
        if left_speed == right_speed:
            # Moving straight; no change in orientation
            delta_orientation = 0.0
        else:
            # Calculate change in orientation
            delta_orientation = (right_speed - left_speed) * self.config.get('orientation_factor', 1.0)

        self.internal_state['orientation'] = self.internal_state.get('orientation', 0.0) + delta_orientation

        # Normalize orientation to [0, 360)
        self.internal_state['orientation'] %= 360.0

    def _update_location(self, left_speed: float, right_speed: float) -> None:
        """
        Update the agent's location based on speeds and orientation.

        :param left_speed: Speed of the left wheel.
        :param right_speed: Speed of the right wheel.
        """
        # Assume that forward movement is based on the average of left and right speeds
        forward_speed = (left_speed + right_speed) / 2.0
        orientation_deg = self.internal_state.get('orientation', 0.0)
        orientation_rad = np.deg2rad(orientation_deg)

        # Calculate delta movements
        delta_x = forward_speed * np.cos(orientation_rad)
        delta_y = forward_speed * np.sin(orientation_rad)

        # Update location
        self._location['x'] += delta_x
        self._location['y'] += delta_y

    def reset(self) -> None:
        """
        Reset the agent's internal state and location.
        """
        self.internal_state = {}
        self._location = {'x': 0.0, 'y': 0.0}

    @property
    def state(self) -> dict[str, Any]:
        """
        Get the current internal state of the agent.

        :return: A dictionary representing the agent's state.
        """
        return self.internal_state

    @state.setter
    def state(self, new_state: dict[str, Any]) -> None:
        """
        Set the agent's internal state.

        :param new_state: A dictionary representing the new state.
        """
        self.internal_state = new_state

    @property
    def id(self) -> str:
        """
        Get the unique identifier of the agent.

        :return: The agent's unique ID.
        """
        return self._id

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
