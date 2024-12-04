import random

import cv2
import numpy as np


class SensorEncoding:
    def __init__(self, max_points=16, max_radius=50):
        self._encoding = {
            'sampling_points': [],  # List of (r, theta) polar coordinates
            'weights': [],         # Weight for each sampling point
            'operations': [],      # List of operations to apply
            'radius': 20.0,       # Base radius
            'channel': 0          # Channel selection
        }

        self._max_points = max_points
        self._max_radius = max_radius

    def mutate(self, mutation_rate=0.1):
        """Mutate the _encoding with various possible mutations"""
        if random.random() < mutation_rate:
            mutation_type = random.choice([
                'add_point', 'remove_point', 'modify_point',
                'modify_weight', 'modify_radius', 'modify_channel',
            ])

            if mutation_type == 'add_point' and len(self._encoding['sampling_points']) < self._max_points:
                r = random.uniform(0, self._encoding['radius'])
                theta = random.uniform(-np.pi, np.pi)
                self._encoding['sampling_points'].append((r, theta))
                self._encoding['weights'].append(random.uniform(-1, 1))

            elif mutation_type == 'remove_point' and len(self._encoding['sampling_points']) > 1:
                idx = random.randrange(len(self._encoding['sampling_points']))
                self._encoding['sampling_points'].pop(idx)
                self._encoding['weights'].pop(idx)

            elif mutation_type == 'modify_point':
                if self._encoding['sampling_points']:
                    idx = random.randrange(len(self._encoding['sampling_points']))
                    r, theta = self._encoding['sampling_points'][idx]
                    r += random.gauss(0, self._encoding['radius'] * 0.1)
                    theta += random.gauss(0, 0.2)
                    self._encoding['sampling_points'][idx] = (
                        np.clip(r, 0, self._max_radius),
                        theta % (2 * np.pi)
                    )

            elif mutation_type == 'modify_radius':
                self._encoding['radius'] *= np.exp(random.gauss(0, 0.1))
                self._encoding['radius'] = np.clip(self._encoding['radius'], 1, self._max_radius)


class SensorGenotype:
    """
    String-based encoding for sensor properties that supports easy mutation and crossover
    Format: 
    R{radius}|P{x,y,w}|P{x,y,w}|...|
    where:
    - R: radius section
    - P: point section with x,y coordinates (in polar) and weight
    Each value is encoded as 8 hex digits
    """
    
    def __init__(self, max_points=16, max_radius=50):
        self.max_points = max_points
        self.max_radius = max_radius
        # Initialize with default values
        self.genome = self._create_initial_genome()

    def _float_to_hex(self, f: float, scale: float = 1.0) -> str:
        """Convert float to 8-character hex string"""
        # Scale the float to use full range of 32-bit int
        scaled = int((f / scale) * ((1 << 31) - 1))
        return f"{scaled & 0xFFFFFFFF:08x}"  # Ensure 8 chars with leading zeros

    def _hex_to_float(self, h: str, scale: float = 1.0) -> float:
        """Convert 8-character hex string back to float"""
        val = int(h, 16)
        # Handle negative numbers (2's complement)
        if val & (1 << 31):
            val -= 1 << 32
        return (val * scale) / ((1 << 31) - 1)

    def _create_initial_genome(self) -> str:
        """Create initial genome string with some default sampling points"""
        # Start with radius encoding
        genome = f"R{self._float_to_hex(20.0, self.max_radius)}"

        # Add some evenly spaced points
        num_initial_points = 8
        angles = np.linspace(0, 2*np.pi, num_initial_points, endpoint=False)
        for theta in angles:
            r = self.max_radius * random.random()
            w = 1.0  # Initial weight
            point_str = (f"|P{self._float_to_hex(r, self.max_radius)}"
                        f"{self._float_to_hex(theta, 2*np.pi)}"
                        f"{self._float_to_hex(w, 1.0)}")
            genome += point_str
            
        return genome

    def decode(self) -> dict:
        """Decode genome string into sensor parameters"""
        parts = self.genome.split('|')
        result = {
            'sampling_points': [],
            'weights': [],
            'radius': 0
        }

        for part in parts:
            if part.startswith('R'):  # Radius
                result['radius'] = self._hex_to_float(part[1:], self.max_radius)
            elif part.startswith('P'):  # Point
                hex_str = part[1:]
                r = self._hex_to_float(hex_str[:8], self.max_radius)
                theta = self._hex_to_float(hex_str[8:16], 2*np.pi)
                w = self._hex_to_float(hex_str[16:], 1.0)
                result['sampling_points'].append((r, theta))
                result['weights'].append(w)

        return result

    def mutate(self, mutation_rate=0.1):
        """Mutate the genome string"""
        parts = self.genome.split('|')

        # Possibly mutate radius
        if random.random() < mutation_rate:
            current_radius = self._hex_to_float(parts[0][1:], self.max_radius)
            mutated_radius = current_radius * np.exp(random.gauss(0, 0.1))
            mutated_radius = np.clip(mutated_radius, 1, self.max_radius)
            parts[0] = f"R{self._float_to_hex(mutated_radius, self.max_radius)}"

        # Mutate points
        points = parts[1:]
        if random.random() < mutation_rate and len(points) < self.max_points:
            # Add new point
            r = random.uniform(0, self.max_radius)
            theta = random.uniform(0, 2*np.pi)
            w = random.uniform(-1, 1)
            new_point = (f"P{self._float_to_hex(r, self.max_radius)}"
                        f"{self._float_to_hex(theta, 2*np.pi)}"
                        f"{self._float_to_hex(w, 1.0)}")
            points.append(new_point)
        
        # Possibly remove a point
        if random.random() < mutation_rate and len(points) > 1:
            points.pop(random.randrange(len(points)))
        
        # Mutate existing points
        for i in range(len(points)):
            if random.random() < mutation_rate:
                hex_str = points[i][1:]
                r = self._hex_to_float(hex_str[:8], self.max_radius)
                theta = self._hex_to_float(hex_str[8:16], 2*np.pi)
                w = self._hex_to_float(hex_str[16:], 1.0)
                
                # Mutate values
                r *= np.exp(random.gauss(0, 0.1))
                r = np.clip(r, 0, self.max_radius)
                theta = (theta + random.gauss(0, 0.1)) % (2*np.pi)
                w += random.gauss(0, 0.1)
                w = np.clip(w, -1, 1)
                
                points[i] = (f"P{self._float_to_hex(r, self.max_radius)}"
                            f"{self._float_to_hex(theta, 2*np.pi)}"
                            f"{self._float_to_hex(w, 1.0)}")
        
        self.genome = '|'.join([parts[0]] + points)

    def crossover(self, other: 'SensorGenotype') -> tuple['SensorGenotype', 'SensorGenotype']:
        """Perform crossover between two genomes"""
        parent1_parts = self.genome.split('|')
        parent2_parts = other.genome.split('|')
        
        # Decide crossover point for radius and points separately
        child1_genome = SensorGenotype(self.max_points, self.max_radius)
        child2_genome = SensorGenotype(self.max_points, self.max_radius)
        
        # Crossover radius (random choice between parents)
        if random.random() < 0.5:
            radius1, radius2 = parent1_parts[0], parent2_parts[0]
        else:
            radius1, radius2 = parent2_parts[0], parent1_parts[0]
            
        # Crossover points
        points1 = parent1_parts[1:]
        points2 = parent2_parts[1:]
        
        # Random point to split point lists
        split = random.randint(0, min(len(points1), len(points2)))
        
        # Create children
        child1_parts = [radius1] + points1[:split] + points2[split:]
        child2_parts = [radius2] + points2[:split] + points1[split:]
        
        # Ensure max points constraint
        if len(child1_parts) > self.max_points + 1:  # +1 for radius
            child1_parts = child1_parts[:self.max_points + 1]
        if len(child2_parts) > self.max_points + 1:
            child2_parts = child2_parts[:self.max_points + 1]
            
        child1_genome.genome = '|'.join(child1_parts)
        child2_genome.genome = '|'.join(child2_parts)
        
        return child1_genome, child2_genome


def visualize_sensor(genotype: SensorGenotype, size=(200, 200), scale=1.0):
    """
    Visualize the sensor _encoding as an image.
    
    Args:
        _encoding: SensorEncoding instance to visualize
        size: Tuple of (width, height) for the output image
        scale: Scale factor for visualization
        
    Returns:
        visualization: numpy array containing the visualization
    """
    import cv2
    import numpy as np
    sensor = SensorEncoding()
    sensor._encoding = genotype.decode()
    print(f"{genotype.genome=}")

    # Create a blank image with white background
    width, height = size
    visualization = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = (width // 2, height // 2)
    
    # Draw reference circles
    for r in np.linspace(0, sensor._max_radius * scale, 4):
        cv2.circle(visualization, center, int(r), (200, 200, 200), 1)
    
    # Draw direction reference
    cv2.line(visualization, center, 
             (int(center[0] + sensor._max_radius * scale), center[1]), 
             (150, 150, 150), 1)
    
    # Create a heatmap overlay for weights
    heatmap = np.zeros((height, width), dtype=np.float32)
    for (r, theta), weight in zip(sensor._encoding['sampling_points'], 
                                sensor._encoding['weights']):
        # Convert polar to cartesian coordinates
        x = center[0] + r * scale * np.cos(theta)
        y = center[1] + r * scale * np.sin(theta)
        pos = (int(x), int(y))
        
        # Draw sampling point
        color = (0, 0, 255) if weight < 0 else (0, 255, 0)  # Red for negative, green for positive
        radius = int(abs(weight * 5) + 2)  # Point size based on weight magnitude
        cv2.circle(visualization, pos, radius, color, -1)
        
        # Add to heatmap
        cv2.circle(heatmap, pos, int(10 * scale), abs(weight), -1)
    
    # Normalize and colorize heatmap
    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend heatmap with original visualization
        alpha = 0.3
        visualization = cv2.addWeighted(visualization, 1 - alpha, 
                                      heatmap_colored, alpha, 0)
    
    # Draw sensor radius
    cv2.circle(visualization, center, 
               int(sensor._encoding['radius'] * scale), 
               (0, 0, 0), 1)
    
    # Add legend
    legend_y = 20
    cv2.putText(visualization, "Positive weight", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(visualization, "Negative weight", (10, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return visualization


def display_sensor_examples():
    """
    Display several example sensor configurations
    """
    # Create a few different sensor configurations
    encodings = []
    
    # Create a basic circular sensor
#    basic = SensorEncoding(max_points=8, max_radius=40)
#    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
#    radii = np.linspace(5, 50, 8, endpoint=False)
#    basic._encoding['sampling_points'] = [(r, theta) for r, theta in zip(radii, angles)]
#    basic._encoding['weights'] = [1.0] * 8
#    encodings.append(("Basic Circular", basic))
#
#    # Create a forward-focused sensor
#    forward = SensorEncoding(max_points=12, max_radius=40)
#    angles = np.linspace(-np.pi/3, np.pi/3, 12)
#    forward._encoding['sampling_points'] = [(r, theta) 
#                                         for r, theta in zip(np.linspace(20, 40, 12), angles)]
#    forward._encoding['weights'] = [1.0] * 12
#    encodings.append(("Forward Focused", forward))
#    
#    # Create a differential sensor (positive front, negative sides)
#    differential = SensorEncoding(max_points=12, max_radius=40)
#    angles = np.linspace(-np.pi, np.pi, 12, endpoint=False)
#    differential._encoding['sampling_points'] = [(30, theta) for theta in angles]
#    differential._encoding['weights'] = [1.0 if abs(theta) < np.pi/3 else -0.5 
#                                      for theta in angles]
    random_sensor = SensorGenotype()
    encodings.append(("Random", random_sensor))
    
    # Create visualization grid
    rows = len(encodings)
    fig_height = 200 * rows
    visualization = np.ones((fig_height, 200, 3), dtype=np.uint8) * 255
    
    # Draw each sensor configuration
    for i, (name, _encoding) in enumerate(encodings):
        vis = visualize_sensor(_encoding)
        y_start = i * 200
        visualization[y_start:y_start+200, :] = vis
        
        # Add title
        cv2.putText(visualization, name, (10, y_start + 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return visualization


if __name__ == "__main__":
    visualization = display_sensor_examples()
    cv2.imshow("Sensor Configurations", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
