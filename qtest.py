import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Arrow, FancyArrowPatch, Rectangle
from matplotlib.figure import Figure
import plotly.graph_objects as go

from typing import Any, Optional
from collections import defaultdict
import math

from hive_mind.agent import Agent
from image_viz import create_hill_image

import io
from PIL import Image


def plot_gauge(kappa_value):
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})

    categories = [
        ('Poor', -1.00, 0.00, 'red'),
        ('Slight', 0.00, 0.20, 'pink'),
        ('Fair', 0.20, 0.40, 'orange'),
        ('Moderate', 0.40, 0.60, 'yellow'),
        ('Substantial', 0.60, 0.80, 'lightblue'),
        ('Almost Perfect', 0.80, 1.00, 'lightgreen')
    ]

    # Create gauge wedges
    for label, start, end, color in categories:
        ax.barh([1], [(end-start) * np.pi / 1.5], left=[(start + 1) * np.pi / 1.5], color=color, edgecolor='k', height=0.3)

    # Add needle
    ax.barh([1], [0.01], left=[(kappa_value + 1) * np.pi / 1.5], color='black', height=0.4)

    ax.set_yticklabels([])  # No radial labels
    ax.set_xticks([])  # No angular ticks
    ax.set_theta_offset(np.pi)  # Set zero to top
    ax.set_theta_direction(-1)  # Flip direction

    plt.show()


def create_gauge(kappa_score):
    # Define the range and corresponding colors for the gauge
    ranges = [-1.00, 0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
    colors = ['red', 'pink', 'orange', 'yellow', 'lightblue', 'green']

    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)  # Rotate to start at the top
    ax.set_theta_direction(-1)  # Clockwise

    # Plot the gauge segments
    for i in range(len(ranges) - 1):
        ax.barh(0, width=np.pi/6, height=1, left=np.pi/3 + i*np.pi/6, color=colors[i], edgecolor='white')

    # Add pointer for the Kappa score
    angle = np.pi/3 + (kappa_score + 1) * np.pi / 3
    ax.plot([0, angle], [0, 1], color='black', lw=3)

    # Hide unnecessary details
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)

    plt.title(f'Kappa Score: {kappa_score}', y=-0.1)
    plt.show()


def create_kappa_gauge(kappa_score: float,
                       figsize: tuple = (10, 5)) -> Image.Image:
    """
    Create a half-circle gauge visualization for a Kappa score using matplotlib.
    """
    categories = [
        (-1.00, 0.00, '#FF4444', 'Poor'),
        (0.00, 0.20, '#FFB6C1', 'Slight'),
        (0.20, 0.40, '#FFA500', 'Fair'),
        (0.40, 0.60, '#FFD700', 'Moderate'),
        (0.60, 0.80, '#87CEEB', 'Substantial'),
        (0.80, 1.00, '#90EE90', 'Almost Perfect')
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    center = (0.5, 0)  # Move center to x-axis
    radius = 0.4

    for start, end, color, label in categories:
        start_angle = 180 - ((start + 1) * 90)
        end_angle = 180 - ((end + 1) * 90)

        arc = Arc(center, radius*2, radius*2,
                 theta1=end_angle,
                 theta2=start_angle,
                 color=color, linewidth=30)
        ax.add_patch(arc)

        mid_angle = np.radians((start_angle + end_angle) / 2)
        label_radius = radius * 1.1
        label_x = center[0] + label_radius * np.cos(mid_angle)
        label_y = center[1] + label_radius * np.sin(mid_angle)

        ha = 'right' if label_x < center[0] else 'left'
        plt.text(label_x, label_y, label, 
                ha=ha, va='center', 
                fontsize=12, 
                fontfamily='sans-serif',
                fontweight='bold')

        inner_radius = radius * 0.98
        inner_x = center[0] + inner_radius * np.cos(mid_angle)
        inner_y = center[1] + inner_radius * np.sin(mid_angle)

        range_text = f"{start:.1f} - {end:.1f}"

        rotation = (start_angle + end_angle) / 2 - 90

        plt.text(inner_x, inner_y, range_text,
                ha='center', va='center',
                rotation=rotation,  # Aligned with gauge curve
                fontsize=8,  # Slightly smaller font
                fontfamily='sans-serif',
                color='white',  # White text for better contrast
                fontweight='bold',  # Bold for better visibility
                zorder=3)


    needle_angle = np.radians(180 - ((kappa_score + 1) * 90))
    needle_length = radius * 0.95
    dx = needle_length * np.cos(needle_angle)
    dy = needle_length * np.sin(needle_angle)

    arrow = FancyArrowPatch(center, 
                           (center[0] + dx, center[1] + dy),
                           color='black',
                           linewidth=2,
                           arrowstyle='-|>',
                           mutation_scale=20,
                           zorder=4)
    ax.add_patch(arrow)
    ax.add_patch(Circle(center, radius=0.01, color='black', zorder=4))

    bg_radius = 0.12
    bg_circle = Circle(center, radius=bg_radius, 
                      facecolor='white', 
                      edgecolor='none', 
                      alpha=0.85,
                      zorder=2)
    ax.add_patch(bg_circle)

    plt.text(center[0], center[1] + .15, 
             f'κ = {kappa_score:.2f}',
             ha='center', va='center',
             fontsize=18, 
             fontweight='bold',
             fontfamily='sans-serif',
             zorder=3)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.02, 0.45)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, 
                format='png',
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none',
                pad_inches=0.05)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image


class FitnessCalculator:
    def __init__(self):
        self.goal: tuple[int, int] | None = None
        self._direction_history = defaultdict(list)
        self._history_size = 30  # Number of frames to consider (adjust based on your needs)
        self._spin_penalty_threshold = 0.5  # Threshold for detecting spinning behavior

    def _calc_fitness(self, agent: Agent) -> tuple[float, float, float]:
        # Distance fitness calculation (unchanged)
        x_y = np.array((agent.location['x'], agent.location['y']))
        dist = np.sum((x_y - self.goal)**2)
        max_fitness = 100
        steepness = 0.01

        distance_fitness = max_fitness * np.exp(-steepness * dist)

        # Direction fitness calculation
        agent_dir = np.array(agent.body_direction)
        target_dir = self.goal - x_y
        target_dir = target_dir / np.linalg.norm(target_dir)

        dot_product = np.dot(agent_dir, target_dir)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle)

        # Calculate instantaneous direction fitness
        direction_fitness = 0.0
        if angle_deg <= 180:
            direction_fitness = max_fitness * (np.cos(angle) + 1) / 2

        # Update direction history
        direction_history = self._direction_history[agent.id]
        direction_history.append(direction_fitness)
        if len(direction_history) > self._history_size:
            direction_history.pop(0)

        consistency_fitness = 0.0
        if len(direction_history) >= 3:

            variance = np.var(direction_history)
            max_possible_variance = max_fitness * max_fitness / 4  # Maximum theoretical variance
            normalized_variance = np.clip(variance / max_possible_variance, 0, 1)


            recent_avg = np.mean(direction_history[-10:]) if len(direction_history) >= 10 else np.mean(direction_history)
            normalized_avg = recent_avg / max_fitness  # 0 to 1 range

            # Combine into consistency metric
            stability = 1 - normalized_variance  # High when variance is low
            consistency_fitness = max_fitness * stability * normalized_avg

            # Optional: Add bonus for very stable good performance
#            if stability > 0.9 and normalized_avg > 0.9:
#                consistency_fitness += bonus * 0.5  # Half bonus for excellent consistency

        return distance_fitness, consistency_fitness



class TestAgent(Agent):
    def __init__(self, config: dict[str, Any], agent_id: str):
        self._id = agent_id
        self._location = {'x': 0.0, 'y': 0.0}
        self._direction = np.array([1.0, 0.0])  # Initial direction (facing right)
        self._rotation = 0  # degrees
        
    # Implement required abstract methods
    def observe(self, input_data: Any) -> None: pass
    def process(self) -> Any: return None
    def reset(self) -> None: pass
    
    @property
    def state(self) -> dict[str, Any]: return {}
    @state.setter
    def state(self, new_state: dict[str, Any]) -> None: pass
    @property
    def id(self) -> str: return self._id
    @property
    def location(self) -> dict[str, float]: return self._location
    @location.setter
    def location(self, new_location: dict[str, float]) -> None:
        self._location = new_location
    @property
    def body_direction(self) -> np.ndarray: return self._direction
    @property
    def gaze_direction(self) -> np.ndarray: return self._direction
    @property
    def focus(self) -> float: return 1.0


def visualize_mcnemar_results(b: int, c: int, chi_square: float, p_value: float, figsize=(10, 8), dpi=100) -> Image.Image:
    """
    Create a visualization of pre-calculated McNemar's test results.

    Parameters:
    -----------
    b : int
        Number of cases where first condition was positive and second negative
    c : int
        Number of cases where first condition was negative and second positive
    chi_square : float
        Pre-calculated chi-square statistic
    p_value : float
        Pre-calculated p-value
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for the output image
    
    Returns:
    --------
    Image.Image
        The image containing the visualization
    """
    total = b + c
    b_proportion = b / total
    is_significant = p_value < 0.05

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = plt.GridSpec(3, 2, height_ratios=[1, 0.5, 0.5])

    ax_bar = fig.add_subplot(gs[0, :])
    chi_square_stat = fig.add_subplot(gs[1, 0])
    p_value_stat = fig.add_subplot(gs[1, 1])
    ax_significance = fig.add_subplot(gs[2, :])

    ax_bar.barh(0, b_proportion, height=0.5, color='#3B82F6', label=f'b: {b}')
    ax_bar.barh(0, 1-b_proportion, height=0.5, color='#E5E7EB', left=b_proportion, label=f'c: {c}')
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.set_yticks([])
    ax_bar.legend(loc='upper right', bbox_to_anchor=(1, 1.3))
    ax_bar.set_title("Proportion of Discordant Pairs", pad=20)

    middle_b = b_proportion / 2
    middle_c = b_proportion + (1 - b_proportion) / 2
    ax_bar.text(middle_b, 0, f'b: {b}', ha='center', va='center', color='white', fontweight='bold')
    ax_bar.text(middle_c, 0, f'c: {c}', ha='center', va='center', color='#4B5563', fontweight='bold')

    box_style = dict(facecolor='#F3F4F6', alpha=0.5, edgecolor='#D1D5DB', boxstyle='round,pad=1')

    chi_square_stat.text(0.5,
                         0.5,
                         f'Chi-Square\n{chi_square:.2f}', 
                         fontsize=16,
                         ha='center',
                         va='center',
                         bbox=box_style)
    chi_square_stat.set_xticks([])
    chi_square_stat.set_yticks([])

    p_value_stat.text(0.5,
                   0.5,
                   f'P-value\n{p_value:.3f}', 
                   fontsize=16,
                   ha='center',
                   va='center',
                   bbox=box_style)

    p_value_stat.set_xticks([])
    p_value_stat.set_yticks([])

    significance_color = '#22C55E' if is_significant else '#EAB308'
    significance_text = ('Significant difference detected (α = 0.05)' 
                        if is_significant 
                        else 'No significant difference detected (α = 0.05)')

    ax_significance.text(0.5, 0.5, significance_text,
                        ha='center', va='center',
                        fontsize=20,
                        bbox=dict(facecolor=significance_color,
                                  alpha=0.8,
                                  edgecolor='none',
                                  boxstyle='round,pad=1'),
                        color='white', fontweight='bold')
    ax_significance.set_xticks([])
    ax_significance.set_yticks([])

    comparison_text = "There is no statistically significant difference between the model disagreement."

    if is_significant:
        if c > b:
            comparison_text = "Current model shows a performance advantage over the new model."
        elif c < b:
            comparison_text = "New model shows a performance advantage over the current model."

    fig.text(0.25,
             0.03, 
             comparison_text,
             fontsize=10,
             color='#4B5563',
             fontweight='bold')


    fig.text(0.10,
             0.54, 
             "Interpretation: The blue portion represents the proportion of b (first condition positive, second negative)\n" +
             "relative to c (first condition negative, second positive). A larger difference indicates a stronger effect.",
             fontsize=10,
             color='#4B5563')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, 
                format='png',
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none',
                pad_inches=0.05)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image


def main():
    image = create_kappa_gauge(.74)
    image.save("/tmp/mah_kappa.png")

    mcnem_img = visualize_mcnemar_results(
        b=12,                  # Your b value
        c=4,                   # Your c value
        chi_square=float("inf"),        # Your pre-calculated chi-square
        p_value=0.0455,        # Your pre-calculated p-value
    )

    mcnem_img.save("/tmp/mcnemar_visualization.png")


#    gauge.write_image("/tmp/kappa.png")


    # Initialize display
#    width, height = 800, 600
#    window_name = "Fitness Visualization"
#    
#    # Create agent and target
#    agent = TestAgent({}, "test_agent")
#    agent.location = {
#        'x': np.random.uniform(0, width),
#        'y': np.random.uniform(0, height)
#    }
#    target_pos = np.array([
#        np.random.uniform(0, width),
#        np.random.uniform(0, height)
#    ])
#
#    
#    fitness_calc = FitnessCalculator()
#    fitness_calc.goal = target_pos
#    rotation_speed = 5
#    
#    while True:
#        # Create base image
#        img = create_hill_image(width, height, target_pos[0], target_pos[1])
#        
#        # Draw target
#        cv2.circle(img, tuple(target_pos.astype(int)), 10, (0, 0, 255), -1)
#        
#        # Draw agent
#        agent_pos = np.array([agent.location['x'], agent.location['y']])
#        cv2.circle(img, tuple(agent_pos.astype(int)), 15, (255, 0, 0), -1)
#        
#        # Draw agent direction
#        direction_length = 40
#        direction_end = agent_pos + agent._direction * direction_length
#        cv2.line(img, tuple(agent_pos.astype(int)), 
#                tuple(direction_end.astype(int)), (0, 255, 0), 2)
#        
#        # Draw ideal direction
#        ideal_direction = target_pos - agent_pos
#        ideal_direction = ideal_direction / np.linalg.norm(ideal_direction)
#        ideal_end = agent_pos + ideal_direction * direction_length
#        cv2.line(img, tuple(agent_pos.astype(int)), 
#                tuple(ideal_end.astype(int)), (255, 255, 0), 2)
#        
#        # Calculate and display fitness
#        dist_fitness, dir_fitness = fitness_calc._calc_fitness(agent)
#        total_fitness = dist_fitness + dir_fitness
#        
#        # Display fitness values
#        cv2.putText(img, f"Distance Fitness: {dist_fitness:.1f}", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#        cv2.putText(img, f"Direction Fitness: {dir_fitness:.1f}", 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#        cv2.putText(img, f"Total Fitness: {total_fitness:.1f}", 
#                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#        
#        # Show image
#        cv2.imshow(window_name, img)
#        
#        # Handle keyboard input
#        key = cv2.waitKey(1) & 0xFF
#        if key == ord('q'):
#            break
#        elif key == ord('a'):  # Rotate left
#            rotation = np.radians(rotation_speed)
#            cos_rot = np.cos(rotation)
#            sin_rot = np.sin(rotation)
#            rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
#            agent._direction = np.dot(rot_matrix, agent._direction)
#        elif key == ord('d'):  # Rotate right
#            rotation = np.radians(-rotation_speed)
#            cos_rot = np.cos(rotation)
#            sin_rot = np.sin(rotation)
#            rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
#            agent._direction = np.dot(rot_matrix, agent._direction)
#    
#    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
