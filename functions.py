# Importing libraries
import os
import subprocess

import copy
import shutil
import random
from tqdm import tqdm

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List, Dict

def create_or_empty_directory(directory_path: str) -> None:
    """
    Creates a directory if it does not exist, and if it exists, empties the directory.

    Parameters:
    directory_path (str): The path of the directory to create or empty.

    Returns:
    None
    """
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(directory_path)

def grid_world_creation(
    grid_size: Tuple[int, int],
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
    traps: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Creates a grid world matrix with specified start point, end point, obstacles, and traps.

    Parameters:
    grid_size (tuple): The size of the grid (rows, columns).
    start_point (tuple): The coordinates of the start point (row, column).
    end_point (tuple): The coordinates of the end point (row, column).
    obstacles (list of tuples): A list of coordinates for obstacles.
    traps (list of tuples): A list of coordinates for traps.

    Returns:
    np.ndarray: A grid world matrix with the specified elements.
    """
    grid = np.zeros(grid_size, dtype=int)
    grid[start_point] = 1
    grid[end_point] = 2
    for obstacle in obstacles:
        grid[obstacle] = 3
    for trap in traps:
        grid[trap] = 4

    return grid

def generate_agent(chromosome_length: int, random_seed: Optional[int] = None) -> str:
    """
    Generates a random bitstring of the specified chromosome length.

    Parameters:
    chromosome_length (int): The length of the bitstring to be generated.
    random_seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    str: A random bitstring of the specified chromosome length.
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    return ''.join(random.choice(['0', '1']) for _ in range(chromosome_length))

def grid_world_to_rgb(grid: np.ndarray, agent_flag: int = 1) -> Tuple[np.ndarray, Dict[int, list]]:
    """
    Converts the grid world values to corresponding RGB values.

    Parameters:
    grid (np.ndarray): The grid world matrix to be converted.
    agent_flag (int, optional): Flag indicating whether to include agent-related elements. Default is 1.

    Returns:
    np.ndarray: The RGB image array.
    dict: The color dictionary used for mapping grid values to RGB.
    """
    color_dictionary = {
        0: [255, 250, 205],  # Empty
        1: [0, 128, 0],      # Start (green)
        2: [0, 0, 255],      # End (blue)
        3: [128, 128, 128],  # Obstacle (gray)
        4: [255, 0, 0],      # Trap (red)
        5: [128, 0, 128],    # Agent Position (purple)
        6: [221, 160, 221],  # Agent Path (light purple)
        7: [77, 77, 77],     # Agent Stuck (dark gray)
        8: [0, 0, 0]         # Agent Failed (black)
    }

    if not agent_flag:
        color_dictionary = {k: color_dictionary[k] for k in range(5)}

    rgb_image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

    for key, color in color_dictionary.items():
        rgb_image[grid == key] = color

    return rgb_image, color_dictionary

def grid_world_visualization(
    grid_world: np.ndarray,
    agent_path: Optional[List[Tuple[int, int]]] = None,
    title: Optional[str] = None,
    agent_flag: int = 1,
    saving_path: Optional[str] = None,
    full_legend: int = 0,
    show_pheromones: Optional[np.ndarray] = None
) -> None:
    """
    Visualizes the grid world with different colors for each value, step numbers for agent path (only last step for revisited cells),
    optional pheromone levels, and an optional legend.

    Parameters:
    grid_world (np.ndarray): The grid world matrix to be visualized.
    agent_path (list of tuples, optional): Sequence of coordinates representing the agent's path.
    title (str, optional): The title of the plot. If None, the default title "Grid World Visualization" is used.
    agent_flag (int, optional): Flag indicating whether to show agent-related elements. Default is 1.
    saving_path (str, optional): Path to save the plot image. If None, the plot is displayed.
    full_legend (int, optional): Flag indicating whether to show the full legend. Default is 0.
    show_pheromones (np.ndarray, optional): Matrix of pheromone levels to display on the grid world cells.

    Returns:
    None
    """
    rgb_image, color_dictionary = grid_world_to_rgb(grid_world, agent_flag)
    
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.imshow(rgb_image)
    if title is None:
        ax.set_title("Grid World Visualization", fontsize=15)
    else:
        ax.set_title(title, fontsize=15)
    ax.axis("off")

    # Display step numbers for agent path
    last_step_for_cell = {}
    if agent_path is not None:
        for step, (y, x) in enumerate(agent_path):
            last_step_for_cell[(y, x)] = step  

    for (y, x), step in last_step_for_cell.items():
        ax.text(x, y, str(step), ha="center", va="center", color="black", fontsize=20, fontweight="bold")

    # Overlay pheromone levels if provided
    if show_pheromones is not None:
        for y in range(show_pheromones.shape[0]):
            for x in range(show_pheromones.shape[1]):
                if grid_world[y, x] not in [1, 2, 3, 4]:
                    if show_pheromones[y, x] <= 0.01:
                        ax.text(x, y, f"{show_pheromones[y, x]:.2f}", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
                    else:
                        ax.text(x, y, f"{show_pheromones[y, x]:.2f}", ha="center", va="center", color="purple", fontsize=10, fontweight="bold")

    # Add legend
    legend_labels = {
        0: "Empty",
        1: "Start",
        2: "End",
        3: "Obstacle",
        4: "Trap",
        5: "Agent Position",
        6: "Agent Path",
        7: "Agent Stuck",
        8: "Agent Failed"
    }

    if not agent_flag:
        legend_labels = {k: legend_labels[k] for k in range(5)}

    if agent_flag and full_legend:
        handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=np.array(color_dictionary[i])/255, markersize=10, label=legend_labels[i])
            for i in sorted(legend_labels.keys())]
    else:
        handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=np.array(color_dictionary[i])/255, markersize=10, label=legend_labels[i])
                for i in sorted(legend_labels.keys()) if np.any(grid_world == i)]

    ax.legend(handles=handles, title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

    # Save or display plot
    if saving_path is not None:
        plt.savefig(saving_path, format="png", bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

def fitness_score_calculation(
    agent_path: str,
    grid_world: np.ndarray,
    chromosome_length: int,
    start_position: Tuple[int, int],
    end_position: Tuple[int, int],
    penalty_coefficients: List[float],
    grid_size: Tuple[int, int]
) -> Tuple[float, int, np.ndarray, List[Tuple[int, int]]]:
    """
    Calculates the fitness score of an agent's path in a grid world, considering penalties for obstacles and traps.

    Parameters:
    agent_path (str): The bitstring representing the agent's path, with each pair of bits representing a movement direction.
    grid_world (np.ndarray): The grid world matrix.
    chromosome_length (int): The length of the agent's path in bits (should be even).
    start_position (tuple): The starting coordinates of the agent (row, column).
    end_position (tuple): The ending coordinates of the agent (row, column).
    penalty_coefficients (list): A list of penalty coefficients for normal moves, obstacles, and traps.
    grid_size (tuple): The size of the grid (rows, columns).

    Returns:
    tuple: A tuple containing:
        - primary_fitness_score (float): The primary fitness score of the agent's path, adjusted by penalty coefficients.
        - secondary_fitness_score (int): The secondary fitness score representing the number of moves made by the agent.
        - grid_world (np.ndarray): The updated grid world matrix after the agent's path.
        - previous_positions (list): A list of positions (tuples) visited by the agent during its path.
    """
    grid_world = copy.deepcopy(grid_world)
    penalty_coefficient = penalty_coefficients[0]
    secondary_fitness_score = 0
    previous_positions = [start_position]
    
    for i in range(0, chromosome_length, 2):
        secondary_fitness_score = len(previous_positions)
        previous_position = previous_positions[-1]
        grid_world[previous_position] = 6

        choice_bytes = agent_path[i] + agent_path[i + 1]
        if choice_bytes == "00":  # down
            new_position = (previous_position[0] + 1, previous_position[1])
        elif choice_bytes == "01":  # right
            new_position = (previous_position[0], previous_position[1] + 1)
        elif choice_bytes == "10":  # left
            new_position = (previous_position[0], previous_position[1] - 1)
        elif choice_bytes == "11":  # up
            new_position = (previous_position[0] - 1, previous_position[1])
        else:
            raise ValueError("Values only could be: '00', '01', '10', '11'.")

        if new_position[0] < 0 or new_position[0] >= grid_size[0] or new_position[1] < 0 or new_position[1] >= grid_size[1]:
            primary_fitness_score = np.inf
            previous_positions.append(new_position)
            break

        if grid_world[new_position] == 3:  
            final_position = new_position
            grid_world[new_position] = 7
            penalty_coefficient = penalty_coefficients[1]
            previous_positions.append(new_position)
            break
        elif grid_world[new_position] == 4:  
            final_position = new_position
            grid_world[new_position] = 8
            penalty_coefficient = penalty_coefficients[2]
            previous_positions.append(new_position)
            break
        elif new_position == end_position:
            final_position = new_position
            previous_positions.append(new_position)
            break
        else:
            final_position = new_position
            grid_world[new_position] = 5
            previous_positions.append(new_position)

    grid_world[start_position] = 1
    primary_fitness_score = round(penalty_coefficient * np.sqrt((final_position[0] - end_position[0]) ** 2 + (final_position[1] - end_position[1]) ** 2), 4)

    return primary_fitness_score, secondary_fitness_score, grid_world, previous_positions

def population_sorting(
    population: List[str],
    primary_fitness_scores: List[float],
    secondary_fitness_scores: List[float]
) -> Tuple[List[str], List[int]]:
    """
    Sorts the population based on primary and secondary fitness scores.

    Parameters:
    population (list of str): The list of population elements, each represented as a bitstring.
    primary_fitness_scores (list of float): The primary fitness scores corresponding to the population.
    secondary_fitness_scores (list of float): The secondary fitness scores corresponding to the population.

    Returns:
    tuple: A tuple containing:
        - list of str: The sorted population based on fitness scores.
        - list of int: The sorted indices of the original population list.
    """
    combined = list(zip(population, primary_fitness_scores, secondary_fitness_scores, range(len(population))))
    sorted_combined = sorted(combined, key=lambda x: (x[1], x[2]))
    population_sorted = [item[0] for item in sorted_combined]
    indices_sorted = [item[3] for item in sorted_combined]
    
    return population_sorted, indices_sorted

def selection(
    population: List[str],
    bias: int = 2,
    mode: str = "uniform",
    random_seed: int = None
) -> str:
    """
    Selects one agent from the population based on the specified mode.

    Parameters:
    population (list of str): The list of population elements, each represented as a bitstring.
    bias (int, optional): The bias factor for rank-based selection, default is 2.
    mode (str, optional): The mode of selection. Can be "uniform" or "rank-based". Default is "uniform".
    random_seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    str: The selected agent from the population.

    Raises:
    ValueError: If an invalid mode is specified.
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    if mode == "uniform":
        return random.choice(population)
    elif mode == "rank-based":
        bias_balanced = bias - 1
        weights = [1/(i + len(population)/bias_balanced) for i in range(1, len(population)+1)]
        total_weight = sum(weights)
        probabilities = [weight/total_weight for weight in weights]
        return random.choices(population, weights=probabilities, k=1)[0]
    else:
        raise ValueError("Invalid mode. Choose 'uniform' or 'rank-based'.")

def crossover(
    parent1: str,
    parent2: str,
    crossover_point: int = None,
    random_seed: int = None
) -> Tuple[str, str]:
    """
    Performs a crossover between two parents to produce two offspring.

    Parameters:
    parent1 (str): The bitstring of the first parent.
    parent2 (str): The bitstring of the second parent.
    crossover_point (int, optional): The point at which crossover occurs. If None, a random point is chosen.
    random_seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    tuple: A tuple containing two offspring resulting from the crossover.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    length = len(parent1)
    if crossover_point is None:
        crossover_point = random.randint(1, length - 1)
    
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2

def check_last_n_generations_same(best_scores: List[float], secondary_scores_of_best: List[int], N: int = 3) -> bool:
    """
    Checks if the last N generations have the same best primary and secondary scores.

    Args:
        best_scores (List[float]): List of best primary scores across generations.
        secondary_scores_of_best (List[int]): List of best secondary scores across generations.
        N (int): Number of recent generations to compare, default is 3.

    Returns:
        bool: True if the last N generations have identical best scores, otherwise False.
    """
    if len(best_scores) < N:
        return False
    
    return (all(best_scores[-i] == best_scores[-1] for i in range(1, N)) and
            all(secondary_scores_of_best[-i] == secondary_scores_of_best[-1] for i in range(1, N)))

def mutate(agent: str, mutation_probability: float = 0.01, random_seed: Optional[int] = None) -> str:
    """
    Performs mutation on an agent by randomly flipping bits with a given probability.

    Parameters:
    agent (str): The bitstring of the agent.
    mutation_probability (float, optional): The probability of flipping each bit. Default is 0.01.
    random_seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    str: The mutated bitstring.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    mutated_agent = []
    for bit in agent:
        if random.random() < mutation_probability:
            mutated_agent.append('1' if bit == '0' else '0')
        else:
            mutated_agent.append(bit)
    
    return ''.join(mutated_agent)

def path_reconstruction(
    best_population_paths: List[List[Tuple[int, int]]],
    initial_grid_world: np.ndarray,
    results_path: str,
    start_position: Tuple[int, int],
    end_position: Tuple[int, int],
    step: int = 1,
    title_type: str = "generation",
    path_flag: bool = False
) -> None:
    """
    Reconstructs and visualizes the paths taken by the best agents in a grid world
    for each generation or iteration. Saves visualizations to a specified directory.

    Parameters:
    best_population_paths (List[List[Tuple[int, int]]]): A list of paths where each path 
        is a list of coordinates (tuples) representing the positions visited by the best agent in each generation.
    initial_grid_world (np.ndarray): The initial state of the grid world as a NumPy array.
    results_path (str): The directory path where the visualizations will be saved.
    start_position (Tuple[int, int]): The starting position (x, y) of the agent in the grid world.
    end_position (Tuple[int, int]): The ending position (x, y) of the agent in the grid world.
    step (int, optional): Interval for selecting generations to visualize. Defaults to 1.
    title_type (str, optional): Type of title to be used in the visualizations, either "generation" or "iteration".
        If another string is provided, it will be used as the title directly. Defaults to "generation".
    path_flag (bool, optional): If True, the entire path taken by the agent is visualized. If False, 
        only the current step is shown. Defaults to False.

    Returns:
    None
    """
    create_or_empty_directory(results_path) 
    selected_indices = list(range(0, len(best_population_paths), step))
    if selected_indices[-1] != len(best_population_paths) - 1:
        selected_indices.append(len(best_population_paths) - 1)

    for index in tqdm(selected_indices, desc="Processing generations/iterations"):
        generation = index + 1
        best_agent_path = best_population_paths[index]
        
        generation_path = f"{results_path}/{generation}. generation"
        if title_type == "generation":
            title = f"{generation}. generation grid world visualization"
        elif title_type == "iteration":
            title = f"{generation}. iteration of ACO grid world visualization"
        else:
            title = title_type

        create_or_empty_directory(generation_path)

        grid_world = copy.deepcopy(initial_grid_world)
        for j, position in enumerate(best_agent_path):
            grid_world[grid_world == 5] = 6
            if position == start_position:
                grid_world[start_position] = 5
            else:
                grid_world[start_position] = 1
                if grid_world[position] == 3:
                    grid_world[position] = 7
                elif grid_world[position] == 4:
                    grid_world[position] = 8
                else:
                    grid_world[position] = 5
                    
            step_path = f"{generation_path}/step_{j+1}.png"
            if not path_flag:
                grid_world_visualization(grid_world, title=title, agent_flag=1, saving_path=step_path, full_legend=1)
            else:
                grid_world_visualization(grid_world, agent_path=best_agent_path, title=title, agent_flag=1, saving_path=step_path, full_legend=1)

def video_creation(
    images_path: str,
    video_path: str,
    fps: int = 5,
    video_format: str = "mp4"
) -> None:
    """
    Creates a video from images stored in a specified directory structure and converts it to H.264 format.

    Parameters:
    images_path (str): The path to the directory containing image directories for each generation.
    video_path (str): The path where the created video will be saved.
    fps (int, optional): Frames per second for the video. Default is 5.
    video_format (str, optional): The format of the video file. Default is 'mp4'.

    Returns:
    None
    """
    temp_video_path = "temp_video.avi"
    
    image_files = []
    for generation_directory in sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0])):
        generation_path = os.path.join(images_path, generation_directory)
        for image_file in sorted(os.listdir(generation_path), key=lambda x: int(x.split('_')[1].split('.')[0])):
            image_files.append(os.path.join(generation_path, image_file))
    
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)

    out.release()
    
    command = ["ffmpeg", "-i", temp_video_path, "-vcodec", "libx264", video_path]
    subprocess.run(command)
    os.remove(temp_video_path)
    print(f"Video saved to {video_path}")

def calculate_heuristic_score(next_position: Tuple[int, int], end_position: Tuple[int, int]) -> float:
    """
    Calculate the heuristic score based on the Euclidean distance between the next position and the end position.

    Parameters:
    next_position (Tuple[int, int]): The position to evaluate.
    end_position (Tuple[int, int]): The target end position.

    Returns:
    float: The heuristic score based on Euclidean distance.
    """
    dx = next_position[0] - end_position[0]
    dy = next_position[1] - end_position[1]
    
    return round(math.sqrt(dx**2 + dy**2), 4)

def is_valid_move(position: Tuple[int, int], grid_world: np.ndarray) -> bool:
    """Check if the move is valid (within grid world and not an obstacle or trap)."""
    return 0 <= position[0] < grid_world.shape[0] and 0 <= position[1] < grid_world.shape[1] and grid_world[position] not in [3, 4]

def calculate_transition_probabilities(
    position: Tuple[int, int],
    pheromone: np.ndarray,
    grid_world: np.ndarray,
    end_position: Tuple[int, int],
    directions: Dict[str, Tuple[int, int]],
    alpha: float,
    beta: float,
    path: List[Tuple[int, int]],
    revisit_possible: bool = False
) -> List[Tuple[float, Tuple[int, int]]]:
    """
    Calculate movement probabilities for each direction based on pheromone levels and a distance heuristic,
    setting probability to 0 if the next position is already in the path or is an invalid move.

    Parameters:
    - position: Current position of the ant.
    - pheromone: 2D array of pheromone values on the grid.
    - grid_world: Grid world matrix.
    - end_position: Target position in the grid.
    - directions: Possible move directions with offsets.
    - alpha: Influence of pheromone.
    - beta: Influence of heuristic.
    - path: List of positions visited by the ant so far.
    - revisit_possible: If True, aborts path on revisit. If False, assigns 0 probability to revisited positions.

    Returns:
    List of tuples with probability and position for each move.
    """
    probabilities = []
    for _, move in directions.items():
        next_position = (position[0] + move[0], position[1] + move[1])
        if next_position in path:
            probabilities.append((0 if not revisit_possible else None, next_position))
        elif not is_valid_move(next_position, grid_world):
            probabilities.append((0, next_position))
        else:
            heuristic_score = calculate_heuristic_score(next_position, end_position)
            prob = (pheromone[next_position] ** alpha) * ((1.0 / (1 + heuristic_score)) ** beta)
            probabilities.append((prob, next_position))

    total_prob = sum(prob for prob, _ in probabilities if prob is not None)
    return [(prob / total_prob if total_prob > 0 else 0, pos) for prob, pos in probabilities if prob is not None]


def ant_walk(
    start: Tuple[int, int],
    end: Tuple[int, int],
    pheromone: np.ndarray,
    grid_world: np.ndarray,
    directions: Dict[str, Tuple[int, int]],
    alpha: float,
    beta: float,
    max_path_length: int,
    revisit_possible: bool = False,
    random_seed: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Simulate ant traversal from start to end position, halting if all probabilities for next moves are 0 or if loops are detected.

    Parameters:
    - start: Starting position of the ant.
    - end: End position the ant is trying to reach.
    - pheromone: 2D array of pheromone values on the grid.
    - grid_world: The grid world matrix.
    - directions: Possible directions the ant can move.
    - alpha: Pheromone importance.
    - beta: Heuristic importance.
    - max_path_length: Maximum allowed path length for the ant.
    - revisit_possible: If True, the path terminates upon revisiting a cell. If False, revisits are given a probability of 0.
    - random_seed: Seed for reproducibility.

    Returns:
    List of positions representing the ant's path.
    """
    if random_seed is not None:
        random.seed(random_seed)

    path = [start]
    current_position = start
    while current_position != end and len(path) < max_path_length:
        probabilities = calculate_transition_probabilities(
            current_position, pheromone, grid_world, end, directions, alpha, beta, path, revisit_possible
        )
        
        if all(prob == 0 for prob, _ in probabilities):
            break

        next_move = random.choices([p for _, p in probabilities], [prob for prob, _ in probabilities])[0]
        if revisit_possible and next_move in path:
            break
        path.append(next_move)
        current_position = next_move

    return path

def update_pheromones(
    paths: List[List[Tuple[int, int]]],
    pheromones: np.ndarray,
    evaporation_rate: float,
    deposit_factor: float,
    pheromone_normalization: bool = False
) -> None:
    """
    Update pheromone levels based on paths traversed by ants, with optional normalization.

    Parameters:
    paths (List[List[Tuple[int, int]]]): List of paths traversed by ants, where each path is a list of grid positions.
    pheromones (np.ndarray): The grid storing pheromone levels for each position.
    evaporation_rate (float): The rate at which pheromones evaporate (0 to 1).
    deposit_factor (float): The amount of pheromone deposited by each ant, scaled by path length.
    pheromone_normalization (bool): If True, normalize pheromone levels after updating. Default is False.
    """
    # Pheromone evaporation
    pheromones *= (1 - evaporation_rate)
    
    # Deposit pheromones for each path
    for path in paths:
        path_length = len(path)
        for position in path:
            pheromones[position] += deposit_factor / path_length  # Pheromone deposit
    
    # Optional pheromone normalization
    if pheromone_normalization:
        max_pheromone = pheromones.max()
        if max_pheromone > 0:
            pheromones /= max_pheromone  # Normalize to the range [0, 1]

def sort_ant_paths(paths: List[List[Tuple[int, int]]], end_position: Tuple[int, int]) -> Tuple[List[List[Tuple[int, int]]], List[int]]:
    """
    Sort paths based on their Euclidean distance to the end position as primary score,
    and by number of steps as secondary score, using the population_sorting function.
    
    Parameters:
    paths (List[List[Tuple[int, int]]]): List of ant paths, where each path is a list of positions.
    end_position (Tuple[int, int]): The target end position for calculating Euclidean distance.
    
    Returns:
    Tuple[List[List[Tuple[int, int]]], List[int]]: Sorted list of paths and their original indices.
    """
    def calculate_euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate the Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # Calculate primary and secondary scores for each path
    primary_fitness_scores = [calculate_euclidean_distance(path[-1], end_position) for path in paths]
    secondary_fitness_scores = [len(path) for path in paths]

    # Call population_sorting to sort based on primary and secondary scores
    sorted_paths, sorted_indices = population_sorting(paths, primary_fitness_scores, secondary_fitness_scores)
    
    return sorted_paths, sorted_indices

def moving_average(data: List[float], window_size: int) -> np.ndarray:
    """
    Computes the moving average of a given data series.

    Parameters:
    data (List[float]): The input data series to smooth.
    window_size (int): The number of data points to include in each window for averaging.

    Returns:
    np.ndarray: The smoothed data as an array of moving averages.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

def find_different_paths(
    primary_fitness_scores: List[float],
    secondary_fitness_scores: List[int],
    N: int
) -> List[int]:
    """
    Finds the indices of the top N unique paths with different secondary fitness scores,
    sorted primarily by primary fitness score and secondarily by secondary fitness score.

    Parameters:
    primary_fitness_scores (List[float]): List of primary fitness scores for each path.
    secondary_fitness_scores (List[int]): List of secondary fitness scores for each path.
    N (int): Number of unique paths to return.

    Returns:
    List[int]: List of indices of the selected paths.
    """
    indexed_scores = [(i, (primary_fitness_scores[i], secondary_fitness_scores[i])) for i in range(len(primary_fitness_scores))]
    sorted_scores = sorted(indexed_scores, key=lambda x: (x[1][0], x[1][1]))

    selected_indices = []
    seen_secondary_scores = set()

    for idx, (primary_score, secondary_score) in sorted_scores:
        if secondary_score not in seen_secondary_scores:
            selected_indices.append(idx)
            seen_secondary_scores.add(secondary_score)
            if len(selected_indices) == N:
                break

    return selected_indices

def create_pheromones_matrix(
    best_positions: List[Tuple[int, int]], 
    grid_size: Tuple[int, int]
) -> np.ndarray:
    """
    Creates a pheromone matrix where pheromone levels depend on the frequency of 
    positions in best_positions. Positions with the highest frequency get level 1, 
    positions with no appearance get 0, and all others are scaled between.

    Parameters:
    best_positions (List[Tuple[int, int]]): List of coordinates representing positions visited in the best paths.
    grid_size (Tuple[int, int]): Size of the grid (rows, columns).

    Returns:
    np.ndarray: A matrix with pheromone levels for each position in the grid.
    """
    pheromones = np.zeros(grid_size, dtype=float)
    position_counts = {}

    for position in best_positions:
        if isinstance(position, tuple):  # Only add valid coordinates, not strings
            position_counts[position] = position_counts.get(position, 0) + 1

    max_count = max(position_counts.values(), default=1)

    for position, count in position_counts.items():
        pheromones[position] = count / max_count

    return pheromones