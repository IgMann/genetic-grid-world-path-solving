# Importing modules
import os
import subprocess

import copy
import shutil
import random
from tqdm import tqdm

import cv2
import math
import numpy as np
import pandas as pd
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
    obstacles: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Creates a grid world matrix with specified start point, end point, and obstacles.

    Parameters:
    grid_size (tuple): The size of the grid (rows, columns).
    start_point (tuple): The coordinates of the start point (row, column).
    end_point (tuple): The coordinates of the end point (row, column).
    obstacles (list of tuples): A list of coordinates for obstacles.

    Returns:
    np.ndarray: A grid world matrix with the specified elements.
    """
    grid = np.zeros(grid_size, dtype=int)
    grid[start_point] = 1  
    grid[end_point] = 2    
    for obstacle in obstacles:
        grid[obstacle] = 3  

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
    Tuple[np.ndarray, Dict[int, list]]:
        - The RGB image array representing the grid.
        - A dictionary mapping grid values to their corresponding RGB colors.
    """
    color_dictionary = {
        0: [255, 250, 205],  # Empty
        1: [0, 128, 0],      # Start (green)
        2: [0, 0, 255],      # End (blue)
        3: [128, 128, 128],  # Obstacle (gray)
        5: [128, 0, 128],    # Agent Position (purple)
        6: [221, 160, 221],  # Agent Path (light purple)
        7: [77, 77, 77]      # Agent Stuck (dark gray)
    }

    if not agent_flag:
        color_dictionary = {k: color_dictionary[k] for k in range(4)}  # Only include empty, start, end, and obstacle

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
    Visualizes the grid world with colors for each value, step numbers for the agent's path,
    optional pheromone levels, and an optional legend.

    Parameters:
    grid_world (np.ndarray): The grid world matrix to be visualized.
    agent_path (list of tuples, optional): Sequence of coordinates representing the agent's path.
    title (str, optional): The title of the plot. If None, a default title is used.
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
                if grid_world[y, x] not in [1, 2, 3]:
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
        5: "Agent Position",
        6: "Agent Path",
        7: "Agent Stuck"
    }

    if not agent_flag:
        legend_labels = {k: legend_labels[k] for k in range(4)}  

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

def create_time_report(
    start_time: float, end_time: float, num_iterations: int, verbose: bool = True
) -> Tuple[float, float, float]:
    """
    Generates a time report for a simulation or iterative process and returns key timing metrics.

    Parameters:
    ----------
    start_time : float
        The start time of the simulation, typically obtained using `time.time()`.
    end_time : float
        The end time of the simulation, typically obtained using `time.time()`.
    num_iterations : int
        The total number of iterations completed during the simulation.
    verbose : bool, default=True
        If True, prints a detailed time report including total time, time per iteration, 
        and iterations per second.

    Returns:
    -------
    Tuple:
        - total_time (float): Total elapsed time in seconds.
        - time_per_iteration (float): Average time taken per iteration in seconds.
        - iterations_per_second (float): Number of iterations completed per second.
    """
    total_time = round(end_time - start_time, 2)
    iterations_per_second = round(num_iterations / total_time, 2) if total_time > 0 else float('inf')
    time_per_iteration = round(total_time / num_iterations, 2) if num_iterations > 0 else float('inf')
    
    if verbose:
        if total_time < 60:
            formatted_total_time = f"{total_time:.2f} seconds"
        else:
            minutes = int(total_time // 60)
            seconds = total_time % 60
            formatted_total_time = f"{minutes} minutes and {seconds:.2f} seconds"

        print(f"Total Time: {formatted_total_time}")
        print(f"Time per Iteration: {time_per_iteration:.4f} seconds")
        print(f"Iterations per Second: {iterations_per_second:.2f}")

    return total_time, time_per_iteration, iterations_per_second

# def fitness_score_calculation(
#     agent_path: str,
#     grid_world: np.ndarray,
#     chromosome_length: int,
#     start_position: Tuple[int, int],
#     end_position: Tuple[int, int],
#     grid_size: Tuple[int, int]
# ) -> Tuple[float, int, np.ndarray, List[Tuple[int, int]]]:
#     """
#     Calculates the fitness score of an agent's path in a grid world.

#     The function evaluates the agent's path based on its distance to the end position,
#     while marking its journey in the grid. Obstacles block progress, and the path
#     terminates upon encountering an obstacle or reaching the end.

#     Parameters:
#     agent_path (str): A bitstring representing the agent's movement directions.
#                       Each pair of bits encodes a movement: 
#                       "00" = down, "01" = right, "10" = left, "11" = up.
#     grid_world (np.ndarray): The grid world matrix where the agent navigates.
#     chromosome_length (int): The total number of bits in the agent's path encoding.
#     start_position (Tuple[int, int]): The starting coordinates of the agent (row, column).
#     end_position (Tuple[int, int]): The target coordinates of the agent (row, column).
#     grid_size (Tuple[int, int]): Dimensions of the grid (rows, columns).

#     Returns:
#     Tuple[float, int, np.ndarray, List[Tuple[int, int]]]:
#         - primary_fitness_score (float): The Euclidean distance from the agent's final position
#           to the end position.
#         - secondary_fitness_score (int): The total number of steps taken by the agent.
#         - grid_world (np.ndarray): The updated grid world with the agent's path marked.
#         - previous_positions (List[Tuple[int, int]]): List of coordinates visited by the agent.
#     """
#     grid_world = copy.deepcopy(grid_world)
#     secondary_fitness_score = 0
#     previous_positions = [start_position]
    
#     for i in range(0, chromosome_length, 2):
#         secondary_fitness_score = len(previous_positions)
#         previous_position = previous_positions[-1]
#         grid_world[previous_position] = 6

#         choice_bytes = agent_path[i] + agent_path[i + 1]
#         if choice_bytes == "00":  # down
#             new_position = (previous_position[0] + 1, previous_position[1])
#         elif choice_bytes == "01":  # right
#             new_position = (previous_position[0], previous_position[1] + 1)
#         elif choice_bytes == "10":  # left
#             new_position = (previous_position[0], previous_position[1] - 1)
#         elif choice_bytes == "11":  # up
#             new_position = (previous_position[0] - 1, previous_position[1])
#         else:
#             raise ValueError("Values only could be: '00', '01', '10', '11'.")

#         if new_position[0] < 0 or new_position[0] >= grid_size[0] or new_position[1] < 0 or new_position[1] >= grid_size[1]:
#             primary_fitness_score = np.inf
#             previous_positions.append(new_position)
#             break

#         if grid_world[new_position] == 3:  
#             final_position = new_position
#             grid_world[new_position] = 7
#             previous_positions.append(new_position)
#             break
#         elif new_position == end_position:
#             final_position = new_position
#             previous_positions.append(new_position)
#             break
#         else:
#             final_position = new_position
#             grid_world[new_position] = 5
#             previous_positions.append(new_position)

#     grid_world[start_position] = 1
#     primary_fitness_score = round(np.sqrt((final_position[0] - end_position[0]) ** 2 + (final_position[1] - end_position[1]) ** 2), 4)

#     return primary_fitness_score, secondary_fitness_score, grid_world, previous_positions

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
    """Check if the move is valid (within grid world and not an obstacle)."""
    return 0 <= position[0] < grid_world.shape[0] and 0 <= position[1] < grid_world.shape[1] and grid_world[position] != 3

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
    Calculate the movement probabilities for an ant to traverse to a neighboring cell.

    Movement probabilities are determined by a combination of pheromone levels (attraction) 
    and a heuristic score based on the distance to the end position. If revisiting is allowed 
    (`revisit_possible=True`), revisited cells are temporarily assigned the minimum pheromone 
    level from surrounding valid cells.

    Parameters:
    - position (Tuple[int, int]): The current position of the ant on the grid.
    - pheromone (np.ndarray): A 2D matrix representing the pheromone levels at each cell.
    - grid_world (np.ndarray): A 2D matrix representing the grid world, where cells are labeled 
        to indicate obstacles, traps, start, end, etc.
    - end_position (Tuple[int, int]): The target end position of the ant.
    - directions (Dict[str, Tuple[int, int]]): A dictionary mapping directions (keys) to their 
        respective movement offsets (values).
    - alpha (float): The weighting factor for the influence of pheromone levels in probability calculation.
    - beta (float): The weighting factor for the influence of the heuristic score in probability calculation.
    - path (List[Tuple[int, int]]): The sequence of positions visited by the ant so far.
    - revisit_possible (bool, optional): Whether revisiting cells in the path is allowed. If True, 
        revisited cells use the lowest pheromone value among valid neighbors for probability calculations. 
        Default is False.

    Returns:
    List[Tuple[float, Tuple[int, int]]]: A list of tuples where each tuple contains:
        - Probability of moving to a particular cell.
        - The position of that cell.

    Notes:
    - Probabilities are normalized such that their sum equals 1, unless all are zero.
    - Revisited cells have their probabilities adjusted based on the surrounding pheromone levels 
      if `revisit_possible=True`.
    """
    probabilities = []
    surrounding_pheromones = []

    # Calculate pheromone levels for all valid surrounding positions
    for _, move in directions.items():
        next_position = (position[0] + move[0], position[1] + move[1])
        if is_valid_move(next_position, grid_world):
            surrounding_pheromones.append(pheromone[next_position])
    
    # Determine the minimum pheromone value in surrounding cells
    min_pheromone = min(surrounding_pheromones, default=0)

    for _, move in directions.items():
        next_position = (position[0] + move[0], position[1] + move[1])
        
        if not is_valid_move(next_position, grid_world):
            probabilities.append((0, next_position))
        elif next_position in path and not revisit_possible:
            probabilities.append((0, next_position))
        else:
            # Use minimum pheromone value for revisited positions
            effective_pheromone = min_pheromone if next_position in path else pheromone[next_position]
            heuristic_score = calculate_heuristic_score(next_position, end_position)
            prob = (effective_pheromone ** alpha) * ((1.0 / (1 + heuristic_score)) ** beta)
            probabilities.append((prob, next_position))

    # Normalize probabilities
    total_prob = sum(prob for prob, _ in probabilities if prob is not None)
    return [(prob / total_prob if total_prob > 0 else 0, pos) for prob, pos in probabilities if prob is not None]


def ant_walk(
    start: Tuple[int, int],
    end: Tuple[int, int],
    pheromone: np.ndarray,
    grid_world: np.ndarray,
    alpha: float,
    beta: float,
    max_path_length: int,
    revisit_possible: bool = False,
    random_seed: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Simulate an ant's traversal from a starting position to a target position in the grid world.

    The ant selects its path based on the transition probabilities determined by pheromone 
    levels and heuristic scores. The walk terminates when the ant reaches the target, exceeds 
    the maximum allowed path length, or encounters a situation where no valid moves are possible.

    Parameters:
    - start (Tuple[int, int]): The starting position of the ant.
    - end (Tuple[int, int]): The target position of the ant.
    - pheromone (np.ndarray): A 2D matrix representing the pheromone levels at each cell.
    - grid_world (np.ndarray): A 2D matrix representing the grid world.
    - alpha (float): Weight of pheromone influence in transition probabilities.
    - beta (float): Weight of heuristic influence in transition probabilities.
    - max_path_length (int): Maximum number of steps the ant can take before the walk is terminated.
    - revisit_possible (bool, optional): Whether revisiting cells in the path is allowed. Default is False.
    - random_seed (Optional[int], optional): Seed for random number generation to ensure reproducibility. 
      Default is None.

    Returns:
    List[Tuple[int, int]]: A list of positions visited by the ant during its traversal.

    Notes:
    - If `revisit_possible=True`, revisited cells are considered with adjusted pheromone values during transition.
    - The ant stops walking if all transition probabilities are zero, indicating no valid moves.
    """
    if random_seed is not None:
        random.seed(random_seed)

    directions = {
        "11": (-1, 0),  # Up
        "00": (1, 0),   # Down
        "01": (0, 1),   # Right
        "10": (0, -1)   # Left
    }

    path = [start]
    current_position = start

    while current_position != end and len(path) < max_path_length:
        probabilities = calculate_transition_probabilities(
            current_position, pheromone, grid_world, end, directions, alpha, beta, path, revisit_possible
        )
        
        if all(prob == 0 for prob, _ in probabilities):
            break

        next_move = random.choices([p for _, p in probabilities], [prob for prob, _ in probabilities])[0]
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
    # Set values lower or equal than 0.1 to 0
    pheromones[pheromones <= 0.1] = 0

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

def sort_ant_paths(
    paths: List[List[Tuple[int, int]]], 
    end_position: Tuple[int, int]
) -> Tuple[List[List[Tuple[int, int]]], List[int], List[int], List[float]]:
    """
    Sort paths based on their Euclidean distance to the end position as primary score,
    and by number of steps as secondary score, using the population_sorting function.
    Additionally, return the lengths of the sorted paths and their heuristic scores.
    
    Parameters:
    paths (List[List[Tuple[int, int]]]): List of ant paths, where each path is a list of positions.
    end_position (Tuple[int, int]): The target end position for calculating Euclidean distance.
    
    Returns:
    Tuple[List[List[Tuple[int, int]]], List[int], List[int], List[float]]: 
        - Sorted list of paths.
        - List of original indices of the paths after sorting.
        - List of lengths of the sorted paths.
        - List of heuristic scores of the sorted paths.
    """
    # Calculate primary fitness scores using calculate_heuristic_score
    primary_fitness_scores = [
        calculate_heuristic_score(path[-1], end_position) for path in paths
    ]
    # Secondary fitness scores are the lengths of the paths
    secondary_fitness_scores = [len(path) for path in paths]

    # Call population_sorting to sort based on primary and secondary scores
    sorted_paths, sorted_indices = population_sorting(
        paths, primary_fitness_scores, secondary_fitness_scores
    )
    # Lengths of sorted paths
    sorted_path_lengths = [len(path) for path in sorted_paths]

    # Heuristic scores of sorted paths
    sorted_heuristic_scores = [
        calculate_heuristic_score(path[-1], end_position) for path in sorted_paths
    ]

    return sorted_paths, sorted_indices, sorted_path_lengths, sorted_heuristic_scores

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

def check_pheromone_path(pheromone_matrix: np.ndarray, num_optimal_steps: int, threshold: float) -> bool:
    """
    Checks if the pheromone matrix represents a valid full path based on column activity
    and consistency between two threshold levels.

    Parameters:
    pheromone_matrix (np.ndarray): Matrix representing pheromone levels.
    num_optimal_steps (int): Number of optimal steps expected for a full path.
    threshold (float): Threshold for considering a cell as part of the path.

    Returns:
    bool: True if the matrix represents a valid full path, False otherwise.
    """
    binary_matrix1 = np.where(pheromone_matrix > threshold, 1, 0)
    binary_matrix2 = np.where(pheromone_matrix > 2 * threshold, 1, 0)
    
    if np.array_equal(binary_matrix1, binary_matrix2):  # Check if the matrices are identical
        total_active_cells = np.sum(binary_matrix1)

        if total_active_cells == num_optimal_steps + 1:
            column_sums = np.sum(binary_matrix1, axis=0)
            if np.all(column_sums[1:-1] > 0):  # Ensure all middle columns have activity
                return True

    return False

def check_missing_values(
    dataframe: pd.DataFrame, 
    double_line: str = "=" * 100, 
    line: str = "-" * 100, 
    return_missing: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Checks and reports missing values in a DataFrame, providing a detailed summary for each column with missing data.
    
    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to check for missing values.
    double_line : str, default='=' * 100
        String used as a double-line separator in the output.
    line : str, default='-' * 100
        String used as a single-line separator in the output.
    return_missing : bool, default=False
        If True, returns a DataFrame containing rows with missing values and a list of column names with missing data.
    
    Returns:
    -------
    Tuple[Optional[pd.DataFrame], Optional[List[str]]]
        - missing_data_df: DataFrame containing rows with missing values (if return_missing is True, otherwise None).
        - missing_data_columns: List of column names with missing values (if return_missing is True, otherwise None).
    """
    print(double_line)
    print("Missing values check:")
    print(double_line)

    missing_data = dataframe.isnull().sum()
    total_rows = len(dataframe)
    missing_columns = missing_data[missing_data > 0]
    missing_data_df = None
    missing_data_columns = None

    line_flag = False
    if missing_columns.empty:
        print("There is no missing values!")
    else:
        for column, missing_count in missing_columns.items():
            if not line_flag:
                line_flag = True
            else:
                print(line)
            
            missing_percentage = (missing_count / total_rows) * 100
            print(f"{column} missing {missing_count} values out of {total_rows} values ({missing_percentage:.2f}%)")
        
        if return_missing:
            missing_data_df = dataframe[dataframe.isnull().any(axis=1)]
            missing_data_columns = list(missing_columns.index)

    print(double_line)

    if return_missing:
        return missing_data_df, missing_data_columns

    return None, None

def aggregate_categorical_results(
    dataframe: pd.DataFrame,
    categorical_columns: List[str]
) -> pd.DataFrame:
    """
    Aggregates numerical results by computing the mean for each unique combination of categorical values.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data to aggregate.
    categorical_columns : List[str]
        The list of column names used as categorical criteria for grouping.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with aggregated numerical values, preserving the order of columns in the input.
    """
    numerical_columns = dataframe.select_dtypes(include="number").columns.difference(categorical_columns)

    grouped_df = (
        dataframe.groupby(categorical_columns)[numerical_columns]
        .mean()
        .reset_index()
    )

    grouped_df = grouped_df.reindex(columns=dataframe.columns)
    grouped_df[numerical_columns] = grouped_df[numerical_columns].round(2)

    return grouped_df



