# Importing libraries
import os
import subprocess

import copy
import shutil
import random
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_or_empty_directory(directory_path):
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

def grid_world_creation(grid_size, start_point, end_point, obstacles, traps):
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

def generate_agent(path_length, random_seed=None):
    """
    Generates a random bitstring of the specified length.

    Parameters:
    path_length (int): The length of the bitstring to be generated.
    random_seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    str: A random bitstring of the specified length.
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    return ''.join(random.choice(['0', '1']) for _ in range(path_length))

def grid_world_to_rgb(grid, agent_flag=1):
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

def grid_world_visualization(grid, title=None, agent_flag=1, saving_path=None, full_legend=0):
    """
    Visualizes the grid world with different colors for each value and an optional legend.

    Parameters:
    grid (np.ndarray): The grid world matrix to be visualized.
    title (str, optional): The title of the plot. If None, the default title "Grid World Visualization" is used.
    agent_flag (int, optional): Flag indicating whether to show agent-related elements. Default is 1.
    saving_path (str, optional): Path to save the plot image. If None, the plot is displayed.
    full_legend (int, optional): Flag indicating whether to show the full legend. Default is 0.

    Returns:
    None
    """
    rgb_image, color_dictionary = grid_world_to_rgb(grid, agent_flag)
    
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.imshow(rgb_image)
    if title is None:
        ax.set_title("Grid World Visualization", fontsize=15)
    else:
        ax.set_title(title, fontsize=15)
    ax.axis("off")

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
                for i in sorted(legend_labels.keys()) if np.any(grid == i)]

    ax.legend(handles=handles, title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

    if saving_path is not None:
        plt.savefig(saving_path, format="png", bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def fitness_score_calculation(agent_path, grid_world, path_length, start_position, end_position, penalty_coefficients, grid_size):
    """
    Calculates the fitness score of an agent's path in a grid world, considering penalties for obstacles and traps.

    Parameters:
    agent_path (str): The bitstring representing the agent's path, with each pair of bits representing a movement direction.
    grid_world (np.ndarray): The grid world matrix.
    path_length (int): The length of the agent's path in bits (should be even).
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
    
    for i in range(0, path_length, 2):
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

def population_sorting(population, primary_fitness_scores, secondary_fitness_scores):
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

def selection(population, bias=2, mode="uniform", random_seed=None):
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

def crossover(parent1, parent2, crossover_point=None, random_seed=None):
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

def mutate(agent, mutation_probability=0.01, random_seed=None):
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

def path_reconstruction(best_population_positions, initial_grid_world, results_path, start_position, end_position, step=1):
    """
    Reconstructs and visualizes the path of the best agent in each generation.

    Parameters:
    best_population_positions (list of list of tuples): A list containing the positions of the best agent in each generation.
    initial_grid_world (np.ndarray): The initial grid world matrix.
    results_path (str): The path where the results (visualizations) will be saved.
    start_position (tuple): The starting position of the agent.
    end_position (tuple): The ending position of the agent.
    step (int, optional): The step size for selecting generations to visualize. Default is 1.

    Returns:
    None
    """
    selected_indices = list(range(0, len(best_population_positions), step))
    if selected_indices[-1] != len(best_population_positions) - 1:
        selected_indices.append(len(best_population_positions) - 1)

    for index in tqdm(selected_indices, desc="Processing generations"):
        generation = index + 1
        best_agent_positions = best_population_positions[index]
        
        generation_path = f"{results_path}/{generation}. generation"
        title = f"{generation}. generation grid world visualization"
        create_or_empty_directory(generation_path)

        grid_world = copy.deepcopy(initial_grid_world)
        for j, position in enumerate(best_agent_positions):
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
            grid_world_visualization(grid_world, title=title, agent_flag=1, saving_path=step_path, full_legend=1)


def video_creation(images_path, video_path, fps=5, video_format="mp4"):
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