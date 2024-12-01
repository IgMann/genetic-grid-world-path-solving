# Importing modules
from time import time
from typing import Tuple, List, Optional

import numpy as np

import functions as fn

def ga_simulation(
    num_generations: int,
    population_size: int,
    chromosome_length: int,
    mutation_rate: float,
    crossover_type: str,
    progressive_mutation: bool,
    bias: int,
    early_stop: bool,
    best_ones_percentage: float,
    worst_ones_percentage: float,
    num_optimal_steps: int,
    start_position: Tuple[int, int],
    end_position: Tuple[int, int],
    grid_size: Tuple[int, int],
    initial_grid_world: np.ndarray,
    random_seed: int,
    simulation_started_message: str,
    simulation_finished_message: str,
    verbose: Optional[str] = None, 
    line: Optional[str] = 100*'-',
    double_line: Optional[str] = 100*'=',
) -> Tuple[
    Optional[int], int, float, int, str, float, float, List[float], List[int], List[float], List[float], List[np.ndarray], List[List[Tuple[int, int]]], int
    ]:
    """
    Simulates a Genetic Algorithm (GA) process to optimize paths in a grid world, 
    supporting multiple crossover strategies and dynamic mutation adjustment.

    Parameters:
    ----------
    num_generations : int
        Total number of generations for the simulation.
    population_size : int
        Number of agents in the population.
    chromosome_length : int
        Length of the bitstring representation of agents.
    mutation_rate : float
        Initial mutation rate for agents.
    crossover_type : str
        Type of crossover strategy to use. Options:
        - "all to all": Performs crossover between all pairs of agents.
        - "best to rest": Performs crossover between the best agents and the rest of the population.
        - "hybrid": Combines the best agents, middle agents, and worst replacements.
    progressive_mutation : bool
        If True, increases mutation rate dynamically when generations show no improvement.
    bias : int
        Bias factor for rank-based selection.
    early_stop : bool
        Whether to stop the simulation early if an optimal solution is found.
    best_ones_percentage : float
        Percentage of the population to consider as "best" in the hybrid or best-to-rest crossover strategies.
    worst_ones_percentage : float
        Percentage of the population to consider as "worst" in the hybrid crossover strategy.
    num_optimal_steps : int
        Number of steps in the optimal path.
    start_position : Tuple[int, int]
        Starting position of agents in the grid.
    end_position : Tuple[int, int]
        Target position of agents in the grid.
    grid_size : Tuple[int, int]
        Dimensions of the grid world.
    initial_grid_world : np.ndarray
        Initial state of the grid world.
    random_seed : int
        Random seed for reproducibility.
    simulation_started_message : str
        Message to display at the start of the simulation.
    simulation_finished_message : str
        Message to display at the end of the simulation.
    verbose : Optional[str], default=None
        Logging level. Options:
        - None: No output is logged.
        - "Restricted": Logs summary metrics at the end of the simulation.
        - "Full": Logs detailed information for each generation and summary metrics.
    line : Optional[str], default='-' * 100
        The line separator used in verbose logs.
    double_line : Optional[str], default='=' * 100
        The double-line separator used in verbose logs.

    Returns:
    -------
    Tuple:
        - first_full_path_generation (Optional[int]): The generation number of the first full path, or None if not found.
        - best_generation (int): The generation number with the best solution.
        - final_best_score (float): The best primary score achieved.
        - final_best_secondary_score (int): The best secondary score achieved.
        - total_time (float): Total elapsed time for the simulation in seconds.
        - time_per_generation (float): Average time taken per generation in seconds.
        - generations_per_second (float): Number of generations completed per second.
        - best_scores (List[float]): Best primary scores for each generation.
        - secondary_scores_of_best (List[int]): Secondary scores of the best solutions for each generation.
        - median_scores (List[float]): Median primary scores for each generation.
        - mean_scores (List[float]): Mean primary scores for each generation.
        - best_grid_worlds (List[np.ndarray]): The grid worlds corresponding to the best solutions for each generation.
        - best_population_paths (List[List[Tuple[int, int]]]): The best paths found for each generation.
        - generation (int): Total number of generations completed.
        - primary_fitness_scores (List[float]): Primary fitness scores of all agents in the final generation.
        - secondary_fitness_scores (List[int]): Secondary fitness scores of all agents in the final generation.
        - grid_worlds (List[np.ndarray]): All grid worlds evaluated in the final generation.
        - population_paths (List[List[Tuple[int, int]]]): All population paths evaluated in the final generation.
    """
    
    def all_to_all_crossover() -> List[Tuple[str, str]]:
        """
        Performs all-to-all crossover for the population.

        Returns:
        -------
        List[Tuple[str, str]]:
            A list of tuples where each tuple contains two new agents produced by crossover.
        """
        new_agents = []
        for i in range(int(population_size/2)):
            agent1 = fn.selection(population_sorted, bias=bias, mode="rank-based", random_seed=i)
            counter = 1
            while True:
                agent2 = fn.selection(population_sorted, bias=bias, mode="rank-based", random_seed=i * 42 + counter)
                if agent1 != agent2:
                    break
                counter += 1

            new_agent1, new_agent2 = fn.crossover(agent1, agent2, random_seed=i)
            new_agents.append((new_agent1, new_agent2))

        return new_agents

    def best_to_rest() -> List[Tuple[str, str]]:
        """
        Performs crossover between the best agents and the rest of the population.

        Returns:
        -------
        List[Tuple[str, str]]:
            A list of tuples where each tuple contains two new agents produced by crossover.
        """
        rest_individuals = population_sorted[num_best:]

        new_agents = []
        for i in range(int(population_size/2)):
            agent1 = fn.selection(best_individuals, mode="uniform", random_seed=i)
            counter = 0
            while True:
                agent2 = fn.selection(rest_individuals, bias=bias, mode="rank-based", random_seed=i * 42 + counter)
                if agent1 != agent2:
                    break
                counter += 1

            new_agent1, new_agent2 = fn.crossover(agent1, agent2, random_seed=i)
            new_agents.append((new_agent1, new_agent2))

        return new_agents

    def hybrid_crossover() -> List[Tuple[str, str]]:
        """
        Performs crossover between the best agents and middle agents while keeping worst replacements.

        Returns:
        -------
        List[Tuple[str, str]]:
            A list of tuples where each tuple contains two new agents produced by crossover.
        """
        middle_individuals = population_sorted[num_best:num_best+num_middle]

        new_agents = []
        for i in range(int(num_middle / 2)):
            agent1 = fn.selection(best_individuals, mode="uniform", random_seed=i)
            counter = 0
            while True:
                agent2 = fn.selection(middle_individuals, bias=bias, mode="rank-based", random_seed=i * 42 + counter)
                if agent1 != agent2:
                    break
                counter += 1

            new_agent1, new_agent2 = fn.crossover(agent1, agent2, random_seed=i)
            new_agents.append((new_agent1, new_agent2))

        return new_agents

    if verbose == "Full":
        print(double_line)
        print(simulation_started_message)
        print(double_line)

    start_time = time()

    if crossover_type != "all to all":
        num_best = int(population_size * best_ones_percentage)

        if crossover_type == "best to rest":
            pass
        elif crossover_type == "hybrid":
            num_worst = int(population_size * worst_ones_percentage)
            num_middle = population_size - num_best - num_worst
        else:
            raise ValueError("Crossover type could be only \"all to all\", \"best to rest\" or \"hybrid\".")

    best_scores = []
    secondary_scores_of_best = []
    median_scores = []
    mean_scores = []
    best_grid_worlds = []
    best_population_paths = []
    first_full_path_generation = None
    convergence_flag = False

    for generation in range(1, num_generations + 1):
        if generation == 1:
            population = [fn.generate_agent(chromosome_length, random_seed=i) for i in range(population_size)]
        else:
            population = []

            if crossover_type == "all to all":
                new_agents = all_to_all_crossover()
            elif crossover_type == "best to rest":
                best_individuals = population_sorted[:num_best]
                
                new_agents = best_to_rest()
            elif crossover_type == "hybrid":
                best_individuals = population_sorted[:num_best]
                population.extend(best_individuals)

                new_agents = hybrid_crossover()

            if progressive_mutation and mutation_rate < 0.1 and fn.check_last_n_generations_same(
                best_scores, secondary_scores_of_best
            ):
                mutation_rate += mutation_rate

            for i, agent_pair in enumerate(new_agents):
                mutated_agent1 = fn.mutate(agent_pair[0], mutation_probability=mutation_rate, random_seed=i)
                mutated_agent2 = fn.mutate(agent_pair[1], mutation_probability=mutation_rate, random_seed=i**2)
                population.extend([mutated_agent1, mutated_agent2])

            if crossover_type == "hybrid":
                worst_replacements = [fn.generate_agent(chromosome_length, random_seed=i + population_size) for i in range(num_worst)]
                population.extend(worst_replacements)

        primary_fitness_scores = []
        secondary_fitness_scores = []
        grid_worlds = []
        population_paths = []

        for i, agent_path in enumerate(population):
            primary_fitness_score, secondary_fitness_score, grid_world, path, agent_path = fn.fitness_score_calculation(
                agent_path=agent_path,
                grid_world=initial_grid_world,
                chromosome_length=chromosome_length,
                start_position=start_position,
                end_position=end_position,
                grid_size=grid_size
            )

            population[i] = agent_path

            primary_fitness_scores.append(primary_fitness_score)
            secondary_fitness_scores.append(secondary_fitness_score)
            grid_worlds.append(grid_world)
            population_paths.append(path)

        population_sorted, indices_sorted = fn.population_sorting(
            population, primary_fitness_scores, secondary_fitness_scores
        )

        best_score = np.min(np.array(primary_fitness_scores))
        secondary_score_of_best = secondary_fitness_scores[indices_sorted[0]]
        median_score = round(np.median(np.array(primary_fitness_scores)), 4)
        mean_score = round(np.mean(np.array(primary_fitness_scores)), 4)
        best_grid_world = grid_worlds[indices_sorted[0]]
        best_agent_path = population_paths[indices_sorted[0]]

        best_scores.append(best_score)
        secondary_scores_of_best.append(secondary_score_of_best)
        median_scores.append(median_score)
        mean_scores.append(mean_score)
        best_grid_worlds.append(best_grid_world)
        best_population_paths.append(best_agent_path)

        if verbose == "Full" and (generation == 1 or generation % 10 == 0):
            print(
                f" {generation}. generation finished - best score: {best_score} - median score: {median_score} - mean score: {mean_score} - steps: {secondary_score_of_best}"
            )
            print(line)

        if best_score == 0:
            if first_full_path_generation is None:
                first_full_path_generation = generation

            if early_stop or secondary_score_of_best == num_optimal_steps:
                convergence_flag = True
                if verbose == "Full":
                    print(
                        f" {generation}. generation finished - best score: {best_score} - median score: {median_score} - mean score: {mean_score} - steps: {secondary_score_of_best}"
                    )
                    print(line)
                break

    _, best_indices_sorted = fn.population_sorting(best_population_paths, best_scores, secondary_scores_of_best)
    best_generation = best_indices_sorted[0] + 1
    final_best_score = best_scores[best_indices_sorted[0]]
    final_best_secondary_score = secondary_scores_of_best[best_indices_sorted[0]]

    end_time = time()

    if verbose in ["Restricted", "Full"]:
        if not first_full_path_generation:
            print("Full path is not found!")
        elif not convergence_flag:
            print("Optimal path is not found!")

        print(f"The best generation: {best_generation}")
        print(f"The best primary score: {final_best_score}")
        print(f"The best secondary score: {final_best_secondary_score}")
        
        if first_full_path_generation:
            print(f"The first full path generation: {first_full_path_generation}")

        total_time, time_per_generation, generations_per_second = fn.create_time_report(
            start_time, 
            end_time, 
            generation,
            verbose=True
        )

    else:
        total_time, time_per_generation, generations_per_second = fn.create_time_report(
            start_time, 
            end_time, 
            generation,
            verbose=False
        )

    if verbose == "Full":
        print(double_line)
        print(simulation_finished_message)
        print(double_line)

    return (
        first_full_path_generation,
        best_generation,
        final_best_score,
        final_best_secondary_score,
        total_time,
        time_per_generation,
        generations_per_second,
        best_scores,
        secondary_scores_of_best,
        median_scores,
        mean_scores,
        best_grid_worlds,
        best_population_paths,
        generation,
        primary_fitness_scores,
        secondary_fitness_scores,
        grid_worlds,
        population_paths
    )

def aco_simulation(
    num_iterations: int,
    num_ants: int,
    start_position: Tuple[int, int],
    end_position: Tuple[int, int],
    initial_pheromones: np.ndarray,
    grid_world: np.ndarray,
    alpha: float,
    beta: float,
    max_path_length: int,
    revisit_possible: bool,
    evaporation_rate: float,
    deposit_factor: float,
    pheromone_normalization: bool,
    random_seed: int,
    num_optimal_steps: int,
    pheromone_threshold: float,
    simulation_started_message: str,
    simulation_finished_message: str,
    verbose: Optional[str] = None, 
    line: Optional[str] = 100*'-',
    double_line: Optional[str] = 100*'=',
) -> Tuple[
    Optional[int], Optional[int], bool, str, float, float
]:
    """
    Simulates the Ant Colony Optimization (ACO) process on a grid world and evaluates the paths taken by the ants.
    The function supports verbose output for detailed iteration logs and returns key simulation metrics.

    Parameters:
    ----------
    num_iterations : int
        Total number of iterations for the simulation.
    num_ants : int
        Number of ants in each iteration.
    start_position : Tuple[int, int]
        Starting position of the ants in the grid.
    end_position : Tuple[int, int]
        Target position of the ants in the grid.
    initial_pheromones : np.ndarray
        Initial pheromone levels on the grid.
    grid_world : np.ndarray
        The grid world representation including obstacles and traps.
    alpha : float
        Influence of pheromone levels on movement probabilities.
    beta : float
        Influence of heuristic information on movement probabilities.
    max_path_length : int
        Maximum allowed path length for ants.
    revisit_possible : bool
        Whether ants are allowed to revisit cells in their paths.
    evaporation_rate : float
        Rate at which pheromones evaporate after each iteration.
    deposit_factor : float
        Amount of pheromone deposited by ants per step.
    pheromone_normalization : bool
        If True, normalizes pheromones after updating.
    random_seed : int
        Random seed for reproducibility.
    num_optimal_steps : int
        Number of steps in the optimal path to the target.
    pheromone_threshold : float
        Threshold for determining convergence based on pheromone levels.
    simulation_started_message : str
        Message to display at the start of the simulation.
    simulation_finished_message : str
        Message to display at the end of the simulation.
    verbose : Optional[str], default=None
        Logging level. Options:
        - None: No output is logged.
        - "Restricted": Logs summary metrics at the end of the simulation.
        - "Full": Logs detailed information for each iteration and summary metrics.
    line : Optional[str], default='-' * 100
        The line separator used in verbose logs.
    double_line : Optional[str], default='=' * 100
        The double-line separator used in verbose logs.

    Returns:
    -------
    Tuple:
        - first_full_path (Optional[int]): Iteration number of the first complete path found, or None if not found.
        - first_optimal_path (Optional[int]): Iteration number of the first optimal path found, or None if not found.
        - convergence_iteration (Optional[int]): Iteration number when convergence was achieved, or None if not achieved.
        - total_time (float): Total elapsed time for the simulation in seconds.
        - time_per_iteration (float): Average time taken per iteration in seconds.
        - iterations_per_second (float): Number of iterations completed per second.
        - best_paths (List[List[Tuple[int, int]]]): The best paths found in each iteration.
        - best_scores (List[float]): The best heuristic scores obtained in each iteration.
        - median_scores (List[float]): Median heuristic scores from all paths in each iteration.
        - mean_scores (List[float]): Mean heuristic scores from all paths in each iteration.
    """
    if verbose == "Full":
        print(double_line)
        print(simulation_started_message)
        print(double_line)

    start_time = time()

    first_full_path = None
    first_optimal_path = None
    convergence_iteration = None
    best_secondary_score = float('inf')

    best_paths = []
    best_scores = []
    median_scores = []
    mean_scores = []

    convergence_flag = False
    pheromones = initial_pheromones

    for iteration in range(1, num_iterations + 1):
        all_paths = []
        full_paths = []

        for ant in range(num_ants):
            path = fn.ant_walk(
                start=start_position,
                end=end_position,
                pheromone=pheromones,
                grid_world=grid_world,
                alpha=alpha,
                beta=beta,
                max_path_length=max_path_length,
                revisit_possible=revisit_possible,
                random_seed=random_seed * iteration + ant**2
            )

            all_paths.append(path)

            if path[-1] == end_position:
                if first_full_path is None:
                    first_full_path = iteration

                if best_secondary_score > len(path) - 1:
                    best_secondary_score = len(path) - 1

                full_paths.append(path)

        sorted_paths, sorted_indices, sorted_path_lengths, sorted_heuristic_scores = fn.sort_ant_paths(
            all_paths, end_position
        )
        best_path = sorted_paths[0]
        best_paths.append(best_path)

        best_score = sorted_heuristic_scores[0]
        median_score = round(np.median(np.array(sorted_heuristic_scores)), 4)
        mean_score = round(np.mean(np.array(sorted_heuristic_scores)), 4)

        best_scores.append(best_score)
        median_scores.append(median_score)
        mean_scores.append(mean_score)

        if verbose == "Full":
            print(f" {iteration}. iteration finished - best score: {best_score} - median score: {median_score} - mean score: {mean_score}")
            print(line)

        if full_paths:
            fn.update_pheromones(
                paths=full_paths,
                pheromones=pheromones,
                evaporation_rate=evaporation_rate,
                deposit_factor=deposit_factor,
                pheromone_normalization=pheromone_normalization,
            )

        if (
            first_optimal_path is None
            and best_score == 0
            and sorted_path_lengths[0] - 1 == num_optimal_steps
        ):
            first_optimal_path = iteration

        if first_optimal_path and fn.check_pheromone_path(pheromones, num_optimal_steps, pheromone_threshold):
            convergence_flag = True
            break

    end_time = time()

    if verbose in ["Restricted", "Full"]:
        if first_full_path:
            print(f"First full path iteration: {first_full_path}")
            if first_optimal_path:
                print(f"First optimal path iteration: {first_optimal_path}")
                if convergence_flag:
                    print(f"Convergence achieved in iteration: {iteration}")
                    convergence_iteration = iteration
            else:
                print("Optimal path is not found!")
                print(f"Shortest full path: {best_secondary_score}")
        else:
            print("Full path is not found!")

        total_time, time_per_iteration, iterations_per_second = fn.create_time_report(
            start_time, 
            end_time, 
            iteration,
            verbose=True
        )

    else:
        total_time, time_per_iteration, iterations_per_second = fn.create_time_report(
            start_time, 
            end_time, 
            iteration,
            verbose=False
        )

    if verbose == "Full":
        print(double_line)
        print(simulation_finished_message)
        print(double_line)

    return (
        first_full_path,
        first_optimal_path,
        convergence_iteration,
        total_time,
        time_per_iteration,
        iterations_per_second,
        best_paths,
        best_scores,
        median_scores,
        mean_scores,
    )

