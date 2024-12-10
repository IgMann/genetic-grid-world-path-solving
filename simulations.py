# Importing modules
from time import time
from typing import Tuple, List, Optional

import numpy as np

import functions as fn

def ga_simulation(
    num_generations: int,
    population_size: int,
    chromosome_length: int,
    initial_mutation_rate: float,
    selection_type: str,
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
    recode_path: Optional[bool] = True,
    revisit_possible: Optional[bool] = False,
    verbose: Optional[str] = None, 
    line: Optional[str] = 100*'-',
    double_line: Optional[str] = 100*'=',
) -> Tuple[
    Optional[int], int, float, int, str, float, float, List[float], List[int], List[float], List[float], List[np.ndarray], List[List[Tuple[int, int]]], int
    ]:
    """
    Simulates a Genetic Algorithm (GA) for solving the pathfinding problem in a grid world.

    Args:
        num_generations (int): The total number of generations to simulate.
        population_size (int): The size of the agent population.
        chromosome_length (int): The length of the chromosome representing each agent's path.
        initial_mutation_rate (float): The initial mutation rate for the genetic algorithm.
        selection_type (str): The type of selection strategy to use ("all to all", "best to rest", or "hybrid").
        progressive_mutation (bool): Enables dynamic mutation rate adjustment when progress stagnates.
        bias (int): The bias factor for rank-based selection.
        early_stop (bool): Whether to terminate the simulation when an optimal path is found.
        best_ones_percentage (float): Fraction of top-performing individuals for breeding.
        worst_ones_percentage (float): Fraction of worst-performing individuals replaced in "hybrid" selection.
        num_optimal_steps (int): The optimal number of steps to complete the path.
        start_position (Tuple[int, int]): The starting position in the grid world.
        end_position (Tuple[int, int]): The ending position in the grid world.
        grid_size (Tuple[int, int]): The dimensions of the grid world.
        initial_grid_world (np.ndarray): The initial grid world configuration.
        random_seed (int): Seed for ensuring reproducibility of random operations.
        simulation_started_message (str): Message displayed at the start of the simulation.
        simulation_finished_message (str): Message displayed at the end of the simulation.
        recode_path (Optional[bool], default=True): If True, rewrites the path in the grid world matrix.
        revisit_possible (Optional[bool], default=False): Allows revisiting grid cells if True.
        verbose (Optional[str], default=None): Sets the verbosity level of the output ("Full", "Restricted", or None).
        line (Optional[str], default="-" * 100): Line separator for verbose output.
        double_line (Optional[str], default="=" * 100): Double line separator for verbose sections.

    Returns:
        Tuple:
            - first_full_path_generation (Optional[int]): The generation where the first optimal path was discovered.
            - best_generation (int): The generation containing the best overall solution.
            - final_best_score (float): The best primary fitness score achieved.
            - final_best_secondary_score (int): The secondary fitness score (e.g., path length) of the best solution.
            - total_time (float): The total runtime of the simulation in seconds.
            - time_per_generation (float): The average time per generation in seconds.
            - generations_per_second (float): The number of generations processed per second.
            - best_scores (List[float]): The best fitness scores across all generations.
            - secondary_scores_of_best (List[int]): Secondary scores of the best solutions for each generation.
            - median_scores (List[float]): Median fitness scores for each generation.
            - mean_scores (List[float]): Mean fitness scores for each generation.
            - best_grid_worlds (List[np.ndarray]): Grid configurations of the best solutions in each generation.
            - best_population_paths (List[List[Tuple[int, int]]]): The paths taken by the best agents in each generation.
            - generation_count (int): The total number of generations processed.
    """
    def all_to_all_selection() -> List[Tuple[str, str]]:
        """
        Implements the "all to all" selection strategy for pairing agents in the population.

        This strategy pairs agents by selecting two individuals from the population repeatedly until 
        the desired number of pairs is created. The selection is rank-based, and agents are not 
        paired with themselves. The resulting pairs undergo crossover to produce new offspring.

        Args:
            None

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple represents a pair of offspring 
            resulting from the crossover operation.

        Details:
            - For each pair, two agents are selected using rank-based selection with a bias parameter.
            - Ensures that the two selected agents are not the same.
            - The selected agents are combined via a crossover operation to generate offspring.
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

    def best_to_rest_selection() -> List[Tuple[str, str]]:
        """
        Implements the "best to rest" selection strategy for pairing agents in the population.

        This strategy selects one agent from the top-performing individuals and another agent 
        from the remaining population for each pair. The selected pairs undergo crossover to 
        produce new offspring.

        Args:
            None

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple represents a pair of offspring 
            resulting from the crossover operation.

        Details:
            - The top-performing individuals (the "best") are selected based on rank.
            - The rest of the population serves as the pool for pairing with the best individuals.
            - Each pair of agents is generated by selecting one agent from the "best" group 
            and one from the "rest" group, ensuring they are not identical.
            - The selected agents are combined via a crossover operation to generate offspring.
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

    def hybrid_selection() -> List[Tuple[str, str]]:
        """
        Implements the "hybrid" selection strategy for pairing agents in the population.

        This strategy combines individuals from the best, middle, and worst performing groups 
        to maintain diversity while promoting strong candidates. Pairs are formed by selecting 
        agents from the best and middle groups, with additional replacements for the worst agents.

        Args:
            None

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple represents a pair of offspring 
            resulting from the crossover operation.

        Details:
            - The population is divided into three groups: best, middle, and worst performers.
            - Agents from the best and middle groups are paired for crossover.
            - The middle group ensures diversity by including individuals with intermediate performance.
            - Additional replacements are generated for worst-performing agents.
            - The resulting pairs undergo crossover to produce offspring for the next generation.
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

    if selection_type != "all to all":
        num_best = int(population_size * best_ones_percentage)

        if selection_type == "best to rest":
            pass
        elif selection_type == "hybrid":
            num_worst = int(population_size * worst_ones_percentage)
            num_middle = population_size - num_best - num_worst
        else:
            raise ValueError("Selection type could be only \"all to all\", \"best to rest\" or \"hybrid\".")

    best_scores = []
    secondary_scores_of_best = []
    median_scores = []
    mean_scores = []
    best_grid_worlds = []
    best_population_paths = []
    mutation_rate = initial_mutation_rate
    first_full_path_generation = None
    convergence_flag = False

    for generation in range(1, num_generations + 1):
        if generation == 1:
            population = [fn.generate_agent(chromosome_length, random_seed=i) for i in range(population_size)]
        else:
            population = []

            if selection_type == "all to all":
                new_agents = all_to_all_selection()
            elif selection_type == "best to rest":
                best_individuals = population_sorted[:num_best]
                
                new_agents = best_to_rest_selection()
            elif selection_type == "hybrid":
                best_individuals = population_sorted[:num_best]
                population.extend(best_individuals)

                new_agents = hybrid_selection()

            if progressive_mutation and mutation_rate < 0.1 and fn.check_last_n_generations_same(
                best_scores, secondary_scores_of_best
            ):
                mutation_rate += initial_mutation_rate
            else:
                mutation_rate = initial_mutation_rate

            for i, agent_pair in enumerate(new_agents):
                mutated_agent1 = fn.mutate(agent_pair[0], mutation_probability=mutation_rate, random_seed=i)
                mutated_agent2 = fn.mutate(agent_pair[1], mutation_probability=mutation_rate, random_seed=i**2)
                population.extend([mutated_agent1, mutated_agent2])

            if selection_type == "hybrid":
                unique_population = population_sorted[:num_best]
                seen_individuals = set(unique_population)
                num_removed = 0

                for agent in population_sorted[num_best:num_best + num_middle]:
                    if agent not in seen_individuals:
                        unique_population.append(agent)
                        seen_individuals.add(agent)
                    else:
                        num_removed += 1

                population = unique_population

                worst_replacements = [
                    fn.generate_agent(chromosome_length, random_seed=i + population_size)
                    for i in range(num_worst + num_removed)
                ]
                population.extend(worst_replacements)

                population = population[:population_size]

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
                grid_size=grid_size,
                num_optimal_steps=num_optimal_steps,
                recode_path=recode_path,
                revisit_possible=revisit_possible
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
        int(first_full_path_generation) if first_full_path_generation is not None else None,
        int(best_generation) if best_generation is not None else None,
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
        int(generation),
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
    shortest_full_path = float('inf')
    global_best_path = ['' for _ in range(100)]

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

                if shortest_full_path > len(path):
                    shortest_full_path = len(path)

                full_paths.append(path)

        sorted_paths, sorted_indices, sorted_path_lengths, sorted_heuristic_scores = fn.sort_ant_paths(
            all_paths, 
            end_position
        )
        best_path = sorted_paths[0]
        best_paths.append(best_path)

        if len(best_path) < len(global_best_path):
            global_best_path = best_path

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
            if first_optimal_path:
                fn.update_pheromones(
                paths=[best_path, global_best_path],
                pheromones=pheromones,
                evaporation_rate=evaporation_rate,
                deposit_factor=deposit_factor,
                pheromone_normalization=pheromone_normalization,
            )

            else:
                full_paths.append(global_best_path)

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
                print(f"Shortest full path: {shortest_full_path}")
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
        int(first_full_path) if first_full_path is not None else None,
        int(first_optimal_path) if first_optimal_path is not None else None,
        int(convergence_iteration) if convergence_iteration is not None else None,
        total_time,
        time_per_iteration,
        iterations_per_second,
        best_paths,
        best_scores,
        median_scores,
        mean_scores,
        int(iteration)
    )

