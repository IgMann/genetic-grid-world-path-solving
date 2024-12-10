# Importing modules
import os
import io
import sys
import time
from typing import Tuple, List

import pandas as pd

import functions as fn
import simulations as sm

# Parameters
# Grid parameters
GRID_SIZE: Tuple[int, int] = (10, 15)
START_POSITION: Tuple[int, int] = (6, 1)
END_POSITION: Tuple[int, int] = (4, 13)
OBSTACLES: List[Tuple[int, int]] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14),
    (1, 0), (1, 3), (1, 14), (2, 0), (2, 3), (2, 5), (2, 6), (2, 7), (2, 11), (2, 14), (3, 0), (3, 3), (3, 10), (3, 11), (3, 13), 
    (3, 14), (4, 0), (4, 3), (4, 9), (4, 10), (4, 11), (4, 14), (5, 0), (5, 3), (5, 6), (5, 11), (5, 13), (5, 14), (6, 0), (6, 6), 
    (6, 9), (6, 11), (6, 14), (7, 0), (7, 6), (7, 9), (7, 14), (8, 0), (8, 3), (8, 6), (8, 14), (9, 0), (9, 1), (9, 2), (9, 3), 
    (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14)
]
NUM_OPTIMAL_STEPS: int = 20

# Simulation parameters - Genetic Algorithms
CHROMOSOME_LENGTH: int = 64
POPULATION_SIZE: int = 100
NUM_GENERATIONS: int = 50
OPTIMAL_PATH_LENGTH: int = 21
BIAS: float = 2.25
PROGRESSIVE_MUTATION: bool = True
MUTATION_RATE: float = 0.01
EARLY_STOP: bool = False
BEST_ONES_PERCENTAGE: float = 0.1
WORST_ONES_PERCENTAGE: float = 0.2
NUM_BEST_PATHS: int = 5
SELECTION_TYPES: List[str] = ["all to all", "best to rest", "hybrid"]

# Simulation parameters - Ant Colony Optimization
ALPHA: List[float] = [1.5, 3.0]
BETA: List[float] = [1.5, 3.0]
EVAPORATION_RATE: float = 0.5
DEPOSIT_FACTOR: int = 100
NUM_ANTS: int = 100
NUM_ITERATIONS: int = 1000
MAX_PATH_LENGTH: int = 33
PHEROMONE_NORMALIZATION: bool = True
PHEROMONE_THRESHOLD: float = 0.25
PATH_SCALING_FACTOR: int = 10

# Paths
RESULTS_PATH: str = "./results"
LOGS_PATH: str = f"{RESULTS_PATH}/logs"
GA_ACO_RESULTS_LOG_PATH: str = f"{LOGS_PATH}/GA-ACO results.log"
GA_ACO_RESULTS_CSV_PATH: str = f"{LOGS_PATH}/GA-ACO results.csv"

# Other
RANDOM_STATES: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
REVISIT_POSSIBILITY: List[bool] = [True, False] 
LINE: str = '-' * 100
DOUBLE_LINE: str = '=' * 100
EXPERIMENT_STARTED: str = '*' * 36 + " !!! EXPERIMENT STARTED !!! " + '*' * 36
EXPERIMENT_FINISHED: str = '*' * 36 + " !!! EXPERIMENT FINISHED !!! " + '*' * 35
SIMULATION_STARTED: str = 36*'-' + " !!! SIMULATION STARTED !!! " + 36*'-'
SIMULATION_FINISHED: str = 36*'-' + " !!! SIMULATION FINISHED !!! " + 35*'-'

# Results directories creation
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Grid world initialization
initial_grid_world = fn.grid_world_creation(GRID_SIZE, START_POSITION, END_POSITION, OBSTACLES)

# Logging start
buffer = io.StringIO()
original_stdout = sys.stdout
sys.stdout = buffer

try:
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)
    print(EXPERIMENT_STARTED)
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)

    print()

    data = []
    counter = 1
    for random_state in RANDOM_STATES: 
        print(DOUBLE_LINE)
        print(f"RANDOM STATE: {random_state}")
        print(DOUBLE_LINE)

        for selection_type in SELECTION_TYPES:
            for ga_revisit_possible in REVISIT_POSSIBILITY:
                for aco_revisit_possible in REVISIT_POSSIBILITY:
                    for alpha in ALPHA:
                        for beta in BETA:  
                            print(LINE)
                            print(f"{counter}. simulation")
                            print(LINE)
                            
                            print("\nParameters:\n")
                            print(f"Random state: {random_state}")
                            print(f"Revisit possible - GA: {ga_revisit_possible}")
                            print(f"Selection type: {selection_type}")
                            print(f"Revisit possible - ACO: {aco_revisit_possible}")
                            print(f"Alpha: {alpha}")
                            print(f"Beta: {beta}\n")

                            print("Results:\n")

                            start_time = time.time()

                            results = sm.ga_simulation(
                                num_generations=NUM_GENERATIONS,
                                population_size=POPULATION_SIZE,
                                chromosome_length=CHROMOSOME_LENGTH,
                                initial_mutation_rate=MUTATION_RATE,
                                selection_type=selection_type,
                                progressive_mutation=PROGRESSIVE_MUTATION,
                                bias=BIAS,
                                early_stop=EARLY_STOP,
                                best_ones_percentage=BEST_ONES_PERCENTAGE,
                                worst_ones_percentage=WORST_ONES_PERCENTAGE,
                                num_optimal_steps=NUM_OPTIMAL_STEPS,
                                start_position=START_POSITION,
                                end_position=END_POSITION,
                                grid_size=GRID_SIZE,
                                initial_grid_world=initial_grid_world,
                                random_seed=random_state,
                                simulation_started_message=SIMULATION_STARTED,
                                simulation_finished_message=SIMULATION_FINISHED,
                                revisit_possible = ga_revisit_possible,
                                verbose="Restricted",
                                line = LINE,
                                double_line = DOUBLE_LINE  
                            )

                            ga_best_generation = results[1]
                            ga_final_best_score = results[2]
                            ga_final_best_secondary_score = results[3]
                            ga_total_time = results[4]
                            ga_generations_per_second = results[6]
                            primary_fitness_scores = results[14]
                            secondary_fitness_scores = results[15]
                            population_paths = results[17]

                            if ga_final_best_score == 0 and ga_final_best_secondary_score == 20:
                                aco_first_full_path = None
                                aco_first_optimal_path = None
                                aco_convergence_iteration = None
                                aco_total_time = None
                                aco_iterations_per_second = None
                            else:
                                # Pheromones initialization
                                selected_paths, _ = fn.find_different_paths(
                                    primary_fitness_scores = primary_fitness_scores, 
                                    secondary_fitness_scores = secondary_fitness_scores, 
                                    population_paths = population_paths,
                                    N = NUM_BEST_PATHS
                                )

                                best_positions = []

                                if ga_revisit_possible:
                                    for selected_path in selected_paths:
                                        unique_path = []
                                        seen = set()
                                        for position in selected_path:
                                            if position not in seen:
                                                unique_path.append(position)
                                                seen.add(position)
                                        
                                        # Process the unique path
                                        if len(unique_path) > OPTIMAL_PATH_LENGTH:
                                            best_positions.extend(unique_path[:OPTIMAL_PATH_LENGTH])
                                        else:
                                            best_positions.extend(unique_path)

                                else:
                                    for selected_path in selected_paths:
                                        if len(selected_path) > OPTIMAL_PATH_LENGTH:
                                                best_positions.extend(selected_path[:OPTIMAL_PATH_LENGTH])
                                        else:
                                            best_positions.extend(selected_path)

                                pheromones = ((PATH_SCALING_FACTOR - 1) * fn.create_pheromones_matrix(best_positions, GRID_SIZE) + 1) / PATH_SCALING_FACTOR

                                results = sm.aco_simulation(
                                    num_iterations=NUM_ITERATIONS,
                                    num_ants=NUM_ANTS,
                                    start_position=START_POSITION,
                                    end_position=END_POSITION,
                                    initial_pheromones=pheromones,
                                    grid_world=initial_grid_world,
                                    alpha=alpha,
                                    beta=beta,
                                    max_path_length=MAX_PATH_LENGTH,
                                    revisit_possible=aco_revisit_possible,
                                    evaporation_rate=EVAPORATION_RATE,
                                    deposit_factor=DEPOSIT_FACTOR,
                                    pheromone_normalization=PHEROMONE_NORMALIZATION,
                                    random_seed=random_state,
                                    num_optimal_steps=NUM_OPTIMAL_STEPS,
                                    pheromone_threshold=PHEROMONE_THRESHOLD,
                                    simulation_started_message=SIMULATION_STARTED,
                                    simulation_finished_message=SIMULATION_FINISHED,
                                    verbose="Restricted"  
                                )

                                aco_first_full_path = results[0]
                                aco_first_optimal_path = results[1]
                                aco_convergence_iteration = results[2]
                                aco_total_time = results[3]
                                aco_iterations_per_second = results[5]

                            end_time = time.time()

                            total_time = round(end_time - start_time, 2)

                            data.append({
                                "Random State": random_state,
                                "GA Revisit Possible": ga_revisit_possible,
                                "Selection Type": selection_type,
                                "ACO Revisit Possible": aco_revisit_possible,
                                "Alpha": alpha,
                                "Beta": beta,
                                "GA Best Generation": ga_best_generation,
                                "GA Best Score": ga_final_best_score,
                                "GA Best Secondary Score": ga_final_best_secondary_score,
                                "GA Generations per Second": ga_generations_per_second,
                                "ACO First Full Path": aco_first_full_path,
                                "ACO First Optimal Path": aco_first_optimal_path,
                                "ACO Convergence Iteration": aco_convergence_iteration,
                                "ACO Iterations per Second": aco_iterations_per_second,
                                "Total Time": total_time
                            })

                            print(f"\nTotal GA-ACO time: {total_time}\n")

                            counter += 1
    
    print(DOUBLE_LINE)

    # Saving results
    df = pd.DataFrame(data)
    df.to_csv(GA_ACO_RESULTS_CSV_PATH, index=False)

    print(f"Results saved in {GA_ACO_RESULTS_CSV_PATH} path.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)
    print(EXPERIMENT_FINISHED)
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)

    sys.stdout = original_stdout
    with open(GA_ACO_RESULTS_LOG_PATH, "w") as log_file:
        log_file.write(buffer.getvalue())

print(DOUBLE_LINE)
print("GA-ACO experiment finished.")
print(f"Logs saved to {GA_ACO_RESULTS_LOG_PATH}.")
print(DOUBLE_LINE)








