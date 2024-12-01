# Importing modules
import os
import io
import sys
from typing import Tuple, List

import numpy as np
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
    (1, 0), (1, 3), (1, 14), (2, 0), (2, 3), (2, 5), (2, 6), (2, 7), (2, 11), (2, 14), (3, 0), (3, 3), (3, 10), (3, 11), (3, 14), 
    (4, 0), (4, 3), (4, 9), (4, 10), (4, 11), (4, 14), (5, 0), (5, 3), (5, 6), (5, 11), (5, 13), (5, 14), (6, 0), (6, 6), (6, 9), 
    (6, 11), (6, 14), (7, 0), (7, 6), (7, 9), (7, 14), (8, 0), (8, 3), (8, 6), (8, 14), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), 
    (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14)
]
NUM_OPTIMAL_STEPS: int = 20

# Simulation parameters
CHROMOSOME_LENGTH: int = 64
POPULATION_SIZE: int = 100
NUM_GENERATIONS: int = 1000
BIAS: List[float] = [1.5, 3.0]
PROGRESSIVE_MUTATION: List[bool] = [True, False]
MUTATION_RATE: float = 0.01
EARLY_STOP: bool = False
BEST_ONES_PERCENTAGE: float = 0.2
WORST_ONES_PERCENTAGE: float = 0.2
CROSSOVER_TYPES: List[str] = ["all to all", "best to rest", "hybrid"]

# Paths
RESULTS_PATH: str = "./results"
LOGS_PATH: str = f"{RESULTS_PATH}/logs"
GA_RESULTS_LOG_PATH: str = f"{LOGS_PATH}/GA results.log"
GA_RESULTS_CSV_PATH: str = f"{LOGS_PATH}/GA results.csv"

# Other
RANDOM_STATES: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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

        for bias in BIAS:
            for progressive_mutation in PROGRESSIVE_MUTATION:
                for crossover_type in CROSSOVER_TYPES:   
                    print(LINE)
                    print(f"{counter}. simulation")
                    print(LINE)
                    
                    print("\nParameters:\n")
                    print(f"Random state: {random_state}")
                    print(f"Crossover type: {crossover_type}")
                    print(f"Progressive mutation: {progressive_mutation}")
                    print(f"Bias: {bias}\n")

                    print("Results:\n")

                    results = sm.ga_simulation(
                        num_generations=NUM_GENERATIONS,
                        population_size=POPULATION_SIZE,
                        chromosome_length=CHROMOSOME_LENGTH,
                        mutation_rate=MUTATION_RATE,
                        crossover_type=crossover_type,
                        progressive_mutation=progressive_mutation,
                        bias=bias,
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
                        verbose="Restricted",
                        line = LINE,
                        double_line = DOUBLE_LINE  
                    )

                    first_full_path_generation = results[0]
                    best_generation = results[1]
                    final_best_score = results[2]
                    final_best_secondary_score = results[3]
                    total_time = results[4]
                    generations_per_second = results[6]
                    final_generation = results[13]

                    if best_generation == final_generation:
                        optimal_path_generation = final_generation
                    else:
                        optimal_path_generation = None

                    data.append({
                        "Random State": random_state,
                        "Crossover Type": crossover_type,
                        "Progressive Mutation": progressive_mutation,
                        "Bias": bias,
                        "First Full Path Generation": first_full_path_generation,
                        "Optimal Path Generation": optimal_path_generation,
                        "Best Generation": best_generation,
                        "Best Score": final_best_score,
                        "Best Secondary Score": final_best_secondary_score,
                        "Total Time": total_time
                    })

                    counter += 1
    
    print(DOUBLE_LINE)

    # Saving results
    df = pd.DataFrame(data)
    df.to_csv(GA_RESULTS_CSV_PATH, index=False)

    print(f"Results saved in {GA_RESULTS_CSV_PATH} path.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)
    print(EXPERIMENT_FINISHED)
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)

    sys.stdout = original_stdout
    with open(GA_RESULTS_LOG_PATH, "w") as log_file:
        log_file.write(buffer.getvalue())

print(DOUBLE_LINE)
print("GA experiment finished.")
print(f"Logs saved to {GA_RESULTS_LOG_PATH}.")
print(DOUBLE_LINE)








