# Importing modules
import os
import io
import sys
import copy
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
    (1, 0), (1, 3), (1, 14), (2, 0), (2, 3), (2, 5), (2, 6), (2, 7), (2, 11), (2, 14), (3, 0), (3, 3), (3, 10), (3, 11), (3, 13), 
    (3, 14), (4, 0), (4, 3), (4, 9), (4, 10), (4, 11), (4, 14), (5, 0), (5, 3), (5, 6), (5, 11), (5, 13), (5, 14), (6, 0), (6, 6), 
    (6, 9), (6, 11), (6, 14), (7, 0), (7, 6), (7, 9), (7, 14), (8, 0), (8, 3), (8, 6), (8, 14), (9, 0), (9, 1), (9, 2), (9, 3), 
    (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14)
]
NUM_OPTIMAL_STEPS: int = 20

# Simulation parameters
ALPHA: List[float] = [1.5, 3.0]
BETA: List[float] = [1.5, 3.0]
EVAPORATION_RATE: float = 0.5
DEPOSIT_FACTOR: int = 10
NUM_ANTS: int = 100
NUM_ITERATIONS: int = 1000
MAX_PATH_LENGTH: int = 32
REVISIT_POSSIBILITY: List[bool] = [True, False]  
PHEROMONE_NORMALIZATION: bool = True
PHEROMONE_THRESHOLD: float = 0.25

# Paths
RESULTS_PATH: str = "./results"
LOGS_PATH: str = f"{RESULTS_PATH}/logs"
ACO_RESULTS_LOG_PATH: str = f"{LOGS_PATH}/ACO results.log"
ACO_RESULTS_CSV_PATH: str = f"{LOGS_PATH}/ACO results.csv"

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

# Grid world and pheromone initialization
initial_grid_world = fn.grid_world_creation(GRID_SIZE, START_POSITION, END_POSITION, OBSTACLES)
initial_pheromones = np.ones(GRID_SIZE)

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

        for revisit_possible in REVISIT_POSSIBILITY:
            for alpha in ALPHA:
                for beta in BETA:
                    print(LINE)
                    print(f"{counter}. simulation")
                    print(LINE)
                    
                    print("\nParameters:\n")
                    print(f"Random state: {random_state}")
                    print(f"Revisit possible: {revisit_possible}")
                    print(f"Alpha: {alpha}")
                    print(f"Beta: {beta}\n")

                    print("Results:\n")

                    pheromones = copy.deepcopy(initial_pheromones)

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
                        revisit_possible=revisit_possible,
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

                    first_full_path = results[0]
                    first_optimal_path = results[1]
                    convergence_iteration = results[2]
                    total_time = results[3]
                    iterations_per_second = results[5]

                    data.append({
                        "Random State": random_state,
                        "Alpha": alpha,
                        "Beta": beta,
                        "Revisit Possible": revisit_possible,
                        "First Full Path": first_full_path,
                        "First Optimal Path": first_optimal_path,
                        "Convergence Iteration": convergence_iteration,
                        "Total Time": total_time
                    })

                    counter += 1

    print(DOUBLE_LINE)

    # Saving results
    df = pd.DataFrame(data)
    df.to_csv(ACO_RESULTS_CSV_PATH, index=False)

    print(f"Results saved in {ACO_RESULTS_CSV_PATH} path.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)
    print(EXPERIMENT_FINISHED)
    print(DOUBLE_LINE)
    print(DOUBLE_LINE)

    sys.stdout = original_stdout
    with open(ACO_RESULTS_LOG_PATH, "w") as log_file:
        log_file.write(buffer.getvalue())

print(DOUBLE_LINE)
print("ACO experiment finished.")
print(f"Logs saved to {ACO_RESULTS_LOG_PATH}.")
print(DOUBLE_LINE)








