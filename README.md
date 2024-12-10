# Path Optimization for Grid World Problem Using Metaheuristic Algorithms

## Description

This project is part of my master's thesis titled **"Comparative Examination and Combination of Metaheuristic Algorithms in Path Optimization"**, completed as part of the Master's program in **Artificial Intelligence and Machine Learning** at the **Faculty of Technical Sciences, University of Novi Sad**.

The project focuses on optimizing pathfinding strategies in grid-world environments using **Genetic Algorithms (GA)** and **Ant Colony Optimization (ACO)**. Its primary objective is to efficiently navigate grids with obstacles while producing high-quality solutions by integrating theoretical frameworks with practical implementations. The research explores standalone GA and ACO approaches as well as a hybrid strategy that combines the strengths of both methods.

---

## Key Features

- **Grid World Environment**:
  - Configurable grid dimensions, start and end points, and obstacle placement.
  - Visual representation of the environment and agent paths.
- **Genetic Algorithms**:
  - Path representation using binary bitstring chromosomes.
  - Multiple selection strategies including "all to all," "best to rest," and "hybrid."
  - Adaptive mutation rates to overcome convergence issues.
- **Ant Colony Optimization**:
  - Pheromone-based navigation emphasizing exploration (α) and exploitation (β).
  - Adjustable evaporation rate and deposit factor for tuning convergence.
  - Supports revisitable and non-revisitable grids.
- **Hybrid Approach**:
  - Combines GA for population-based exploration with ACO for local optimization.
  - Enables higher solution quality by leveraging the strengths of both methods.
- **Performance Metrics**:
  - Tracks fitness scores, path length, and computational efficiency.

---

## Dependencies

The project uses a `conda` environment for dependency management. To set up the environment, use the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate metaheuristic_path_optimization
```

---

## Project structure

```
project/
├── functions.py                    # Core utility functions
├── simulations.py                  # GA and ACO simulation logic
├── GA approach.ipynb               # Genetic Algorithm experiments and visualizations
├── ACO approach.ipynb              # Ant Colony Optimization experiments and visualizations
├── Hybrid GA-ACO approach.ipynb    # Hybrid GA-ACO experiments and visualizations
├── GA approach.py                  # Genetic Algorithm implementation
├── ACO approach.py                 # Ant Colony Optimization implementation
├── Hybrid GA-ACO approach.py       # Hybrid GA-ACO implementation
├── optimal_path_length.py          # Optimal path calculation using A*
├── environment.yml                 # Conda environment configuration
├── results/                        # Output logs and visualizations
├── LICENSE                         # Project License
└── README.md                       # Project documentation
```

---

## Author

- Name: Igor Mandarić
- Affiliation: Faculty of Technical Sciences, University of Novi Sad
- Contact: igor.mandarich@protonmail.com

---

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file.

---