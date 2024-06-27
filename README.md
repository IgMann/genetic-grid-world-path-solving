# Solving path problems in grid world using genetic algorithms

## Table of Contents

- [Introduction](#introduction) 
- [Usage](#usage)
	- [Environment Setup](##environment-setup)
	- [Running the Notebook](##running-the-notebook)
- [Genetic Algorithm Theory](#genetic-algorithm-theory)
- [Cases Overview](#cases-overview)
  - [Case 1: Genetic Algorithm and Bias](#case-1-genetic-algorithm-and-bias)
  - [Case 2: Selection Strategies and the Pareto Principle](#case-2-selection-strategies-and-the-pareto-principle)
  - [Case 3: Convergence Analysis and Population Segmentation](#case-3-convergence-analysis-and-population-segmentation)
- [Results](#results)
- [License](#license)

## Introduction

This project demonstrates the application of genetic algorithms in a grid world environment. The project includes three cases to showcase different scenarios and optimization challenges with goal to find optimal path in grid world environment. Genetic algorithms are inspired by the process of natural selection and are commonly used to find solutions to complex problems by evolving a population of candidate solutions. Project is made as part of a master’s studies in the AI ​​& ML master’s program at the University of Novi Sad.

## Usage
### Environment Setup

This project uses a conda environment named `genetic_grid_world`. To set up the environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/IgMann/genetic-grid-world-path-solving.git
    cd genetic-grid-world-path-solving
    ```

2. Create the conda environment:
    ```bash
    conda create --name genetic_grid_world python=3.8
    ```

3. Activate the conda environment:
    ```bash
    conda activate genetic_grid_world
    ```

4. Install the required packages:
    ```bash
    conda install numpy pandas matplotlib opencv ffmpeg-python tqdm
    ```

5. Install Jupyter Notebook:
    ```bash
    conda install jupyter
    ```

### Running the Notebook

To run the Jupyter notebook, use the following command:
```bash
jupyter notebook main.ipynb
```

## Genetic Algorithm Theory

Genetic algorithms (GAs) are a class of optimization algorithms that simulate the process of natural selection. They operate on a population of potential solutions, applying the principles of selection, crossover, and mutation to evolve better solutions over generations.
Key Concepts:

    1. Population: A set of candidate solutions to the problem.
    2. Chromosome: A representation of a solution, typically encoded as a string.
    3. Fitness Function: A function that evaluates the quality of a solution.
    4. Selection: The process of choosing the fittest individuals from the population to reproduce.
    5. Crossover (Recombination): Combining parts of two parent solutions to create offspring.
    6. Mutation: Randomly altering parts of a solution to introduce variability.
    7. Generations: Iterations of the algorithm where selection, crossover, and mutation are applied to produce new populations.

Process:

    1. Initialization: Generate an initial population of random solutions.
    2. Evaluation: Compute the fitness of each solution using the fitness function.
    3. Selection: Select the best-performing solutions for reproduction.
    4. Crossover: Combine pairs of selected solutions to create new offspring.
    5. Mutation: Apply random changes to some of the offspring to maintain genetic diversity.
    6. Replacement: Form a new population by replacing some or all of the old population with the new offspring.
    7. Termination: Repeat the process until a stopping criterion is met, such as a maximum number of generations or a satisfactory fitness level.

Genetic algorithms are particularly useful for problems where the search space is large and complex, and traditional optimization methods are impractical.

## Cases Overview

### Case 1: Genetic Algorithm and Bias

In this simulation, we explore the principles of genetic algorithms with a focus on the role of bias in the selection process. Genetic algorithms, inspired by natural selection, use mechanisms such as selection, crossover, and mutation to evolve solutions over generations. By introducing bias in the selection of parents, we aim to emphasize the importance of fitter individuals in guiding the evolution process. This bias ensures that better-performing agents have a higher chance of passing on their traits, leading to a more efficient optimization of paths and behaviors in the grid world.

### Case 2: Selection Strategies and the Pareto Principle

This case delves into the application of the Pareto Principle, or the 80/20 rule, within our genetic algorithm. By segmenting the population into the top 20% best-performing individuals and the remaining 80%, we leverage the principle that a small percentage of individuals can contribute significantly to the overall improvement of the population. The top 20% are prioritized for reproduction, ensuring that their superior traits are propagated, while the rest provide the necessary genetic diversity. This approach balances exploitation and exploration, driving the population towards optimal solutions without stagnation.

### Case 3: Convergence Analysis and Population Segmentation

In this case, we analyze the convergence behavior of our genetic algorithm, focusing on the segmentation of the population into three groups: the top 20% best-performing individuals, the middle 60% representing the rest of the population, and the bottom 20% initially included but later merged into the rest. The best individuals are preserved and prioritized for reproduction, ensuring that their superior traits are consistently passed on to future generations. The middle group provides necessary genetic diversity, promoting exploration and preventing premature convergence. This approach balances exploitation of the best solutions, preservation of superior traits, and exploration of new possibilities, driving the population towards optimal solutions over successive generations.

## Results

Through these cases, we demonstrated the effectiveness of genetic algorithms in optimizing paths within a grid world. The results showed that:

    Case 2: Selection Strategies and the Pareto Principle was the most efficient, achieving the target in just 15 generations. 

    Case 3: Convergence Analysis and Population Segmentation came next, reaching the target in 21 generations. 

    Case 1: Genetic Algorithm and Bias required 35 generations to achieve the target. 

These results underline the potential of genetic algorithms in solving complex optimization problems by mimicking natural evolutionary processes.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.