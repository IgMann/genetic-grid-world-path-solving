# Importing modules
import heapq
from typing import Tuple, List, Set, Optional

# Parameters
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
LINE: str = 100 * '-'
DOUBLE_LINE: str = 100 * '='

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculates the Manhattan distance between two points.

    Parameters:
    a (Tuple[int, int]): The first point (row, column).
    b (Tuple[int, int]): The second point (row, column).

    Returns:
    int: The Manhattan distance between the two points.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start: Tuple[int, int], end: Tuple[int, int], obstacles: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """
    Finds the shortest path from start to end in a grid using the A* algorithm.

    Parameters:
    start (Tuple[int, int]): The starting point (row, column).
    end (Tuple[int, int]): The target point (row, column).
    obstacles (Set[Tuple[int, int]]): A set of coordinates representing obstacles.

    Returns:
    Optional[List[Tuple[int, int]]]: A list of coordinates representing the path if found, or None if no path exists.
    """
    open_set: List[Tuple[int, Tuple[int, int]]] = []
    heapq.heappush(open_set, (0, start))
    came_from: dict = {}
    g_score: dict = {start: 0}
    f_score: dict = {start: heuristic(start, end)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == end:
            path: List[Tuple[int, int]] = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < GRID_SIZE[0] and 0 <= neighbor[1] < GRID_SIZE[1] and neighbor not in obstacles):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None

optimal_path: Optional[List[Tuple[int, int]]] = a_star(START_POSITION, END_POSITION, set(OBSTACLES))

print(DOUBLE_LINE)
print(f"Optimal path: {optimal_path}")
print(LINE)
print(f"Optimal path length: {len(optimal_path) if optimal_path else 'No path found'}")
print(LINE)
print(f"Optimal number of steps: {len(optimal_path) - 1 if optimal_path else 'No path found'}")
print(DOUBLE_LINE)
