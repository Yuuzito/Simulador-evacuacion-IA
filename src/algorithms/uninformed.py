from collections import deque
from typing import List, Optional, Dict, Set
from src.environment.grid import Grid, Coord


def dfs(grid: Grid, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    # Si el agente ya esta en la meta
    if start == goal:
        return []

    # Pila LIFO y control de ciclos
    stack: List[Coord] = [start]
    visited: Set[Coord] = {start}
    came_from: Dict[Coord, Coord] = {}
    found_goal: bool = False

    # Exploracion en profundidad
    while stack:
        current = stack.pop()
        if current == goal:
            found_goal = True
            break

        # Se filtran muros y celdas con fuego
        for neighbor in grid.get_neighbors(current[0], current[1]):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

    # Si se vacio la pila sin tocar la meta
    if not found_goal:
        return None

    # Reconstruccion del camino desde goal hacia start
    path: List[Coord] = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]

    # Invertir para dejarlo en orden de avance: [paso_1, paso_2, ..., goal]
    path.reverse()
    return path

def bfs(grid: Grid, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    """
    Búsqueda en Anchura (BFS).
    Garantiza el camino con el menor número de movimientos.
    """
    if start == goal:
        return []

    # Cola FIFO en vez de pila LIFO
    queue: deque[Coord] = deque([start])
    visited: Set[Coord] = {start}
    came_from: Dict[Coord, Coord] = {}
    found_goal: bool = False

    while queue:
        current = queue.popleft()

        if current == goal:
            found_goal = True
            break

        for neighbor in grid.get_neighbors(current[0], current[1]):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    if not found_goal:
        return None

    # Reconstrucción del camino
    path: List[Coord] = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path