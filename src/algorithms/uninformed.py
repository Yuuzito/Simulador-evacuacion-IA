from collections import deque
from typing import List, Optional, Dict, Set
from src.environment.grid import Grid, Coord


def _dfs_search(grid: Grid, start: Coord, goal: Coord, avoid_fire_zone: bool) -> Optional[List[Coord]]:
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

        # Se filtran muros, fuego y opcionalmente zona adyacente al fuego
        for neighbor in grid.get_neighbors(current[0], current[1], avoid_fire_zone=avoid_fire_zone):
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


def dfs(grid: Grid, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    # 1. Intentar encontrar ruta segura evitando distancia 1 al fuego
    safe_path = _dfs_search(grid, start, goal, avoid_fire_zone=True)
    if safe_path is not None:
        return safe_path
    # 2. Fallback si no hay otra opción
    return _dfs_search(grid, start, goal, avoid_fire_zone=False)


def _bfs_search(grid: Grid, start: Coord, goal: Coord, avoid_fire_zone: bool) -> Optional[List[Coord]]:
    # Si el agente ya esta en la meta
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

        # Se filtran muros, fuego y opcionalmente zona adyacente al fuego
        for neighbor in grid.get_neighbors(current[0], current[1], avoid_fire_zone=avoid_fire_zone):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    # Si no se alcanzo la meta
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


def bfs(grid: Grid, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    """
    Búsqueda en Anchura (BFS).
    Garantiza el camino con el menor número de movimientos.
    """
    # 1. Intentar encontrar ruta óptima segura evitando distancia 1 al fuego
    safe_path = _bfs_search(grid, start, goal, avoid_fire_zone=True)
    if safe_path is not None:
        return safe_path
    # 2. Fallback si la única vía pasa pegada al fuego
    return _bfs_search(grid, start, goal, avoid_fire_zone=False)