import heapq
import math
from typing import List, Optional, Dict, Set, Callable
from src.environment.grid import Grid, Coord


def manhattan_distance(a: Coord, b: Coord) -> float:
    #Calcula la distancia Manhattan entre dos coordenadas.
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

def euclidean_distance(a: Coord, b: Coord) -> float:
    #Calcula la distancia Euclidiana entre dos coordenadas.
    return math.hypot(a[0] - b[0], a[1] - b[1])

"""
Expande el nodo mas prometedor guiandose unicamente por la funcion heuristica h(n),
no garantiza encontrar el camino de menor costo.
"""
def _greedy_search(
    grid: Grid,
    start: Coord,
    goal: Coord,
    heuristic: Callable[[Coord, Coord], float],
    avoid_fire_zone: bool
) -> Optional[List[Coord]]:
    # Si el agente ya esta en la meta
    if start == goal:
        return []

    # Cola de prioridad: almacena tuplas de (h_score, contador_desempate, coordenada)
    counter = 0
    frontier: List[tuple] = []
    heapq.heappush(frontier, (heuristic(start, goal), counter, start))

    visited: Set[Coord] = {start}
    came_from: Dict[Coord, Coord] = {}
    found_goal: bool = False

    # Mientras la cola no este vacia, es decir, mientras queden nodos por explorar
    while len(frontier) > 0:
        # Se extrae la coordenada del nodo con mayor prioridad (menor heuristica)
        _, _, current = heapq.heappop(frontier)

        # Si se alcanzo la meta
        if current == goal:
            found_goal = True
            break

        # Se exploran los vecinos de "current" (nodo actual) y se agregan a la cola si no han sido visitados
        for neighbor in grid.get_neighbors(current[0], current[1], avoid_fire_zone=avoid_fire_zone):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                counter += 1
                h = heuristic(neighbor, goal)
                heapq.heappush(frontier, (h, counter, neighbor))

    # Si no se alcanzo la meta
    if not found_goal:
        return None

    # Reconstruccion del camino desde goal hacia start
    path: List[Coord] = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


def greedy(
    grid: Grid,
    start: Coord,
    goal: Coord,
    heuristic: Callable[[Coord, Coord], float] = manhattan_distance
) -> Optional[List[Coord]]: 
    # 1. Intentar encontrar una ruta segura que evite celdas a distancia 1 del fuego
    safe_path = _greedy_search(grid, start, goal, heuristic, avoid_fire_zone=True)
    if safe_path is not None:
        return safe_path
    # 2. Si no hay camino seguro (único pasillo roza el fuego), fallback para escapar
    return _greedy_search(grid, start, goal, heuristic, avoid_fire_zone=False)


"""
Encuentra el camino optimo evaluando f(n) = g(n) + h(n),
donde g(n) es el costo acumulado considerando congestión/paso y h(n) es la heurística admisible.
"""
def _a_star_search(
    grid: Grid,
    start: Coord,
    goal: Coord,
    heuristic: Callable[[Coord, Coord], float],
    avoid_fire_zone: bool
) -> Optional[List[Coord]]:
    # Si el agente ya esta en la meta
    if start == goal:
        return []

    # Cola de prioridad: almacena tuplas de (f_score, contador_desempate, coordenada)
    counter = 0
    frontier: List[tuple] = []
    
    # Costo acumulado g_score
    g_score: Dict[Coord, float] = {start: 0.0}
    f_start = heuristic(start, goal)
    heapq.heappush(frontier, (f_start, counter, start))

    came_from: Dict[Coord, Coord] = {}
    closed_set: Set[Coord] = set()
    found_goal: bool = False

    while frontier:
        current_f, _, current = heapq.heappop(frontier)

        # Si se alcanzo la meta
        if current == goal:
            found_goal = True
            break

        # Si ya cerramos este nodo, lo ignoramos
        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor in grid.get_neighbors(current[0], current[1], avoid_fire_zone=avoid_fire_zone):
            if neighbor in closed_set:
                continue

            # Consideramos el costo de transitar a la celda vecina (incluyendo congestión y proximidad al fuego)
            step_cost = grid.get_cost(neighbor[0], neighbor[1])
            tentative_g = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(frontier, (f_score, counter, neighbor))

    # Si no se alcanzo la meta
    if not found_goal:
        return None

    # Reconstrucción del camino desde goal hacia start
    path: List[Coord] = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


def a_star(
    grid: Grid,
    start: Coord,
    goal: Coord,
    heuristic: Callable[[Coord, Coord], float] = manhattan_distance
) -> Optional[List[Coord]]:
    # 1. Intentar encontrar ruta óptima con margen de seguridad (evitando distancia 1 al fuego)
    safe_path = _a_star_search(grid, start, goal, heuristic, avoid_fire_zone=True)
    if safe_path is not None:
        return safe_path
    # 2. Fallback si la única opción disponible bordea el fuego
    return _a_star_search(grid, start, goal, heuristic, avoid_fire_zone=False)


# Alias comunes
greedy_best_first_search = greedy
astar = a_star
