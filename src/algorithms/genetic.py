import random
import heapq
from typing import List, Optional, Tuple, Dict, Set, Any
import numpy as np
from src.environment.grid import Grid, Coord


# Direcciones ortogonales de movimiento: Arriba, Abajo, Izquierda, Derecha
DIRECTIONS: List[Coord] = [
    (-1, 0),  # Arriba
    (1, 0),   # Abajo
    (0, -1),  # Izquierda
    (0, 1),   # Derecha
]

# Direcciones opuestas para evitar retrocesos inmediatos innecesarios
OPPOSITE: Dict[Coord, Coord] = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1)
}


def manhattan_distance(a: Coord, b: Coord) -> int:
    """Calcula la distancia Manhattan entre dos coordenadas."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_to_directions(path: List[Coord], start: Coord) -> List[Coord]:
    """Convierte una secuencia de coordenadas en una secuencia de direcciones relativas."""
    dirs: List[Coord] = []
    curr = start
    for p in path:
        dirs.append((p[0] - curr[0], p[1] - curr[1]))
        curr = p
    return dirs


class Individual:
    """
    Representa un individuo en la población del algoritmo genético.
    El cromosoma es una secuencia de movimientos ortogonales (genes).
    """

    def __init__(self, chromosome: List[Coord]):
        self.chromosome: List[Coord] = chromosome
        self.fitness: float = 0.0
        self.reached_goal: bool = False
        self.path: List[Coord] = []
        self.steps_to_goal: int = len(chromosome)


def evaluate_individual(
    individual: Individual,
    grid: Grid,
    start: Coord,
    goal: Coord,
    walkable: np.ndarray,
    cost_map: np.ndarray
) -> None:
    """
    Simula la trayectoria del individuo en la grilla y calcula su aptitud (fitness).
    Premia llegar a la meta rápidamente, penaliza la cercanía al fuego, la congestión y los bucles.
    """
    current = start
    path: List[Coord] = []
    collisions = 0
    total_cost = 0.0
    reached = False
    step_count = 0
    max_steps = len(individual.chromosome)
    rows, cols = grid.rows, grid.cols
    gr, gc = goal

    visited: Set[Coord] = {start}

    for step_idx, direction in enumerate(individual.chromosome):
        step_count = step_idx + 1
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]

        if 0 <= nr < rows and 0 <= nc < cols and walkable[nr, nc]:
            current = (nr, nc)
            path.append(current)
            total_cost += cost_map[nr, nc]

            # Penalización por revisitar la misma celda (evitar bucles o ciclos inútiles)
            if current in visited:
                total_cost += 10.0
            visited.add(current)

            if current == goal:
                reached = True
                break
        else:
            collisions += 1

    individual.reached_goal = reached
    individual.path = path
    individual.steps_to_goal = step_count

    if reached:
        steps_saved = max_steps - step_count
        individual.fitness = 10000.0 + (steps_saved * 20.0) - total_cost
    else:
        dist_to_goal = abs(current[0] - gr) + abs(current[1] - gc)
        individual.fitness = (2000.0 / (dist_to_goal + 1.0)) - (collisions * 3.0) - (total_cost * 0.5)


def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    """Selecciona el individuo con mayor fitness entre un subconjunto aleatorio (torneo)."""
    selected = random.sample(population, tournament_size)
    return max(selected, key=lambda ind: ind.fitness)


def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Cruce espacial de intersección (Geographic Crossover):
    Si los caminos de ambos padres comparten casillas intermedias transitables, se cruzan
    exactamente en ese punto común, preservando la continuidad topológica de la ruta.
    Si no comparten casillas, se aplica cruce estándar de un punto.
    """
    p1_coords = {coord: idx for idx, coord in enumerate(parent1.path)}
    common = []
    for idx2, coord in enumerate(parent2.path):
        if coord in p1_coords and p1_coords[coord] < len(parent1.chromosome) and idx2 < len(parent2.chromosome):
            common.append((p1_coords[coord], idx2))

    if common:
        idx1, idx2 = random.choice(common)
        c1 = parent1.chromosome[:idx1 + 1] + parent2.chromosome[idx2 + 1:]
        c2 = parent2.chromosome[:idx2 + 1] + parent1.chromosome[idx1 + 1:]
        return Individual(c1), Individual(c2)

    # Fallback: cruce de un punto estándar
    min_len = min(len(parent1.chromosome), len(parent2.chromosome))
    if min_len > 2:
        point = random.randint(1, min_len - 1)
        c1 = parent1.chromosome[:point] + parent2.chromosome[point:]
        c2 = parent2.chromosome[:point] + parent1.chromosome[point:]
        return Individual(c1), Individual(c2)

    return Individual(list(parent1.chromosome)), Individual(list(parent2.chromosome))


def mutate(individual: Individual, mutation_rate: float = 0.08) -> None:
    """
    Modifica aleatoriamente algunos genes (direcciones) con probabilidad mutation_rate,
    evitando retrocesos inmediatos sobre la celda previa.
    """
    chrom = individual.chromosome
    for i in range(len(chrom)):
        if random.random() < mutation_rate:
            prev = chrom[i - 1] if i > 0 else None
            choices = [d for d in DIRECTIONS if prev is None or d != OPPOSITE.get(prev)]
            chrom[i] = random.choice(choices) if choices else random.choice(DIRECTIONS)


def heuristic_seed_path(grid: Grid, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    """
    Genera una ruta inicial viable hacia la meta evitando fuego y zonas de peligro
    para semillar la población inicial del algoritmo genético (Algoritmo Genético Híbrido).
    """
    for avoid_danger in (True, False):
        frontier = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start)]
        came_from: Dict[Coord, Coord] = {}
        visited: Set[Coord] = {start}
        counter = 0

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dr, dc in DIRECTIONS:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if grid.in_bounds(nr, nc) and not grid.walls[nr, nc] and not grid.fire[nr, nc]:
                    if avoid_danger and neighbor != goal and grid.is_near_fire(nr, nc):
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        came_from[neighbor] = current
                        counter += 1
                        h = abs(nr - goal[0]) + abs(nc - goal[1])
                        heapq.heappush(frontier, (h, counter, neighbor))

        if goal in came_from or start == goal:
            path = []
            curr = goal
            while curr != start:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            return path

    return None


def genetic(
    grid: Grid,
    start: Coord,
    goal: Coord,
    population_size: int = 30,
    generations: int = 25,
    mutation_rate: float = 0.08,
    crossover_rate: float = 0.8,
    tournament_size: int = 3,
    elitism_count: int = 2,
    max_steps: Optional[int] = None
) -> Optional[List[Coord]]:
    """
    Algoritmo Genético Híbrido para encontrar una ruta de evacuación:
    - Inicialización informada mediante semillado heurístico y caminatas orientadas.
    - Cruce por intersección espacial entre trayectorias continuas.
    - Mutación consciente que penaliza bucles y retrocesos.
    - Función de fitness que optimiza tiempo y castiga severamente la proximidad al fuego.
    - Fallback de mejor esfuerzo (retorna la ruta que más se acerca a la meta si el camino queda cortado).
    
    Retorna la lista de coordenadas [paso_1, ..., goal], o una ruta parcial si la meta está bloqueada.
    """
    if start == goal:
        return []

    rows, cols = grid.rows, grid.cols

    # Precomputar matrices de transitabilidad y costo UNA SOLA VEZ para acelerar drásticamente la evaluación
    walkable = ~grid.walls & ~grid.fire
    cost_map = 1.0 + 0.5 * (grid.congestion ** 2)
    for r in range(rows):
        for c in range(cols):
            if (r, c) != grid.exit_pos and grid.is_near_fire(r, c):
                cost_map[r, c] += 300.0

    if max_steps is None:
        max_steps = int((rows + cols) * 1.5)

    population: List[Individual] = []

    # 1. Semillado Heurístico (Hibridación Genética)
    seed = heuristic_seed_path(grid, start, goal)
    if seed:
        seed_dirs = path_to_directions(seed, start)
        if len(seed_dirs) < max_steps:
            seed_dirs = seed_dirs + [random.choice(DIRECTIONS) for _ in range(max_steps - len(seed_dirs))]
        else:
            seed_dirs = seed_dirs[:max_steps]

        population.append(Individual(list(seed_dirs)))

        # Generar variantes mutadas de la semilla para explorar rutas alternativas (evitar congestión)
        for _ in range(min(4, population_size - 1)):
            variant = Individual(list(seed_dirs))
            mutate(variant, mutation_rate=0.12)
            population.append(variant)

    # 2. Completar población inicial con caminatas dirigidas (filtrando muros y fuego)
    while len(population) < population_size:
        chrom = []
        curr = start
        for _ in range(max_steps):
            valid_dirs = []
            for d in DIRECTIONS:
                nr, nc = curr[0] + d[0], curr[1] + d[1]
                if 0 <= nr < rows and 0 <= nc < cols and walkable[nr, nc]:
                    valid_dirs.append(d)
            chosen_dir = random.choice(valid_dirs) if valid_dirs else random.choice(DIRECTIONS)
            chrom.append(chosen_dir)
            curr = (curr[0] + chosen_dir[0], curr[1] + chosen_dir[1])
        population.append(Individual(chrom))

    best_overall: Optional[Individual] = None

    # 3. Ciclo de generaciones evolutivas
    for gen in range(generations):
        for ind in population:
            if ind.fitness == 0.0:
                evaluate_individual(ind, grid, start, goal, walkable, cost_map)

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        current_best = population[0]

        if best_overall is None or current_best.fitness > best_overall.fitness:
            best_overall = current_best

        # Parada temprana tras algunas generaciones de refinamiento si ya se tiene una solución de alta calidad
        if best_overall.reached_goal and gen >= 5:
            return best_overall.path

        # 4. Elitismo: conservar los mejores individuos intactos
        next_population: List[Individual] = []
        for ind in population[:elitism_count]:
            elite = Individual(list(ind.chromosome))
            elite.fitness = ind.fitness
            elite.reached_goal = ind.reached_goal
            elite.path = ind.path
            next_population.append(elite)

        # 5. Reproducción (Selección por torneo, Cruce y Mutación)
        while len(next_population) < population_size:
            p1 = tournament_selection(population, tournament_size)
            p2 = tournament_selection(population, tournament_size)

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = Individual(list(p1.chromosome)), Individual(list(p2.chromosome))

            mutate(c1, mutation_rate)
            mutate(c2, mutation_rate)

            next_population.append(c1)
            if len(next_population) < population_size:
                next_population.append(c2)

        population = next_population

    # Si se alcanzó la meta en alguna generación, retornar la mejor ruta
    if best_overall and best_overall.reached_goal:
        return best_overall.path

    # Fallback de mejor esfuerzo: si la salida quedó cortada o no se alcanzó, retornar el camino parcial más cercano
    if best_overall and best_overall.path:
        return best_overall.path

    return None


# Alias comunes
genetic_algorithm = genetic
