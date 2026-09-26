import random
from typing import List, Optional, Tuple, Any
from src.environment.grid import Grid, Coord


# Direcciones ortogonales de movimiento: Arriba, Abajo, Izquierda, Derecha
DIRECTIONS: List[Coord] = [
    (-1, 0),  # Arriba
    (1, 0),   # Abajo
    (0, -1),  # Izquierda
    (0, 1),   # Derecha
]


def manhattan_distance(a: Coord, b: Coord) -> int:
    # Calcula la distancia Manhattan entre dos coordenadas.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Individual:
    """
    Representa un individuo en la poblacion del algoritmo genetico.
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
    walkable: Optional[Any] = None,
    cost_map: Optional[Any] = None
) -> None:
    """
    Simula la trayectoria del individuo en la grilla y calcula su aptitud (fitness).
    Premia llegar a la meta rapidamente y penaliza quedar lejos o chocar contra obstaculos/fuego.
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

    # Evaluación rápida usando matrices precomputadas si están disponibles
    use_fast_maps = walkable is not None and cost_map is not None

    for step_idx, direction in enumerate(individual.chromosome):
        step_count = step_idx + 1
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]

        is_step_valid = (
            (0 <= nr < rows and 0 <= nc < cols and walkable[nr, nc])
            if use_fast_maps
            else grid.is_walkable(nr, nc)
        )

        if is_step_valid:
            current = (nr, nc)
            path.append(current)
            total_cost += cost_map[nr, nc] if use_fast_maps else grid.get_cost(nr, nc)

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
        individual.fitness = 10000.0 + (steps_saved * 50.0) - total_cost
    else:
        dist_to_goal = abs(current[0] - gr) + abs(current[1] - gc)
        individual.fitness = (1000.0 / (dist_to_goal + 1.0)) - (collisions * 2.0)


def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    # Selecciona el individuo con mayor fitness entre un subconjunto aleatorio (torneo)
    selected = random.sample(population, tournament_size)
    return max(selected, key=lambda ind: ind.fitness)


def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    # Realiza un cruce de un punto entre dos padres
    point = random.randint(1, len(parent1.chromosome) - 1)
    child1_chrom = parent1.chromosome[:point] + parent2.chromosome[point:]
    child2_chrom = parent2.chromosome[:point] + parent1.chromosome[point:]
    return Individual(child1_chrom), Individual(child2_chrom)


def mutate(individual: Individual, mutation_rate: float = 0.05) -> None:
    # Modifica aleatoriamente algunos genes (direcciones) con probabilidad mutation_rate
    chrom = individual.chromosome
    for i in range(len(chrom)):
        if random.random() < mutation_rate:
            chrom[i] = random.choice(DIRECTIONS)


def genetic(
    grid: Grid,
    start: Coord,
    goal: Coord,
    population_size: int = 40,
    generations: int = 40,
    mutation_rate: float = 0.06,
    crossover_rate: float = 0.8,
    tournament_size: int = 3,
    elitism_count: int = 2,
    max_steps: Optional[int] = None
) -> Optional[List[Coord]]:
    """
    Algoritmo Genetico para encontrar una ruta de evacuacion.
    
    Retorna la lista de coordenadas del camino desde el inicio hasta la meta [paso_1, ..., goal],
    o None si ninguna generacion logro alcanzar la meta.
    """
    # Si el agente ya esta en la meta
    if start == goal:
        return []

    rows, cols = grid.rows, grid.cols

    # Precomputar mapas de transitabilidad y costo UNA SOLA VEZ para acelerar drásticamente la evaluación
    walkable = ~grid.walls & ~grid.fire
    cost_map = 1.0 + 0.5 * (grid.congestion ** 2)
    for r in range(rows):
        for c in range(cols):
            if (r, c) != grid.exit_pos and grid.is_near_fire(r, c):
                cost_map[r, c] += 60.0

    # Determinar longitud maxima razonable del cromosoma si no se especifico
    if max_steps is None:
        max_steps = int((rows + cols) * 1.5)

    # 1. Crear poblacion inicial con cromosomas aleatorios
    population: List[Individual] = [
        Individual([random.choice(DIRECTIONS) for _ in range(max_steps)])
        for _ in range(population_size)
    ]

    best_overall: Optional[Individual] = None

    # 2. Ciclo de generaciones
    for _ in range(generations):
        # Evaluar aptitud (fitness) solo de individuos no evaluados (ahorra reevaluar élite)
        for ind in population:
            if ind.fitness == 0.0:
                evaluate_individual(ind, grid, start, goal, walkable=walkable, cost_map=cost_map)

        # Ordenar poblacion por fitness descendente
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        current_best = population[0]
        if best_overall is None or current_best.fitness > best_overall.fitness:
            best_overall = current_best

        # Parada temprana si ya se encontró una ruta exitosa a la meta
        if best_overall.reached_goal:
            return best_overall.path

        # 3. Elitismo: conservar los mejores individuos intactos con su fitness ya calculado
        next_population: List[Individual] = []
        for ind in population[:elitism_count]:
            elite = Individual(list(ind.chromosome))
            elite.fitness = ind.fitness
            elite.reached_goal = ind.reached_goal
            elite.path = ind.path
            next_population.append(elite)

        # 4. Reproduccion (Seleccion por torneo, Cruce y Mutacion)
        while len(next_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1 = Individual(list(parent1.chromosome))
                child2 = Individual(list(parent2.chromosome))

            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)

            next_population.append(child1)
            if len(next_population) < population_size:
                next_population.append(child2)

        population = next_population

    return best_overall.path if best_overall and best_overall.reached_goal else None


# Alias comunes
genetic_algorithm = genetic
