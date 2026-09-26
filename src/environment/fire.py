import math
import random
from typing import List, Set, Optional
from src.environment.grid import Grid, Coord


def get_random_fire_foci(
    grid: Grid,
    count: int = 1,
    min_dist_to_exit: float = 7.0,
    spawn_pos: Coord = (1, 1),
    min_dist_to_spawn: float = 3.0
) -> List[Coord]:
    """
    Selecciona casillas aleatorias para iniciar focos de fuego asegurando:
    - Que no sean muros ni la salida.
    - Distancia euclidiana >= min_dist_to_exit respecto a la salida (por defecto >= 7.0).
    - Distancia euclidiana >= min_dist_to_spawn respecto a los agentes (por defecto >= 3.0).
    """
    candidates: List[Coord] = []
    exit_pos = grid.exit_pos
    if exit_pos is None:
        raise ValueError("El mapa no tiene salida definida.")

    for r in range(grid.rows):
        for c in range(grid.cols):
            # No puede ser muro ni la salida
            if grid.walls[r, c] or (r, c) == exit_pos:
                continue

            dist_exit = math.hypot(r - exit_pos[0], c - exit_pos[1])
            dist_agents = math.hypot(r - spawn_pos[0], c - spawn_pos[1])

            if dist_exit >= min_dist_to_exit and dist_agents >= min_dist_to_spawn:
                candidates.append((r, c))

    if not candidates:
        raise ValueError(
            f"No hay celdas libres con distancia >= {min_dist_to_exit} a la salida "
            f"y >= {min_dist_to_spawn} a los agentes."
        )

    return random.sample(candidates, min(count, len(candidates)))


#Gestiona la propagacion del fuego en intervalos de k turnos.
class FireSpread:
    def __init__(self, grid: Grid, spread_interval: int = 3, spread_probability: float = 0.8):
        self.grid: Grid = grid
        self.k: int = spread_interval
        self.spread_probability: float = spread_probability
        self.active_flames: Set[Coord] = set()

    """Inicia un foco de incendio en una celda si no es muro ni la salida."""
    def ignite(self, r: int, c: int) -> bool:
        if self.grid.in_bounds(r, c) and not self.grid.walls[r, c] and (r, c) != self.grid.exit_pos:
            self.grid.add_fire(r, c)
            self.active_flames.add((r, c))
            return True
        return False

    """
    Avanza un turno en el modelo de fuego.
    Si current_turn es múltiplo de k, propaga hacia celdas adyacentes.
    Retorna la lista de nuevas celdas consumidas en este turno.
    """
    def step(self, current_turn: int) -> List[Coord]:
        if current_turn == 0 or current_turn % self.k != 0:
            return []
        newly_ignited: List[Coord] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Se evalua la expansion a partir del frente de fuego actual
        for r, c in list(self.active_flames):
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                # Solo puede prender si es valida, no es muro, no tiene fuego y no es la salida
                if (
                    self.grid.in_bounds(nr, nc)
                    and not self.grid.walls[nr, nc]
                    and not self.grid.fire[nr, nc]
                    and (nr, nc) != self.grid.exit_pos
                ):
                    # Factor estocastico de propagacion
                    if random.random() < self.spread_probability:
                        self.grid.add_fire(nr, nc)
                        newly_ignited.append((nr, nc))
                        
        self.active_flames.update(newly_ignited)
        return newly_ignited

    """Limpia el estado del fuego para reiniciar entre iteraciones del benchmark."""
    def reset(self) -> None:
        self.active_flames.clear()
        self.grid.reset_state()