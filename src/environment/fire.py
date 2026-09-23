import random
from typing import List, Set
from src.environment.grid import Grid, Coord


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