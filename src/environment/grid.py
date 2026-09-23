import numpy as np
from typing import List, Tuple, Optional

# Definición de tipos
Coord = Tuple[int, int]

"""
Gestiona el mapa 2D del edificio, obstáculos fijos, fuego dinámico 
y el cálculo del costo de paso penalizado por congestión.
"""
class Grid:
    WALL_CHAR = '#'
    EMPTY_CHAR = '.'
    EXIT_CHAR = 'E'

    def __init__(self, map_file_path: str):
        self.map_file_path: str = map_file_path
        self.rows: int = 0
        self.cols: int = 0
        self.exit_pos: Optional[Coord] = None

        # Matrices de muro, fuego y cantidad de agentes en el mapa.
        self.walls: np.ndarray = np.array([], dtype=bool)
        self.fire: np.ndarray = np.array([], dtype=bool)
        self.congestion: np.ndarray = np.array([], dtype=int)

        self._load_map()

    """Lee el archivo .txt e inicializa las matrices de muros y la salida."""
    def _load_map(self) -> None:
        with open(self.map_file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\r\n') for line in f if line.strip()]

        self.rows = len(lines)
        self.cols = len(lines[0]) if self.rows > 0 else 0

        # Inicializamos las matrices ahora que tenemos el tamaño del mapa
        self.walls = np.zeros((self.rows, self.cols), dtype=bool)
        self.fire = np.zeros((self.rows, self.cols), dtype=bool)
        self.congestion = np.zeros((self.rows, self.cols), dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                char = lines[r][c]
                if char == self.WALL_CHAR:
                    self.walls[r, c] = True
                elif char == self.EXIT_CHAR:
                    self.exit_pos = (r, c)

        if self.exit_pos is None:
            raise ValueError(f"No se encontró la salida 'E' en: {self.map_file_path}")
        
    """Verifica si la celda cae dentro de los límites del mapa."""
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    """
    Una celda es transitable si está dentro del mapa,
    no es un muro y no ha sido consumida por el fuego.
    """
    def is_walkable(self, r: int, c: int) -> bool:
        if not self.in_bounds(r, c):
            return False
        return not self.walls[r, c] and not self.fire[r, c]

    """
    Retorna las celdas adyacentes ortogonales (arriba, abajo, izquierda, derecha)
    que estén libres y sin fuego.
    """
    def get_neighbors(self, r: int, c: int) -> List[Coord]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid_neighbors = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_walkable(nr, nc):
                valid_neighbors.append((nr, nc))
        return valid_neighbors

    """
    Calcula el costo de atravesar una celda segun la congestion.
    Costo = 1.0 (base) + penalizacion cuadratica por agentes presentes.
    """
    def get_cost(self, r: int, c: int) -> float:
        num_agents = self.congestion[r, c]
        return 1.0 + 0.5 * (num_agents ** 2)

    """
    Propaga el fuego a una celda si no es un muro.
    """
    def add_fire(self, r: int, c: int) -> None:
        if self.in_bounds(r, c) and not self.walls[r, c]:
            self.fire[r, c] = True

    """
    Limpia fuego y congestion para iniciar una nueva iteracion del benchmark.
    """
    def reset_state(self) -> None:
        self.fire.fill(False)
        self.congestion.fill(0)