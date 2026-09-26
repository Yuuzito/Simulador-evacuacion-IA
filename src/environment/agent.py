from enum import Enum
from typing import List, Optional
from src.environment.grid import Grid, Coord


class AgentState(Enum):
    ALIVE = "ALIVE"
    ESCAPED = "ESCAPED"
    DEAD = "DEAD"


class Agent:
    def __init__(self, id: int, start_pos: Coord):
        self.id: int = id
        self.pos: Coord = start_pos
        self.state: AgentState = AgentState.ALIVE
        self.path: List[Coord] = []
        self.steps_taken: int = 0
        self.waited_turns: int = 0
        self.consecutive_waits: int = 0
        self.replan_count: int = 0
        self.escaped_at_turn: Optional[int] = None

    def set_path(self, path: Optional[List[Coord]]) -> None:
        """Asigna una nueva ruta planificada al agente."""
        self.path = list(path) if path else []
        self.consecutive_waits = 0

    def clear_path(self) -> None:
        """Invalida la ruta actual (sirve para forzar replanificación)."""
        self.path.clear()
        self.consecutive_waits = 0

    def step(self, grid: Grid, current_turn: int, cell_capacity: int = 2) -> Coord:
        """
        Ejecuta la acción del agente para el turno actual:
        - Si el camino está despejado y la siguiente celda tiene capacidad, avanza.
        - Si la siguiente celda está saturada por personas, ejecuta la acción de ESPERAR
          para descongestionar el paso.
        - Si el fuego bloquea la celda, descarta la ruta para replanificar.
        """
        if self.state != AgentState.ALIVE:
            return self.pos

        # 1. Si la celda actual fue consumida por el fuego mientras estaba en ella
        if grid.fire[self.pos]:
            self.state = AgentState.DEAD
            return self.pos

        # 2. Si tiene una ruta planificada, intenta dar el siguiente paso
        if self.path:
            next_cell = self.path[0]

            # Verificar si el siguiente paso fue consumido por el fuego
            if not grid.is_walkable(next_cell[0], next_cell[1]):
                self.clear_path()
                return self.pos

            # Comprobar capacidad física finita de la celda (la salida 'E' no tiene límite de capacidad)
            is_exit = (next_cell == grid.exit_pos)
            has_capacity = is_exit or (grid.congestion[next_cell] < cell_capacity)

            if has_capacity:
                old_pos = self.pos
                # Se desplaza a la siguiente celda
                self.pos = self.path.pop(0)
                self.steps_taken += 1
                self.consecutive_waits = 0

                # Actualizar dinámicamente la congestión para que los siguientes agentes del turno respeten el límite de 2
                if old_pos != grid.exit_pos and grid.congestion[old_pos] > 0:
                    grid.congestion[old_pos] -= 1
                if self.pos != grid.exit_pos:
                    grid.congestion[self.pos] += 1
            else:
                # Acción de ESPERAR: permanece en la celda actual respetando el límite de 2 agentes por casilla
                self.waited_turns += 1
                self.consecutive_waits += 1

        # 3. Comprobar si alcanzó la salida
        if self.pos == grid.exit_pos:
            self.state = AgentState.ESCAPED
            self.escaped_at_turn = current_turn

        return self.pos

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, pos={self.pos}, state={self.state.value})"
