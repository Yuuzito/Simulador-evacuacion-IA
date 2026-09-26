import time
import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional
from src.environment.grid import Grid, Coord
from src.environment.fire import FireSpread, get_random_fire_foci
from src.environment.agent import Agent, AgentState


class Simulation:
    """
    Gestiona la simulación multi-agente de evacuación turno a turno:
    - Control de colas y saturación física de pasillos (capacidad finita).
    - Actualización dinámica de la congestión y penalización de costos.
    - Propagación estocástica del fuego cada k turnos.
    - Replanificación reactiva de rutas ante corte de vías o saturación.
    - Cálculo de métricas requeridas: tasa de supervivencia y tiempos de despeje.
    """

    def __init__(
        self,
        map_path: str,
        num_agents: int = 200,
        spawn_pos: Coord = (1, 1),
        fire_foci: Optional[List[Coord]] = None,
        spread_interval: int = 3,
        spread_probability: float = 0.8,
        cell_capacity: int = 2,
        algorithm_name: str = "A*",
        algorithm_func: Optional[Callable[[Grid, Coord, Coord], Optional[List[Coord]]]] = None
    ):
        self.map_path: str = map_path
        self.num_agents: int = num_agents
        self.spawn_pos: Coord = spawn_pos
        self.spread_interval: int = spread_interval
        self.spread_probability: float = spread_probability
        self.cell_capacity: int = cell_capacity
        self.algorithm_name: str = algorithm_name
        self.algorithm_func = algorithm_func

        self.grid: Grid = Grid(map_path)
        self.fire: FireSpread = FireSpread(
            self.grid,
            spread_interval=spread_interval,
            spread_probability=spread_probability
        )

        # Si no se proporcionan focos, generar 1 o varios focos aleatorios que cumplan las restricciones
        if fire_foci is None:
            fire_foci = get_random_fire_foci(
                self.grid,
                count=1,
                min_dist_to_exit=7.0,
                spawn_pos=self.spawn_pos,
                min_dist_to_spawn=3.0
            )

        self.initial_fire_foci: List[Coord] = list(fire_foci)
        for r, c in fire_foci:
            self.fire.ignite(r, c)

        # Crear los agentes todos naciendo en spawn_pos
        self.agents: List[Agent] = [
            Agent(id=i + 1, start_pos=self.spawn_pos) for i in range(num_agents)
        ]

    def _update_congestion(self) -> None:
        """Actualiza la matriz de congestión con la cantidad de agentes en cada celda."""
        self.grid.congestion.fill(0)
        for agent in self.agents:
            if agent.state == AgentState.ALIVE:
                r, c = agent.pos
                self.grid.congestion[r, c] += 1

    def run(self, max_turns: int = 500, record_history: bool = False) -> Dict[str, Any]:
        """
        Ejecuta la simulación hasta que todos los agentes evacuen, mueran o se alcance max_turns.
        Si record_history=True, almacena el estado visual de cada turno para reproducirlo en la GUI.
        """
        turn = 0
        start_time = time.perf_counter()
        history: List[Dict[str, Any]] = []
        cached_unreachable: Set[Coord] = set()

        if record_history:
            history.append({
                "turn": 0,
                "fire": np.copy(self.grid.fire),
                "agents": [(a.pos, a.state) for a in self.agents]
            })

        while turn < max_turns:
            # Condición de parada: ningún agente queda vivo en el edificio
            alive_agents = [a for a in self.agents if a.state == AgentState.ALIVE]
            if not alive_agents:
                break

            turn += 1

            # 1. Actualizar la matriz de congestión con las posiciones actuales
            self._update_congestion()

            # 2. Turno de toma de decisiones y avance de cada agente vivo
            turn_route_cache: Dict[Coord, Optional[List[Coord]]] = {}
            for agent in alive_agents:
                # Si el agente no tiene ruta planificada (o fue invalidada por fuego), replanifica
                if not agent.path:
                    if self.algorithm_func:
                        if agent.pos in cached_unreachable:
                            continue

                        # Si ya se calculó una ruta desde esta posición en este turno, se comparte (líder de grupo)
                        if agent.pos not in turn_route_cache:
                            turn_route_cache[agent.pos] = self.algorithm_func(self.grid, agent.pos, self.grid.exit_pos)
                        
                        new_path = turn_route_cache[agent.pos]
                        if new_path:
                            agent.set_path(new_path)
                            agent.replan_count += 1
                        else:
                            # Sin ruta posible hacia la salida mientras el fuego no cambie
                            cached_unreachable.add(agent.pos)
                            continue

                # El agente intenta avanzar; si la casilla está llena, espera
                agent.step(self.grid, turn, cell_capacity=self.cell_capacity)

            # 3. Propagación del fuego cada k turnos
            new_flames = self.fire.step(turn)
            if new_flames:
                cached_unreachable.clear()

            # 4. Si el fuego consumió la casilla de algún agente, pasa a estado DEAD
            for agent in alive_agents:
                if agent.state == AgentState.ALIVE and self.grid.fire[agent.pos]:
                    agent.state = AgentState.DEAD

            # 5. Invalidar rutas de agentes si las nuevas llamas cortaron casillas de su camino
            if new_flames:
                new_flames_set = set(new_flames)
                for agent in self.agents:
                    if agent.state == AgentState.ALIVE and agent.path:
                        if any(coord in new_flames_set for coord in agent.path):
                            agent.clear_path()

            # 6. Grabar snapshot para animación si está habilitado
            if record_history:
                history.append({
                    "turn": turn,
                    "fire": np.copy(self.grid.fire),
                    "agents": [(a.pos, a.state) for a in self.agents]
                })

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Compilación de métricas de la simulación
        escaped_agents = [a for a in self.agents if a.state == AgentState.ESCAPED]
        dead_agents = [a for a in self.agents if a.state == AgentState.DEAD]
        trapped_agents = [a for a in self.agents if a.state == AgentState.ALIVE]

        survivors_count = len(escaped_agents)
        dead_count = len(dead_agents)
        trapped_count = len(trapped_agents)
        survival_rate = (survivors_count / self.num_agents) * 100.0

        # Tiempos de despeje
        escaped_turns = [a.escaped_at_turn for a in escaped_agents if a.escaped_at_turn is not None]
        clearance_time = max(escaped_turns) if escaped_turns else None
        avg_escape_time = (sum(escaped_turns) / len(escaped_turns)) if escaped_turns else None

        avg_waited = sum(a.waited_turns for a in self.agents) / self.num_agents
        total_replans = sum(a.replan_count for a in self.agents)

        return {
            "algorithm": self.algorithm_name,
            "map_path": self.map_path,
            "total_agents": self.num_agents,
            "survivors": survivors_count,
            "dead": dead_count,
            "trapped": trapped_count,
            "survival_rate": survival_rate,
            "clearance_time": clearance_time,  # Turnos del último sobreviviente
            "avg_escape_time": avg_escape_time,
            "avg_waited_turns": avg_waited,
            "total_turns": turn,
            "total_replans": total_replans,
            "elapsed_ms": elapsed_ms,
            "burned_cells": int(self.grid.fire.sum()),
            "history": history,
            "grid": self.grid
        }
