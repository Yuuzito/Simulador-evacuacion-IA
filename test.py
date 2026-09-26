import math
from typing import List, Tuple
from src.environment.grid import Grid, Coord
from src.environment.fire import get_random_fire_foci
from src.environment.simulation import Simulation
from src.algorithms.uninformed import dfs, bfs
from src.algorithms.informed import greedy, a_star
from src.algorithms.genetic import genetic


def print_final_grid(grid: Grid, title: str = "") -> None:
    """Imprime el estado final del mapa con fuego ('F'), muros ('#') y salida ('E')."""
    if title:
        print(f"\n--- {title} ---")
    print("=" * grid.cols)
    for r in range(grid.rows):
        line = []
        for c in range(grid.cols):
            coord = (r, c)
            if coord == grid.exit_pos:
                line.append("E")
            elif grid.fire[r, c]:
                line.append("F")
            elif grid.walls[r, c]:
                line.append("#")
            elif grid.congestion[r, c] > 0:
                line.append(str(min(grid.congestion[r, c], 9)))
            else:
                line.append(".")
        print("".join(line))
    print("=" * grid.cols)


def main():
    NUM_AGENTS = 100
    CELL_CAPACITY = 2
    SPAWN_POS = (1, 1)

    scenarios = [
        {
            "nombre": "Escenario 1 (Cuello de Botella)",
            "path": "resources/maps/escenario_1.txt",
            "k": 3
        },
        {
            "nombre": "Escenario 2 (Laberinto Corporativo)",
            "path": "resources/maps/escenario_2.txt",
            "k": 3
        },
        {
            "nombre": "Escenario 3 (Dispersión Abierta)",
            "path": "resources/maps/escenario_3.txt",
            "k": 3
        }
    ]

    algorithms = {
        "DFS": dfs,
        "BFS": bfs,
        "Greedy": greedy,
        "A*": a_star,
        "Genético": lambda g, s, e: genetic(g, s, e, population_size=80, generations=80, mutation_rate=0.06)
    }

    print("\n" + "=" * 80)
    print(f" SIMULADOR DE EVACUACIÓN MULTI-AGENTE (Tarea 1: Escape de la Torre)")
    print(f" Agentes: {NUM_AGENTS} | Capacidad por celda: {CELL_CAPACITY} | Spawn inicial: {SPAWN_POS}")
    print("=" * 80)

    for sc in scenarios:
        # Cargar mapa para determinar la salida y elegir el foco aleatorio
        temp_grid = Grid(sc["path"])
        foco_aleatorio = get_random_fire_foci(
            temp_grid,
            count=1,
            min_dist_to_exit=7.0,
            spawn_pos=SPAWN_POS,
            min_dist_to_spawn=3.0
        )
        foco = foco_aleatorio[0]

        d_exit = math.hypot(foco[0] - temp_grid.exit_pos[0], foco[1] - temp_grid.exit_pos[1])
        d_spawn = math.hypot(foco[0] - SPAWN_POS[0], foco[1] - SPAWN_POS[1])

        print("\n" + "#" * 80)
        print(f" EVALUANDO: {sc['nombre']}")
        print(f" Mapa: {sc['path']} | Propagación fuego: cada k = {sc['k']} turnos")
        print(f" Foco de fuego aleatorio: {foco} (Dist. salida: {d_exit:.2f} >= 7 | Dist. agentes: {d_spawn:.2f} >= 3)")
        print("#" * 80)

        results = []

        for name, algo_func in algorithms.items():
            print(f"  > Simulando con {name:<10}...", end="", flush=True)

            sim = Simulation(
                map_path=sc["path"],
                num_agents=NUM_AGENTS,
                spawn_pos=SPAWN_POS,
                fire_foci=[foco],
                spread_interval=sc["k"],
                spread_probability=0.75,
                cell_capacity=CELL_CAPACITY,
                algorithm_name=name,
                algorithm_func=algo_func
            )

            res = sim.run(max_turns=600)
            results.append(res)
            print(f" Completado en {res['elapsed_ms']:.1f} ms (Sobrevivientes: {res['survivors']}/{NUM_AGENTS})")

        # Tabla comparativa de métricas obligatorias
        print("\n" + "=" * 90)
        print(f"{'Algoritmo':<10} | {'Supervivencia':<14} | {'Bajas (F)':<10} | {'Despeje (turnos)':<18} | {'Espera prom.':<13} | {'Replanif.':<10}")
        print("=" * 90)

        for r in results:
            tasa_str = f"{r['survival_rate']:.1f}% ({r['survivors']}/{r['total_agents']})"
            despeje_str = str(r['clearance_time']) if r['clearance_time'] is not None else "Sin evacuados"
            espera_str = f"{r['avg_waited_turns']:.1f} t"
            print(f"{r['algorithm']:<10} | {tasa_str:<14} | {r['dead']:<10} | {despeje_str:<18} | {espera_str:<13} | {r['total_replans']:<10}")
        print("=" * 90)


if __name__ == "__main__":
    main()
