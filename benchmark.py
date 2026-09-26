import os
import csv
import math
import numpy as np
from typing import List, Dict, Any
from src.environment.grid import Grid, Coord
from src.environment.fire import get_random_fire_foci
from src.environment.simulation import Simulation
from src.algorithms.uninformed import dfs, bfs
from src.algorithms.informed import greedy, a_star
from src.algorithms.genetic import genetic


def run_benchmark(
    iterations: int = 80,
    num_agents: int = 100,
    cell_capacity: int = 2,
    spawn_pos: Coord = (1, 1),
    spread_interval_k: int = 3,
    output_csv: str = "results/benchmark_results.csv"
):
    scenarios = [
        {"name": "Mapa 1 (Cuello de Botella)", "path": "resources/maps/escenario_1.txt"},
        {"name": "Mapa 2 (Laberinto Corporativo)", "path": "resources/maps/escenario_2.txt"},
        {"name": "Mapa 3 (Dispersión Abierta)", "path": "resources/maps/escenario_3.txt"},
    ]

    algorithms = {
        "DFS": dfs,
        "BFS": bfs,
        "Greedy": greedy,
        "A*": a_star,
        "Genético": lambda g, s, e: genetic(g, s, e, population_size=40, generations=40, mutation_rate=0.06)
    }

    print("\n" + "=" * 105)
    print(" EJECUTANDO BENCHMARKING ESTADÍSTICO OBLIGATORIO (Tarea 1: Escape de la Torre)")
    print(f" Iteraciones por configuración: {iterations} | Agentes: {num_agents} | Capacidad celda: {cell_capacity}")
    print("=" * 105)

    summary_rows = []

    for sc in scenarios:
        map_name = sc["name"]
        map_path = sc["path"]
        temp_grid = Grid(map_path)

        print(f"\n>>> Evaluando: {map_name} ({map_path})")

        for algo_name, algo_func in algorithms.items():
            survival_rates = []
            clearance_times = []  # Solo de iteraciones donde al menos 1 agente evacuó

            for it in range(iterations):
                # Contador visual de progreso dinámico en tiempo real
                percent = ((it + 1) / iterations) * 100.0
                print(f"\r  * Algoritmo: {algo_name:<10} [Iteración {it+1:3d}/{iterations} ({percent:3.0f}%)]...", end="", flush=True)

                # Generar foco aleatorio estocástico que cumpla: distancia >= 7 a salida y >= 3 a agentes
                foco = get_random_fire_foci(
                    temp_grid,
                    count=1,
                    min_dist_to_exit=7.0,
                    spawn_pos=spawn_pos,
                    min_dist_to_spawn=3.0
                )[0]

                sim = Simulation(
                    map_path=map_path,
                    num_agents=num_agents,
                    spawn_pos=spawn_pos,
                    fire_foci=[foco],
                    spread_interval=spread_interval_k,
                    spread_probability=0.75,
                    cell_capacity=cell_capacity,
                    algorithm_name=algo_name,
                    algorithm_func=algo_func
                )

                res = sim.run(max_turns=500)
                survival_rates.append(res["survival_rate"])

                if res["clearance_time"] is not None:
                    clearance_times.append(res["clearance_time"])

            # Cálculo de estadísticos descriptivos con NumPy
            surv_mean = float(np.mean(survival_rates))
            surv_std = float(np.std(survival_rates))

            if clearance_times:
                time_mean = float(np.mean(clearance_times))
                time_std = float(np.std(clearance_times))
                time_min = int(np.min(clearance_times))
                time_max = int(np.max(clearance_times))
            else:
                time_mean = 0.0
                time_std = 0.0
                time_min = 0
                time_max = 0

            # Limpiar línea y mostrar resumen de las iteraciones
            print(f"\r  * Algoritmo: {algo_name:<10} [{iterations}/{iterations}] Listo! Supervivencia: {surv_mean:.1f}% ± {surv_std:.1f}% | Despeje: {time_mean:.1f} ± {time_std:.1f} turnos" + " " * 8)

            row_data = {
                "map": map_name,
                "algorithm": algo_name,
                "iterations": iterations,
                "surv_mean": surv_mean,
                "surv_std": surv_std,
                "time_mean": time_mean,
                "time_std": time_std,
                "time_min": time_min,
                "time_max": time_max,
                "evac_success_runs": len(clearance_times)
            }
            summary_rows.append(row_data)
            print(f"Listo! Supervivencia media: {surv_mean:.1f}% | Despeje medio: {time_mean:.1f} turnos")

    # Imprimir Tabla Final Formateada en Consola
    print("\n" + "=" * 115)
    print(f"{'Mapa':<30} | {'Algoritmo':<10} | {'Superv. (Media ± Std)':<23} | {'Despeje (Media ± Std)':<23} | {'Mín':<6} | {'Máx':<6}")
    print("=" * 115)

    for r in summary_rows:
        surv_str = f"{r['surv_mean']:.1f}% ± {r['surv_std']:.1f}%"
        if r["evac_success_runs"] > 0:
            time_str = f"{r['time_mean']:.1f} ± {r['time_std']:.1f} t"
            min_str = str(r["time_min"])
            max_str = str(r["time_max"])
        else:
            time_str = "Sin evacuados"
            min_str = "-"
            max_str = "-"

        print(f"{r['map']:<30} | {r['algorithm']:<10} | {surv_str:<23} | {time_str:<23} | {min_str:<6} | {max_str:<6}")
    print("=" * 115)

    # Exportar a CSV para Excel / Gráficos del informe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "map", "algorithm", "iterations",
            "surv_mean", "surv_std",
            "time_mean", "time_std", "time_min", "time_max",
            "evac_success_runs"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f"\n[+] Resultados exportados con éxito a: {output_csv}")


if __name__ == "__main__":
    import sys
    # Permite especificar el número de iteraciones por argumento: python benchmark.py 80
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    run_benchmark(iterations=iters)
