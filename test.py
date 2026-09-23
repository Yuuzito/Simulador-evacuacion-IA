import sys
from src.environment.grid import Grid
from src.environment.fire import FireSpread
from src.algorithms.uninformed import dfs, bfs


def print_path_on_grid(grid: Grid, path: list, start: tuple, title: str = "") -> None:
    """Imprime el mapa en consola mostrando la trayectoria del agente."""
    if title:
        print(f"\n--- {title} ---")
    
    path_set = set(path) if path else set()
    print("=" * grid.cols)
    
    for r in range(grid.rows):
        line = []
        for c in range(grid.cols):
            coord = (r, c)
            if coord == start:
                line.append("S")          # Inicio
            elif coord == grid.exit_pos:
                line.append("E")          # Salida (Exit)
            elif coord in path_set:
                line.append("*")          # Ruta tomada
            elif grid.walls[r, c]:
                line.append("#")          # Muro
            elif grid.fire[r, c]:
                line.append("F")          # Fuego
            else:
                line.append(".")          # Espacio libre
        print("".join(line))
        
    print("=" * grid.cols)


def main():
    map_path = "resources/maps/escenario_1.txt"
    print(f"Cargando escenario: {map_path}")
    
    grid = Grid(map_path)
    start = (1, 1)
    goal = grid.exit_pos

    print(f"Dimensiones: {grid.rows}x{grid.cols}")
    print(f"Posición inicial: {start}")
    print(f"Salida detectada: {goal}\n")

    # 1. Ejecutar DFS
    ruta_dfs = dfs(grid, start, goal)
    pasos_dfs = len(ruta_dfs) if ruta_dfs else 0
    print(f"Resultado DFS: {pasos_dfs} pasos")

    # 2. Ejecutar BFS
    ruta_bfs = bfs(grid, start, goal)
    pasos_bfs = len(ruta_bfs) if ruta_bfs else 0
    print(f"Resultado BFS: {pasos_bfs} pasos")

    # 3. Mostrar visualizaciones
    if ruta_dfs:
        print_path_on_grid(grid, ruta_dfs, start, title=f"RUTA DFS ({pasos_dfs} pasos)")

    if ruta_bfs:
        print_path_on_grid(grid, ruta_bfs, start, title=f"RUTA ÓPTIMA BFS ({pasos_bfs} pasos)")


if __name__ == "__main__":
    main()