import os
import sys
from src.environment.grid import Grid
from src.environment.fire import FireSpread


def test_environment() -> None:
    print("=== Iniciando pruebas del entorno ===")

    # 1. Validar existencia del mapa
    map_path = os.path.join("resources","maps", "escenario_1.txt")
    if not os.path.exists(map_path):
        print(f"[ERROR] No se encontró el archivo en: {map_path}")
        print("Verifica que la carpeta 'resources' y el archivo existan en la raíz.")
        sys.exit(1)

    # 2. Cargar grilla y validar dimensiones
    grid = Grid(map_path)
    print(f"[OK] Mapa cargado con éxito: {grid.rows} filas x {grid.cols} columnas.")
    print(f"[OK] Salida encontrada en coordenadas: {grid.exit_pos}")

    assert grid.exit_pos is not None, "La salida no puede ser None"
    assert grid.rows == 25 and grid.cols == 40, f"Dimensiones inesperadas: {grid.rows}x{grid.cols}"

    # 3. Probar colisiones y transitabilidad básica
    er, ec = grid.exit_pos
    assert grid.is_walkable(er, ec), "La salida debe ser una celda transitable"
    assert not grid.is_walkable(0, 0), "La celda (0, 0) debería ser un muro perimetral"

    # Vecinos de una celda interior libre (ej: fila 1, col 1)
    neighbors = grid.get_neighbors(1, 1)
    print(f"[OK] Vecinos transitables desde (1, 1): {neighbors}")
    assert len(neighbors) > 0, "La celda (1, 1) debería tener al menos un vecino libre"

    # 4. Probar propagación del fuego
    print("\n=== Probando sistema de fuego ===")
    k_interval = 2
    fire_system = FireSpread(grid, spread_interval=k_interval, spread_probability=1.0)

    # Foco inicial en una celda libre del tercio superior
    start_fire = (2, 2)
    ignited = fire_system.ignite(start_fire[0], start_fire[1])
    assert ignited, f"No se pudo encender el foco en {start_fire}"
    assert grid.fire[start_fire], "La matriz de fuego de Grid no registró el foco inicial"
    assert not grid.is_walkable(start_fire[0], start_fire[1]), "La celda en llamas debe ser intransitable"
    print(f"[OK] Foco inicial encendido en {start_fire}. Transitabilidad: {grid.is_walkable(*start_fire)}")

    # Simular 4 turnos
    total_burned_before = len(fire_system.active_flames)
    for turno in range(1, 5):
        nuevas = fire_system.step(current_turn=turno)
        if turno % k_interval == 0:
            print(f"-> Turno {turno} (múltiplo de k={k_interval}): Se propagó a {len(nuevas)} celdas nuevas.")
            assert len(nuevas) > 0, f"En el turno {turno} el fuego debió avanzar"
        else:
            print(f"-> Turno {turno}: El fuego no se propaga (esperando intervalo k).")
            assert len(nuevas) == 0, f"En el turno {turno} el fuego no debió avanzar"

    assert len(fire_system.active_flames) > total_burned_before, "El fuego no aumentó su área activa"

    # 5. Probar reseteo del entorno
    print("\n=== Probando reseteo para benchmarking ===")
    fire_system.reset()
    assert len(fire_system.active_flames) == 0, "El conjunto de llamas no se vació"
    assert not grid.fire.any(), "La matriz booleana de fuego aún contiene celdas activas"
    assert grid.is_walkable(start_fire[0], start_fire[1]), "La celda del foco inicial debió volver a ser transitable"
    print("[OK] Entorno reseteado exitosamente a su estado original.")

    print("\n>>> TODAS LAS PRUEBAS BASE PASARON CORRECTAMENTE <<<")


if __name__ == "__main__":
    test_environment()