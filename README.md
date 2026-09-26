# Escape de la Torre — Simulador de Evacuación con IA

Simulador de evacuación multi-agente en una cuadrícula 2D bajo condiciones de propagación dinámica de fuego, congestión en tiempo real y capacidad finita por casilla. Este proyecto compara cuantitativa y cualitativamente algoritmos de búsqueda no informada, búsqueda informada y algoritmos bioinspirados.

## 📋 Descripción del Proyecto

El sistema modela una torre en llamas representada como una cuadrícula discreta de dimensiones variables con obstáculos (paredes), salidas de emergencia, focos de fuego estocásticos y un conjunto de agentes evacuando simultáneamente hacia las salidas seguras.

### Características y Restricciones del Entorno
* **Capacidad Finita de Casilla:** Máximo **2 agentes** por casilla en todo momento. Si una casilla está llena, los agentes esperan su turno en la cola o buscan desvíos sin perder coordinación.
* **Costo Cuadrático de Congestión:** Cada paso por una casilla ocupada por $n$ agentes incurre en un costo:
  $$C(n) = 1.0 + 0.5 \cdot n^2$$
* **Propagación Dinámica de Fuego:** El fuego se expande cada $k = 3$ turnos hacia celdas ortogonales adyacentes de acuerdo a probabilidades físicas ($\alpha = 0.65$ base, $\beta = 0.25$ factor de proximidad).
* **Penalización por Proximidad Térmica:** Las casillas ortogonales a un foco activo incrementan su costo de tránsito ($+3.0$).
* **Focos Iniciales Estocásticos:** Generados con distancias de seguridad mínimas ($\ge 7.0$ de la salida y $\ge 3.0$ del punto de inicio/spawn).

---

## 🧠 Algoritmos Implementados

| Categoría | Algoritmo | Heurística / Función de Selección | Características de Diseño |
| :--- | :--- | :--- | :--- |
| **No Informada** | **DFS** (Depth-First Search) | N/A (LIFO) | Búsqueda exhaustiva en profundidad con prevención de ciclos (`visited`). No garantiza optimalidad. |
| **No Informada** | **BFS** (Breadth-First Search) | N/A (FIFO) | Búsqueda por niveles que garantiza la ruta de menor número de pasos en grafos no ponderados. |
| **Informada** | **Greedy Best-First Search** | $h(n)$ (Distancia Manhattan / Euclidiana) | Prioriza avance voraz hacia la salida más cercana; sensible a trampas locales y frentes de fuego. |
| **Informada** | **A\*** (A-Star Search) | $f(n) = g(n) + h(n)$ | Considera el costo acumulado $g(n)$ (incluyendo congestión) y heurística admisible $h(n)$, garantizando optimalidad. |
| **Bioinspirado** | **Algoritmo Genético** | Fitness: $F = 1000/(L + C) + 10000 \cdot I_{meta} - \text{penalizaciones}$ | Cromosomas de secuencias de direcciones ortogonales, selección por torneo, cruce en un punto, mutación estocástica, elitismo y matrices precomputadas de coste. |

---

## 📁 Estructura del Repositorio

```text
Simulador-evacuacion-IA/
├── resources/
│   ├── fire.png                      # Sprite gráfico del fuego
│   └── maps/                         # Escenarios en formato texto
│       ├── escenario_1.txt           # Mapa 1: Cuadrícula con obstáculos estándar (30x30)
│       ├── escenario_2.txt           # Mapa 2: Entorno tipo laberinto estrecho (31x31)
│       └── escenario_3.txt           # Mapa 3: Espacios amplios con columnas dispersas (31x31)
├── results/
│   ├── benchmark_results.csv         # Resultados consolidados de 80 ejecuciones por combinación
│   └── ...                           # Gráficos y análisis estadísticos
├── src/
│   ├── algorithms/
│   │   ├── genetic.py                # Algoritmo genético optimizado
│   │   ├── informed.py               # Greedy Best-First Search y A*
│   │   └── uninformed.py             # DFS y BFS
│   └── environment/
│       ├── agent.py                  # Lógica del agente y control de capacidad por celda
│       ├── fire.py                   # Modelo de propagación estocástica del fuego
│       ├── grid.py                   # Parser del mapa, matriz de fuego, costos y congestión
│       └── simulation.py             # Orquestador multi-agente, caché de rutas y métricas
├── benchmark.py                      # Script de evaluación empírica (1.200 simulaciones totales)
├── gui.py                            # Interfaz gráfica interactiva en Tkinter (visor, step-by-step y comparador)
├── requirements.txt                  # Dependencias del proyecto
├── .gitignore                        # Exclusiones de Git (entornos virtuales, caches, temporales)
└── README.md                         # Documentación general del proyecto
```

---

## 🚀 Requisitos e Instalación

### Requisitos Previos
* **Python 3.10** o superior instalado en el sistema.
* Acceso a terminal (PowerShell, CMD, Bash o Zsh).

### 1. Clonar el Repositorio
```bash
git clone https://github.com/Yuuzito/Simulador-evacuacion-IA.git
cd Simulador-evacuacion-IA
```

### 2. Configuración del Entorno Virtual (`.venv`)

Se recomienda aislar las dependencias utilizando un entorno virtual en la carpeta `.venv`:

#### En Windows (PowerShell):
```powershell
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno virtual
.\.venv\Scripts\Activate.ps1
```
*(Nota: Si PowerShell restringe la ejecución de scripts, ejecutar previamente `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`).*

#### En Windows (CMD):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### En Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalación de Dependencias

Con el entorno virtual activado, instale los paquetes requeridos vía [`requirements.txt`](file:///c:/Users/alfon/OneDrive/Escritorio/VISUAL/Python/Simulador-evacuacion-IA/requirements.txt):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Las dependencias principales son:
* `numpy`: Operaciones matriciales vectorizadas de costo y transitabilidad.
* `pandas`: Procesamiento y consolidación de métricas tabulares.
* `matplotlib`: Visualización de gráficos estadísticos y mapas de calor.
* `pygame`: Soporte para componentes visuales y sprites auxiliares.

---

## 🎮 Ejecución del Sistema

### Modo Interactivo: Interfaz Gráfica (GUI)

La interfaz permite visualizar la propagación del fuego, la toma de decisiones de los agentes y comparar algoritmos paso a paso con controles de reproducción:

```bash
python gui.py
```

#### Funcionalidades de la GUI:
* **Selector de Mapa y Algoritmo:** Carga dinámica de escenarios y selección del método de búsqueda.
* **Configuración Interactiva:**
  * Clic izquierdo en el mapa para establecer el punto de inicio de los agentes (Spawn).
  * Clic derecho para ubicar focos manuales de fuego (o usar el generador estocástico).
* **Controles de Simulación:** Iniciar, Pausar, Reanudar, Reiniciar y barra deslizante (slider) de control de pasos en el tiempo.
* **Modo Comparador:** Ejecución paralela lado a lado de dos algoritmos bajo las mismas condiciones exactas de fuego inicial y semilla aleatoria.

---

### Modo Evaluación: Benchmark Estadístico

Para reproducir los experimentos del informe bajo rigor estadístico (80 iteraciones por combinación = 1.200 simulaciones):

```bash
python benchmark.py
```

Los resultados se almacenan automáticamente en [`results/benchmark_results.csv`](file:///c:/Users/alfon/OneDrive/Escritorio/VISUAL/Python/Simulador-evacuacion-IA/results/benchmark_results.csv), reportando:
* Tasa de supervivencia promedio ($\mu$) y desviación estándar ($\sigma$).
* Tiempo de evacuación promedio ($\mu$), desviación estándar ($\sigma$), mínimo y máximo.
* Longitud promedio de las rutas y tiempo promedio de cómputo por replanificación.

---

## 📊 Síntesis de Resultados Empíricos

* **A\* y Greedy** exhiben las tasas de supervivencia más altas ($\approx 55\% - 70\%$) y los menores tiempos de despacho en todos los mapas, equilibrando evasión de fuego y congestión.
* **BFS** mantiene rutas de mínima cantidad de giros/pasos, pero al no considerar el costo acumulado de proximidad al fuego puede acercar a los agentes a frentes activos.
* **DFS** sufre severamente en mapas con amplios grados de libertad (Mapa 3) debido a bucles de exploración profunda que agotan el tiempo disponible antes de que el fuego alcance al enjambre.
* **Algoritmo Genético** demuestra adaptabilidad estocástica, pero su convergencia en cuellos de botella estrechos (Mapa 2) se ve limitada por la necesidad de una población y número de generaciones suficientemente altos para resolver laberintos complejos en tiempo real.
