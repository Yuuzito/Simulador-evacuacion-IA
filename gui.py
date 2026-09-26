import math
import random
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from src.environment.grid import Grid, Coord
from src.environment.fire import get_random_fire_foci
from src.environment.simulation import Simulation
from src.algorithms.uninformed import dfs, bfs
from src.algorithms.informed import greedy, a_star
from src.algorithms.genetic import genetic


class EvacuationSimulatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simulador de Evacuación Multi-Agente - Escape de la Torre")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 720)

        # Paleta de colores Dark Theme moderno
        self.COLORS = {
            "bg": "#1e1e2e",
            "panel": "#25273a",
            "panel_border": "#363a4f",
            "text": "#cad3f5",
            "subtext": "#a5adcb",
            "accent": "#8aadf4",
            "success": "#a6da95",
            "danger": "#ed8796",
            "warning": "#eed49f",
            "wall": "#494d64",
            "empty": "#181825",
            "empty_border": "#282a36",
            "exit": "#a6da95",
            "fire": "#f5a97f",
            "fire_hot": "#ee99a0",
            "agent": "#8aadf4",
            "agent_cluster": "#c6a0f6",
            "spawn": "#7dc4e4"
        }

        self.root.configure(bg=self.COLORS["bg"])

        # Estado de configuración
        self.scenarios_info = {
            "Mapa 1: Cuello de Botella": "resources/maps/escenario_1.txt",
            "Mapa 2: Laberinto Corporativo": "resources/maps/escenario_2.txt",
            "Mapa 3: Dispersión Abierta": "resources/maps/escenario_3.txt",
        }
        self.current_grid: Optional[Grid] = None
        self.spawn_pos: Coord = (1, 1)
        self.fire_focus: Optional[Coord] = None
        self.click_mode: Optional[str] = None  # 'spawn' o 'fire'

        # Estado de reproducción de la mejor simulación
        self.best_history: List[Dict[str, Any]] = []
        self.current_replay_turn: int = 0
        self.is_playing: bool = False
        self.play_speed_ms: int = 150
        self.best_run_meta: Dict[str, Any] = {}

        self._configure_styles()
        self._build_layout()

        # Cargar mapa inicial
        self._on_map_changed()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=self.COLORS["bg"], foreground=self.COLORS["text"])
        style.configure("TLabel", background=self.COLORS["panel"], foreground=self.COLORS["text"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.COLORS["panel"], foreground=self.COLORS["accent"], font=("Segoe UI", 12, "bold"))
        style.configure("Header.TLabel", background=self.COLORS["bg"], foreground=self.COLORS["accent"], font=("Segoe UI", 15, "bold"))

        style.configure("TFrame", background=self.COLORS["panel"])
        style.configure("Card.TFrame", background=self.COLORS["panel"], relief="solid", borderwidth=1)

        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, background=self.COLORS["panel_border"], foreground=self.COLORS["text"])
        style.map("TButton", background=[("active", self.COLORS["accent"])], foreground=[("active", "#1e1e2e")])

        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=8, background=self.COLORS["success"], foreground="#1e1e2e")
        style.map("Accent.TButton", background=[("active", "#8bd5ca")])

        style.configure("Danger.TButton", font=("Segoe UI", 9, "bold"), padding=5, background=self.COLORS["danger"], foreground="#1e1e2e")
        style.map("Danger.TButton", background=[("active", "#f0c6c6")])

        style.configure("TCombobox", fieldbackground=self.COLORS["empty"], background=self.COLORS["panel_border"], foreground=self.COLORS["text"])

        # Estilo para Treeview
        style.configure("Treeview", background=self.COLORS["panel"], foreground=self.COLORS["text"], fieldbackground=self.COLORS["panel"], font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=self.COLORS["panel_border"], foreground=self.COLORS["text"], font=("Segoe UI", 9, "bold"))

    def _build_layout(self):
        # Contenedor principal dividido en dos paneles
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # --- PANEL IZQUIERDO: CONTROLES Y RESULTADOS ---
        left_container = tk.Frame(main_paned, bg=self.COLORS["bg"], width=420)
        main_paned.add(left_container, weight=0)

        # Scrollable frame para controles y resultados
        canvas_scroll = tk.Canvas(left_container, bg=self.COLORS["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self.control_panel = tk.Frame(canvas_scroll, bg=self.COLORS["bg"])

        self.control_panel.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=self.control_panel, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_control_card()
        self._build_results_card()

        # --- PANEL DERECHO: VISUALIZADOR Y ANIMACIÓN ---
        right_container = tk.Frame(main_paned, bg=self.COLORS["panel"], relief="solid", bd=1)
        main_paned.add(right_container, weight=1)

        self._build_viewer_panel(right_container)

    def _build_control_card(self):
        card = tk.LabelFrame(
            self.control_panel,
            text=" ⚙️ CONFIGURACIÓN DE ESCENARIO ",
            bg=self.COLORS["panel"],
            fg=self.COLORS["accent"],
            font=("Segoe UI", 11, "bold"),
            padx=12, pady=10
        )
        card.pack(fill=tk.X, padx=5, pady=5)

        # 1. Selector de Mapa
        tk.Label(card, text="Mapa / Entorno:", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(anchor="w", pady=(2, 2))
        self.map_var = tk.StringVar(value="Mapa 1: Cuello de Botella")
        self.map_combo = ttk.Combobox(card, textvariable=self.map_var, values=list(self.scenarios_info.keys()), state="readonly")
        self.map_combo.pack(fill=tk.X, pady=(0, 8))
        self.map_combo.bind("<<ComboboxSelected>>", lambda e: self._on_map_changed())

        # 2. Selector de Algoritmo
        tk.Label(card, text="Método / Algoritmo:", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(anchor="w", pady=(2, 2))
        self.algo_var = tk.StringVar(value="A*")
        self.algo_combo = ttk.Combobox(
            card,
            textvariable=self.algo_var,
            values=["A*", "BFS", "Greedy", "DFS", "Genético", "Comparar Todos"],
            state="readonly"
        )
        self.algo_combo.pack(fill=tk.X, pady=(0, 8))

        # 3. Cantidad de Agentes (80 - 200)
        agents_header = tk.Frame(card, bg=self.COLORS["panel"])
        agents_header.pack(fill=tk.X)
        tk.Label(agents_header, text="Cantidad de Agentes:", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(side=tk.LEFT)
        self.agents_lbl = tk.Label(agents_header, text="100", bg=self.COLORS["panel"], fg=self.COLORS["success"], font=("Segoe UI", 10, "bold"))
        self.agents_lbl.pack(side=tk.RIGHT)

        self.agents_scale = tk.Scale(
            card, from_=80, to=200, orient=tk.HORIZONTAL, bg=self.COLORS["panel"],
            fg=self.COLORS["text"], highlightthickness=0, troughcolor=self.COLORS["empty"],
            showvalue=False, command=lambda v: self.agents_lbl.config(text=str(v))
        )
        self.agents_scale.set(100)
        self.agents_scale.pack(fill=tk.X, pady=(0, 8))

        # 4. Posición de Inicio (Spawn)
        spawn_frame = tk.Frame(card, bg=self.COLORS["panel"])
        spawn_frame.pack(fill=tk.X, pady=4)
        tk.Label(spawn_frame, text="Inicio Agentes:", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(side=tk.LEFT)
        self.spawn_lbl = tk.Label(spawn_frame, text="(1, 1)", bg=self.COLORS["panel"], fg=self.COLORS["spawn"], font=("Segoe UI", 10, "bold"))
        self.spawn_lbl.pack(side=tk.LEFT, padx=5)

        self.btn_pick_spawn = tk.Button(
            spawn_frame, text="📌 Clic en Mapa", bg=self.COLORS["panel_border"],
            fg=self.COLORS["text"], relief="flat", font=("Segoe UI", 8),
            command=lambda: self._set_click_mode("spawn")
        )
        self.btn_pick_spawn.pack(side=tk.RIGHT)

        # 5. Posición de Inicio del Fuego
        fire_frame = tk.Frame(card, bg=self.COLORS["panel"])
        fire_frame.pack(fill=tk.X, pady=4)
        tk.Label(fire_frame, text="Foco de Fuego:", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(side=tk.LEFT)
        self.fire_lbl = tk.Label(fire_frame, text="Aleatorio", bg=self.COLORS["panel"], fg=self.COLORS["fire"], font=("Segoe UI", 10, "bold"))
        self.fire_lbl.pack(side=tk.LEFT, padx=5)

        fire_btns = tk.Frame(card, bg=self.COLORS["panel"])
        fire_btns.pack(fill=tk.X, pady=(0, 6))

        self.btn_random_fire = tk.Button(
            fire_btns, text="🎲 Aleatorio (>=7 salida, >=3 agentes)", bg=self.COLORS["panel_border"],
            fg=self.COLORS["text"], relief="flat", font=("Segoe UI", 8),
            command=self._generate_random_fire
        )
        self.btn_random_fire.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self.btn_pick_fire = tk.Button(
            fire_btns, text="🔥 Clic Mapa", bg=self.COLORS["panel_border"],
            fg=self.COLORS["text"], relief="flat", font=("Segoe UI", 8),
            command=lambda: self._set_click_mode("fire")
        )
        self.btn_pick_fire.pack(side=tk.RIGHT, padx=(2, 0))

        # 6. Salida Fija
        exit_frame = tk.Frame(card, bg=self.COLORS["panel"])
        exit_frame.pack(fill=tk.X, pady=(2, 8))
        tk.Label(exit_frame, text="Salida de Evacuación (Fija):", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(side=tk.LEFT)
        self.exit_lbl = tk.Label(exit_frame, text="--", bg=self.COLORS["panel"], fg=self.COLORS["exit"], font=("Segoe UI", 10, "bold"))
        self.exit_lbl.pack(side=tk.LEFT, padx=5)

        # 7. Iteraciones
        iters_frame = tk.Frame(card, bg=self.COLORS["panel"])
        iters_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(iters_frame, text="Cantidad de Iteraciones:", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(side=tk.LEFT)
        self.iters_var = tk.IntVar(value=1)
        self.iters_spin = ttk.Spinbox(iters_frame, from_=1, to=80, textvariable=self.iters_var, width=5)
        self.iters_spin.pack(side=tk.RIGHT)

        # 8. Botón Ejecutar
        self.btn_run = ttk.Button(card, text="🚀 INICIAR SIMULACIÓN", style="Accent.TButton", command=self._start_simulation)
        self.btn_run.pack(fill=tk.X, pady=(5, 5))

    def _build_results_card(self):
        self.results_card = tk.LabelFrame(
            self.control_panel,
            text=" 📊 MÉTRICAS Y RESULTADOS ",
            bg=self.COLORS["panel"],
            fg=self.COLORS["accent"],
            font=("Segoe UI", 11, "bold"),
            padx=10, pady=10
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.res_summary_lbl = tk.Label(
            self.results_card,
            text="Configura los parámetros y pulsa 'Iniciar Simulación'.",
            bg=self.COLORS["panel"],
            fg=self.COLORS["subtext"],
            justify=tk.LEFT,
            wraplength=380,
            font=("Segoe UI", 9)
        )
        self.res_summary_lbl.pack(anchor="w", pady=(0, 8))

        # Tabla comparativa si se ejecutan múltiples algoritmos / iteraciones
        columns = ("algo", "surv", "clear", "waited")
        self.tree = ttk.Treeview(self.results_card, columns=columns, show="headings", height=5)
        self.tree.heading("algo", text="Algoritmo")
        self.tree.heading("surv", text="Superv.")
        self.tree.heading("clear", text="Despeje")
        self.tree.heading("waited", text="Espera")

        self.tree.column("algo", width=80, anchor="w")
        self.tree.column("surv", width=95, anchor="center")
        self.tree.column("clear", width=80, anchor="center")
        self.tree.column("waited", width=75, anchor="center")

        self.tree.pack(fill=tk.BOTH, expand=True)

    def _build_viewer_panel(self, parent: tk.Frame):
        # Header del visualizador
        header_frame = tk.Frame(parent, bg=self.COLORS["panel"], padx=10, pady=8)
        header_frame.pack(fill=tk.X)

        self.viewer_title_lbl = tk.Label(
            header_frame,
            text="VISUALIZADOR DEL EDIFICIO",
            bg=self.COLORS["panel"],
            fg=self.COLORS["accent"],
            font=("Segoe UI", 12, "bold")
        )
        self.viewer_title_lbl.pack(side=tk.LEFT)

        self.click_mode_lbl = tk.Label(
            header_frame,
            text="",
            bg=self.COLORS["panel"],
            fg=self.COLORS["warning"],
            font=("Segoe UI", 9, "bold")
        )
        self.click_mode_lbl.pack(side=tk.RIGHT)

        # Canvas interactivo para la grilla
        canvas_frame = tk.Frame(parent, bg=self.COLORS["empty"], padx=5, pady=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=self.COLORS["empty"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_current_state())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Barra de Controles de Reproducción (Play, Pausa, Slider de turnos)
        control_bar = tk.Frame(parent, bg=self.COLORS["panel"], padx=10, pady=8)
        control_bar.pack(fill=tk.X)

        self.replay_status_lbl = tk.Label(
            control_bar,
            text="Turno: 0 / 0 | Evacuados: 0 | Vivos: 0",
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Segoe UI", 10, "bold")
        )
        self.replay_status_lbl.pack(anchor="w", pady=(0, 4))

        slider_frame = tk.Frame(control_bar, bg=self.COLORS["panel"])
        slider_frame.pack(fill=tk.X, pady=2)

        self.turn_slider = tk.Scale(
            slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, bg=self.COLORS["panel"],
            fg=self.COLORS["text"], highlightthickness=0, troughcolor=self.COLORS["empty"],
            showvalue=False, command=self._on_slider_moved
        )
        self.turn_slider.pack(fill=tk.X)

        btns_frame = tk.Frame(control_bar, bg=self.COLORS["panel"])
        btns_frame.pack(fill=tk.X, pady=(4, 0))

        self.btn_reset_replay = ttk.Button(btns_frame, text="⏮ Reiniciar", width=10, command=self._reset_replay)
        self.btn_reset_replay.pack(side=tk.LEFT, padx=2)

        self.btn_step_back = ttk.Button(btns_frame, text="◀ Paso", width=8, command=lambda: self._step_replay(-1))
        self.btn_step_back.pack(side=tk.LEFT, padx=2)

        self.btn_play_pause = ttk.Button(btns_frame, text="▶ Reproducir", width=12, command=self._toggle_play)
        self.btn_play_pause.pack(side=tk.LEFT, padx=2)

        self.btn_step_fwd = ttk.Button(btns_frame, text="Paso ▶", width=8, command=lambda: self._step_replay(1))
        self.btn_step_fwd.pack(side=tk.LEFT, padx=2)

        # Selector de Velocidad
        tk.Label(btns_frame, text="Velocidad:", bg=self.COLORS["panel"], fg=self.COLORS["text"]).pack(side=tk.LEFT, padx=(15, 5))
        self.speed_combo = ttk.Combobox(btns_frame, values=["0.5x", "1.0x", "2.0x", "5.0x"], width=6, state="readonly")
        self.speed_combo.set("1.0x")
        self.speed_combo.pack(side=tk.LEFT)
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_changed)

    def _on_map_changed(self):
        map_key = self.map_var.get()
        map_path = self.scenarios_info[map_key]
        self.current_grid = Grid(map_path)
        self.exit_lbl.config(text=str(self.current_grid.exit_pos))

        # 1. Limpiar repetición y resetear controles ANTES de generar el nuevo foco y redibujar
        self.best_history = []
        self.current_replay_turn = 0
        self.turn_slider.config(to=0)
        self.replay_status_lbl.config(text="Vista Previa del Mapa (Listo para configurar)")
        self.viewer_title_lbl.config(text=f"VISTA PREVIA: {map_key}")

        # 2. Restablecer foco aleatorio por defecto (que a su vez llama a _draw_current_state)
        self._generate_random_fire()

    def _generate_random_fire(self):
        if not self.current_grid:
            return
        foci = get_random_fire_foci(
            self.current_grid,
            count=1,
            min_dist_to_exit=7.0,
            spawn_pos=self.spawn_pos,
            min_dist_to_spawn=3.0
        )
        self.fire_focus = foci[0]
        self.fire_lbl.config(text=str(self.fire_focus))
        self._draw_current_state()

    def _set_click_mode(self, mode: str):
        if self.click_mode == mode:
            self.click_mode = None
            self.click_mode_lbl.config(text="")
        else:
            self.click_mode = mode
            txt = "👉 Haz clic en una celda libre para mover el INICIO (S)" if mode == "spawn" else "👉 Haz clic para situar el FOCO DE FUEGO (F)"
            self.click_mode_lbl.config(text=txt)

    def _on_canvas_click(self, event):
        if not self.current_grid or not self.click_mode:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        rows, cols = self.current_grid.rows, self.current_grid.cols

        cell_size = min(cw // cols, ch // rows)
        if cell_size <= 0:
            return

        offset_x = (cw - cell_size * cols) // 2
        offset_y = (ch - cell_size * rows) // 2

        c = (event.x - offset_x) // cell_size
        r = (event.y - offset_y) // cell_size

        if not self.current_grid.in_bounds(r, c) or self.current_grid.walls[r, c] or (r, c) == self.current_grid.exit_pos:
            return

        if self.click_mode == "spawn":
            self.spawn_pos = (r, c)
            self.spawn_lbl.config(text=str(self.spawn_pos))
            # Si el fuego quedó muy cerca, regenerar
            if self.fire_focus and math.hypot(r - self.fire_focus[0], c - self.fire_focus[1]) < 3.0:
                self._generate_random_fire()
        elif self.click_mode == "fire":
            # Verificar restricciones para fuego manual
            d_exit = math.hypot(r - self.current_grid.exit_pos[0], c - self.current_grid.exit_pos[1])
            d_spawn = math.hypot(r - self.spawn_pos[0], c - self.spawn_pos[1])

            if d_exit < 7.0:
                messagebox.showwarning("Restricción", f"El fuego debe estar a distancia euclidiana >= 7.0 de la salida (actual: {d_exit:.2f}).")
                return
            if d_spawn < 3.0:
                messagebox.showwarning("Restricción", f"El fuego debe estar a distancia euclidiana >= 3.0 de los agentes (actual: {d_spawn:.2f}).")
                return

            self.fire_focus = (r, c)
            self.fire_lbl.config(text=str(self.fire_focus))

        self.click_mode = None
        self.click_mode_lbl.config(text="")
        self._draw_current_state()

    def _draw_current_state(self):
        if not self.current_grid:
            return

        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        rows, cols = self.current_grid.rows, self.current_grid.cols
        cell_size = max(4, min(cw // cols, ch // rows))

        offset_x = (cw - cell_size * cols) // 2
        offset_y = (ch - cell_size * rows) // 2

        # 1. Si hay replay activo, leer snapshot del turno actual
        current_snapshot = None
        if self.best_history and 0 <= self.current_replay_turn < len(self.best_history):
            snap = self.best_history[self.current_replay_turn]
            if snap.get("fire") is not None and snap["fire"].shape == (rows, cols):
                current_snapshot = snap

        # 2. Dibujar celdas base
        for r in range(rows):
            for c in range(cols):
                x1 = offset_x + c * cell_size
                y1 = offset_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                is_wall = self.current_grid.walls[r, c]
                is_exit = (r, c) == self.current_grid.exit_pos

                # Determinar si tiene fuego
                has_fire = False
                if current_snapshot:
                    has_fire = bool(current_snapshot["fire"][r, c])
                elif self.fire_focus and (r, c) == self.fire_focus:
                    has_fire = True

                if is_wall:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.COLORS["wall"], outline=self.COLORS["panel_border"])
                elif is_exit:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.COLORS["exit"], outline="#ffffff", width=2)
                    if cell_size >= 12:
                        self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="E", fill="#1e1e2e", font=("Segoe UI", max(8, cell_size // 2), "bold"))
                elif has_fire:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.COLORS["fire"], outline=self.COLORS["fire_hot"])
                    if cell_size >= 12:
                        self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="🔥", font=("Segoe UI", max(7, cell_size // 2)))
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.COLORS["empty"], outline=self.COLORS["empty_border"])

        # 3. Dibujar Agentes
        if current_snapshot:
            # Agrupar agentes vivos por celda para mostrar congestión
            agent_counts: Dict[Coord, int] = {}
            for pos, state in current_snapshot["agents"]:
                if state.value == "ALIVE":
                    agent_counts[pos] = agent_counts.get(pos, 0) + 1

            for pos, count in agent_counts.items():
                r, c = pos
                x1 = offset_x + c * cell_size + 2
                y1 = offset_y + r * cell_size + 2
                x2 = offset_x + (c + 1) * cell_size - 2
                y2 = offset_y + (r + 1) * cell_size - 2

                agent_color = self.COLORS["agent"] if count <= 2 else self.COLORS["agent_cluster"]
                self.canvas.create_oval(x1, y1, x2, y2, fill=agent_color, outline="#ffffff")
                if cell_size >= 14:
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=str(count), fill="#1e1e2e", font=("Segoe UI", max(7, cell_size // 2), "bold"))
        else:
            # Vista previa: dibujar punto de spawn de agentes
            r, c = self.spawn_pos
            x1 = offset_x + c * cell_size + 2
            y1 = offset_y + r * cell_size + 2
            x2 = offset_x + (c + 1) * cell_size - 2
            y2 = offset_y + (r + 1) * cell_size - 2
            self.canvas.create_oval(x1, y1, x2, y2, fill=self.COLORS["spawn"], outline="#ffffff", width=2)
            if cell_size >= 12:
                self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="S", fill="#1e1e2e", font=("Segoe UI", max(8, cell_size // 2), "bold"))

    def _start_simulation(self):
        map_key = self.map_var.get()
        map_path = self.scenarios_info[map_key]
        num_agents = int(self.agents_scale.get())
        algo_choice = self.algo_var.get()
        iterations = int(self.iters_var.get())

        if not self.fire_focus:
            self._generate_random_fire()

        # Determinar algoritmos a correr
        algo_map = {
            "A*": a_star,
            "BFS": bfs,
            "Greedy": greedy,
            "DFS": dfs,
            "Genético": lambda g, s, e: genetic(g, s, e, population_size=40, generations=40, mutation_rate=0.06)
        }

        if algo_choice == "Comparar Todos":
            algos_to_run = algo_map
        else:
            algos_to_run = {algo_choice: algo_map[algo_choice]}

        # Limpiar resultados previos
        for row in self.tree.get_children():
            self.tree.delete(row)

        self.btn_run.config(state="disabled", text="⏳ Simulando...")
        self.root.update_idletasks()

        best_overall_run = None
        summary_results = []

        for name, func in algos_to_run.items():
            survival_rates = []
            clearance_times = []
            waited_turns_list = []
            best_run_for_algo = None

            for it in range(iterations):
                # Si hay más de 1 iteración, generar fuegos aleatorios en cada una
                foco_actual = self.fire_focus if iterations == 1 else get_random_fire_foci(
                    self.current_grid, count=1, min_dist_to_exit=7.0, spawn_pos=self.spawn_pos, min_dist_to_spawn=3.0
                )[0]

                sim = Simulation(
                    map_path=map_path,
                    num_agents=num_agents,
                    spawn_pos=self.spawn_pos,
                    fire_foci=[foco_actual],
                    spread_interval=3,
                    spread_probability=0.75,
                    cell_capacity=2,
                    algorithm_name=name,
                    algorithm_func=func
                )

                # Grabar historial de la simulación
                res = sim.run(max_turns=500, record_history=True)

                survival_rates.append(res["survival_rate"])
                waited_turns_list.append(res["avg_waited_turns"])
                if res["clearance_time"] is not None:
                    clearance_times.append(res["clearance_time"])

                if best_run_for_algo is None or res["survival_rate"] > best_run_for_algo["survival_rate"]:
                    best_run_for_algo = res

            # Estadísticos del algoritmo
            mean_surv = float(np.mean(survival_rates))
            std_surv = float(np.std(survival_rates))
            mean_clear = float(np.mean(clearance_times)) if clearance_times else 0.0
            std_clear = float(np.std(clearance_times)) if clearance_times else 0.0
            mean_wait = float(np.mean(waited_turns_list))

            surv_text = f"{mean_surv:.1f}%" if iterations == 1 else f"{mean_surv:.1f}% ± {std_surv:.1f}%"
            clear_text = f"{mean_clear:.0f} t" if clearance_times else "Sin evacuados"
            wait_text = f"{mean_wait:.1f} t"

            self.tree.insert("", tk.END, values=(name, surv_text, clear_text, wait_text))

            summary_results.append({
                "algo": name,
                "surv": mean_surv,
                "best_run": best_run_for_algo
            })

            if best_overall_run is None or (best_run_for_algo and best_run_for_algo["survival_rate"] > best_overall_run["survival_rate"]):
                best_overall_run = best_run_for_algo

        self.btn_run.config(state="normal", text="🚀 INICIAR SIMULACIÓN")

        # Cargar la mejor simulación en el reproductor interactivo
        if best_overall_run and "history" in best_overall_run and best_overall_run["history"]:
            self.best_history = best_overall_run["history"]
            self.best_run_meta = best_overall_run
            self.current_replay_turn = 0
            self.turn_slider.config(to=len(self.best_history) - 1)
            self.turn_slider.set(0)

            best_algo = best_overall_run["algorithm"]
            best_surv = best_overall_run["survival_rate"]
            best_clear = best_overall_run["clearance_time"]
            self.viewer_title_lbl.config(
                text=f"🏆 REPRODUCIENDO MEJOR SIMULACIÓN: {best_algo} (Supervivencia: {best_surv:.1f}%)"
            )

            self.res_summary_lbl.config(
                text=f"Simulación completada con éxito.\n"
                     f"• Mejor algoritmo: {best_algo}\n"
                     f"• Tasa de supervivencia: {best_surv:.1f}% ({best_overall_run['survivors']}/{num_agents})\n"
                     f"• Tiempo de despeje: {best_clear if best_clear is not None else 'Sin evacuados'} turnos\n"
                     f"• Bajas por fuego: {best_overall_run['dead']}"
            )

            self._update_replay_view()
            self._start_playback()

    def _update_replay_view(self):
        if not self.best_history:
            return

        self.turn_slider.set(self.current_replay_turn)
        snap = self.best_history[self.current_replay_turn]

        alive_cnt = sum(1 for _, st in snap["agents"] if st.value == "ALIVE")
        escaped_cnt = sum(1 for _, st in snap["agents"] if st.value == "ESCAPED")
        dead_cnt = sum(1 for _, st in snap["agents"] if st.value == "DEAD")

        self.replay_status_lbl.config(
            text=f"Turno: {self.current_replay_turn} / {len(self.best_history) - 1}  |  "
                 f"🏃 En Tránsito: {alive_cnt}  |  "
                 f"✅ Evacuados: {escaped_cnt}  |  "
                 f"💀 Bajas: {dead_cnt}"
        )
        self._draw_current_state()

    def _on_slider_moved(self, val):
        turn = int(val)
        if turn != self.current_replay_turn and self.best_history:
            self.current_replay_turn = turn
            self._update_replay_view()

    def _step_replay(self, delta: int):
        if not self.best_history:
            return
        new_turn = max(0, min(len(self.best_history) - 1, self.current_replay_turn + delta))
        if new_turn != self.current_replay_turn:
            self.current_replay_turn = new_turn
            self._update_replay_view()

    def _reset_replay(self):
        self.is_playing = False
        self.btn_play_pause.config(text="▶ Reproducir")
        self.current_replay_turn = 0
        self._update_replay_view()

    def _toggle_play(self):
        if not self.best_history:
            return
        self.is_playing = not self.is_playing
        self.btn_play_pause.config(text="⏸ Pausa" if self.is_playing else "▶ Reproducir")
        if self.is_playing:
            self._playback_loop()

    def _start_playback(self):
        self.is_playing = True
        self.btn_play_pause.config(text="⏸ Pausa")
        self._playback_loop()

    def _playback_loop(self):
        if not self.is_playing or not self.best_history:
            return

        if self.current_replay_turn < len(self.best_history) - 1:
            self.current_replay_turn += 1
            self._update_replay_view()
            self.root.after(self.play_speed_ms, self._playback_loop)
        else:
            self.is_playing = False
            self.btn_play_pause.config(text="▶ Reproducir")

    def _on_speed_changed(self, event):
        val = self.speed_combo.get()
        speed_map = {"0.5x": 300, "1.0x": 150, "2.0x": 75, "5.0x": 30}
        self.play_speed_ms = speed_map.get(val, 150)


def main():
    root = tk.Tk()
    app = EvacuationSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
