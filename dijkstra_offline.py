#!/usr/bin/env python3
"""
=============================================================
  PLANEJAMENTO DE CAMINHOS — GRUPO 2 — DIJKSTRA
  UERN — Ciência da Computação — Robótica
=============================================================

BIBLIOTECAS:
    numpy
    pillow          # PIL
    scipy
    matplotlib
    pyyaml

    pip install numpy pillow scipy matplotlib pyyaml --break-system-packages

DESCRIÇÃO:
  PRECISA ter o mapa na mesma pasta do arquivo python
  Lê o mapa gerado pelo gmapping, infla os obstáculos,
  roda o Dijkstra e gera os waypoints para o controlador.

USO:
  python3 dijkstra_offline.py

SAÍDA:
  - mapa_dijkstra.png  → visualização do caminho
  - waypoints.txt      → coordenadas para o controlador
"""

import os, sys, heapq, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# =============================================================
#  CONFIGURAÇÕES — AJUSTE AQUI
# =============================================================

MAP_YAML           = "map.yaml"
MAP_IMAGE          = None         # None = lê do YAML automaticamente

START_WORLD        = (2.0,  0.0)  # (x, y) em metros
GOAL_WORLD         = (10.0, 0.0)  # (x, y) em metros

INFLATION_RADIUS_M = 0.40         # raio de inflação dos obstáculos (m)
CELL_SIZE_M        = 0.20         # referência de tamanho de célula (m)

OUTPUT_IMAGE       = "mapa_dijkstra.png"
OUTPUT_WAYPOINTS   = "waypoints.txt"

# =============================================================
#  1. LEITURA DO MAPA
# =============================================================

def load_map(yaml_path, image_path=None):
    print("\n[1/5] Lendo o mapa...")

    if not os.path.exists(yaml_path):
        sys.exit(f"  ERRO: '{yaml_path}' não encontrado.")

    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    resolution = float(meta['resolution'])
    origin     = meta['origin']

    img_file = image_path or meta['image']
    if not os.path.isabs(img_file):
        img_file = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_file)

    if not os.path.exists(img_file):
        sys.exit(f"  ERRO: Imagem '{img_file}' não encontrada.")

    img       = Image.open(img_file).convert('L')
    img_array = np.array(img, dtype=np.uint8)
    height, width = img_array.shape

    # Só pixels pretos (<50) são obstáculos reais
    # Cinza (205) = desconhecido → tratado como livre
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[img_array < 50] = 1

    print(f"  Resolução  : {resolution:.4f} m/pixel")
    print(f"  Tamanho    : {width}x{height} px  ({width*resolution:.1f}x{height*resolution:.1f} m)")
    print(f"  Origem     : x={origin[0]:.2f}, y={origin[1]:.2f}")
    print(f"  Obstáculos : {grid.sum()} células ({100*grid.mean():.1f}%)")

    return grid, resolution, origin, img_array.astype(np.float32)

# =============================================================
#  2. INFLAÇÃO DOS OBSTÁCULOS
# =============================================================

def inflate_obstacles(grid, resolution, inflation_radius_m):
    print(f"\n[2/5] Inflando obstáculos (raio={inflation_radius_m}m)...")

    radius_cells = max(1, int(np.ceil(inflation_radius_m / resolution)))
    d = radius_cells
    y_idx, x_idx = np.ogrid[-d:d+1, -d:d+1]
    kernel   = (x_idx**2 + y_idx**2 <= d**2)
    inflated = binary_dilation(grid, structure=kernel).astype(np.uint8)

    print(f"  Raio em células    : {radius_cells}")
    print(f"  Células adicionadas: {inflated.sum() - grid.sum()}")

    return inflated

# =============================================================
#  3. CONVERSÃO COORDENADAS
# =============================================================

def world_to_cell(wx, wy, origin, resolution, height):
    col = int((wx - origin[0]) / resolution)
    row = int(height - (wy - origin[1]) / resolution)
    return (row, col)

def cell_to_world(row, col, origin, resolution, height):
    wx = origin[0] + col * resolution
    wy = origin[1] + (height - row) * resolution
    return (wx, wy)

def validate_cell(cell, grid, label):
    rows, cols = grid.shape
    r, c = cell
    if not (0 <= r < rows and 0 <= c < cols):
        sys.exit(f"  ERRO: {label} fora dos limites do mapa!")
    if grid[r, c] == 1:
        print(f"  AVISO: {label} em obstáculo, buscando célula livre próxima...")
        for dr in range(-10, 11):
            for dc in range(-10, 11):
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    print(f"  → Ajustado para ({nr},{nc})")
                    return (nr, nc)
        sys.exit(f"  ERRO: Nenhuma célula livre próxima de {label}.")
    return cell

# =============================================================
#  4. DIJKSTRA
# =============================================================

def dijkstra(grid, start, goal):
    print(f"\n[3/5] Rodando Dijkstra...")
    print(f"  Início : {start}  |  Destino: {goal}")

    rows, cols = grid.shape
    dist = np.full((rows, cols), np.inf, dtype=np.float32)
    dist[start] = 0.0
    prev = {}

    MOVES = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1,-1, 1.4142), (-1, 1, 1.4142), (1,-1, 1.4142), (1, 1, 1.4142)
    ]

    heap    = [(0.0, start)]
    visited = 0
    t0      = time.time()

    while heap:
        cost, current = heapq.heappop(heap)
        if current == goal:
            break
        if cost > dist[current]:
            continue
        visited += 1
        r, c = current
        for dr, dc, step_cost in MOVES:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                new_cost = cost + step_cost
                if new_cost < dist[nr, nc]:
                    dist[nr, nc] = new_cost
                    prev[(nr, nc)] = current
                    heapq.heappush(heap, (new_cost, (nr, nc)))

    elapsed = time.time() - t0

    if goal not in prev and start != goal:
        sys.exit("  ERRO: Caminho não encontrado!")

    path = []
    node = goal
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(start)
    path.reverse()

    print(f"  Células visitadas : {visited}")
    print(f"  Tempo             : {elapsed:.3f}s")
    print(f"  Waypoints         : {len(path)}")
    print(f"  Distância         : {dist[goal]*CELL_SIZE_M:.2f} m")

    return path, elapsed

# =============================================================
#  5. GERAR WAYPOINTS
# =============================================================

def generate_waypoints(path, origin, resolution, height, output_file):
    print(f"\n[4/5] Gerando waypoints...")

    waypoints = [cell_to_world(r, c, origin, resolution, height) for r, c in path]

    with open(output_file, 'w') as f:
        f.write("# Waypoints — Grupo 2 — Dijkstra\n")
        f.write("# x(m)  y(m)\n")
        f.write(f"# Total: {len(waypoints)}\n\n")
        for wx, wy in waypoints:
            f.write(f"{wx:.4f}  {wy:.4f}\n")

    print(f"  {len(waypoints)} waypoints salvos em '{output_file}'")
    return waypoints

# =============================================================
#  6. VISUALIZAÇÃO
# =============================================================

def draw_grid(ax, height, width, resolution, step_m=1.0):
    step_px = max(1, int(step_m / resolution))
    for c in range(0, width, step_px):
        ax.axvline(x=c, color='cyan', linewidth=0.3, alpha=0.5, zorder=2)
    for r in range(0, height, step_px):
        ax.axhline(y=r, color='cyan', linewidth=0.3, alpha=0.5, zorder=2)

def visualize(img_array, grid_original, grid_inflated, path,
              start_cell, goal_cell, output_file, resolution):
    print(f"\n[5/5] Gerando visualização...")

    height, width = grid_original.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Planejamento de Caminhos — Grupo 2 — Dijkstra",
                 fontsize=14, fontweight='bold')

    axes[0].set_title("Mapa Original (gmapping)")
    axes[0].imshow(img_array, cmap='gray', origin='upper')
    draw_grid(axes[0], height, width, resolution)

    display = np.zeros((height, width, 3), dtype=np.uint8)
    display[grid_inflated == 0]                    = [240, 240, 240]
    display[grid_original == 1]                    = [50,  50,  50]
    display[(grid_inflated==1)&(grid_original==0)] = [220, 80,  80]

    axes[1].set_title(f"Obstáculos Inflados (raio={INFLATION_RADIUS_M}m)")
    axes[1].imshow(display, origin='upper')
    draw_grid(axes[1], height, width, resolution)
    axes[1].legend(handles=[
        mpatches.Patch(color='#323232', label='Obstáculo real'),
        mpatches.Patch(color='#dc5050', label='Zona inflada'),
        mpatches.Patch(color='#f0f0f0', label='Livre'),
    ], loc='lower right', fontsize=7)

    axes[2].set_title("Caminho Planejado (Dijkstra)")
    axes[2].imshow(display, origin='upper')
    draw_grid(axes[2], height, width, resolution)
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    axes[2].plot(path_cols, path_rows, color='#00dd00', linewidth=2.0, label='Caminho', zorder=4)
    axes[2].scatter(path_cols, path_rows, color='#00dd00', s=8, zorder=5)
    axes[2].plot(start_cell[1], start_cell[0], 'o', color='#0055ff', markersize=10, label='Início', zorder=6)
    axes[2].plot(goal_cell[1],  goal_cell[0],  '*', color='#ff8800', markersize=14, label='Destino', zorder=6)
    axes[2].legend(loc='lower right', fontsize=7)

    for ax in axes:
        ax.set_xlabel("Coluna (pixels)")
        ax.set_ylabel("Linha (pixels)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Imagem salva em '{output_file}'")

# =============================================================
#  MAIN
# =============================================================

def main():
    print("="*55)
    print("  DIJKSTRA OFFLINE — GRUPO 2")
    print("="*55)

    grid_original, resolution, origin, img_array = load_map(MAP_YAML, MAP_IMAGE)
    height, width = grid_original.shape

    grid_inflated = inflate_obstacles(grid_original, resolution, INFLATION_RADIUS_M)

    start_cell = world_to_cell(START_WORLD[0], START_WORLD[1], origin, resolution, height)
    goal_cell  = world_to_cell(GOAL_WORLD[0],  GOAL_WORLD[1],  origin, resolution, height)
    print(f"\n  START {START_WORLD} → célula {start_cell}")
    print(f"  GOAL  {GOAL_WORLD}  → célula {goal_cell}")
    start_cell = validate_cell(start_cell, grid_inflated, "Início")
    goal_cell  = validate_cell(goal_cell,  grid_inflated, "Destino")

    path, exec_time = dijkstra(grid_inflated, start_cell, goal_cell)

    waypoints = generate_waypoints(path, origin, resolution, height, OUTPUT_WAYPOINTS)

    visualize(img_array, grid_original, grid_inflated, path,
              start_cell, goal_cell, OUTPUT_IMAGE, resolution)

    print("\n"+"="*55)
    print("  CONCLUÍDO!")
    print(f"  → Imagem    : {OUTPUT_IMAGE}")
    print(f"  → Waypoints : {OUTPUT_WAYPOINTS}")
    print(f"  → Dijkstra  : {exec_time:.3f}s")
    print(f"  → Pontos    : {len(waypoints)}")
    print("="*55)

if __name__ == "__main__":
    main()