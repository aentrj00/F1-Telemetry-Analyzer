import fastf1 as ff1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np
import os

if not os.path.exists('cache'):
    os.makedirs('cache')

ff1.Cache.enable_cache('cache')

print("Descargando datos...")

session = ff1.get_session(2024, 'Bahrain', 'R')
session.load()

ver_lap = session.laps.pick_driver('VER').pick_fastest()

telemetry_ver = ver_lap.get_telemetry()

# Gráfico con forma del circuito
fig_ver, ax_ver = plt.subplots(figsize=(12, 12))

x_ver = telemetry_ver['X'].to_numpy()
y_ver = telemetry_ver['Y'].to_numpy()
speed_ver = telemetry_ver['Speed'].to_numpy()
distance_ver = telemetry_ver['Distance'].to_numpy()

# Crear LineCollection
points_ver = np.array([x_ver, y_ver]).T.reshape(-1, 1, 2)
segments_ver = np.concatenate([points_ver[:-1], points_ver[1:]], axis=1)

cmap = cm.plasma
norm = plt.Normalize(speed_ver.min(), speed_ver.max())
lc = LineCollection(segments_ver, cmap=cmap, norm=norm, linewidth=5)
lc.set_array(speed_ver)

ax_ver.add_collection(lc)

# Añadir marcadores cada 500 metros
marcadores_distancia_ver = range(0, int(distance_ver.max()), 500)
for dist in marcadores_distancia_ver:
    # Encontrar índice más cercano a esta distancia
    idx = (np.abs(distance_ver - dist)).argmin()

    # Plotear punto y texto
    ax_ver.plot(x_ver[idx], y_ver[idx], 'ko', markersize=8, zorder=10)
    ax_ver.text(x_ver[idx], y_ver[idx], f'{int(dist)}m', 
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax_ver.set_xlim(x_ver.min() - 200, x_ver.max() + 200)
ax_ver.set_ylim(y_ver.min() - 200, y_ver.max() + 200)
ax_ver.set_xlabel('Posición X (metros)', fontsize=12)
ax_ver.set_ylabel('Posición Y (metros)', fontsize=12)
ax_ver.set_title('Circuito Bahrain 2024 - Verstappen (Vuelta rápida)\nColoreado por velocidad', fontsize=14)
ax_ver.set_aspect('equal')
cbar = plt.colorbar(lc, ax=ax_ver)
cbar.set_label('Velocidad (km/h)', fontsize=12)

plt.tight_layout()
plt.savefig('circuito_con_distancias.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado!")


# Obtener y mostrar telemetría de Lando Norris
nor_lap = session.laps.pick_driver('NOR').pick_fastest()
telemetry_nor = nor_lap.get_telemetry()
