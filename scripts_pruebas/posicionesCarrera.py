import fastf1 as ff1
import matplotlib
matplotlib.use('TkAgg')  # ← ESTO FUERZA LA VENTANA EMERGENTE
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# CONFIGURACIÓN
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('graficos_posiciones_carrera'):
    os.makedirs('graficos_posiciones_carrera')

ff1.Cache.enable_cache('cache')

# ============================================
# PROGRAMA PRINCIPAL
# ============================================

print("=" * 60)
print("EVOLUCION DE POSICIONES EN CARRERA + PIT STOPS")
print("=" * 60)

# Pedir datos
year = input("Year (ej: 2024): ").strip()
if not year:
    year = '2024'

gp = input("Gran Premio (ej: Bahrain, Monaco, Monza): ").strip()
if not gp:
    gp = 'Bahrain'

print("\n" + "=" * 60)
print(f"Cargando carrera: {year} - {gp} GP")
print("=" * 60)

# Cargar sesión de CARRERA
try:
    session = ff1.get_session(int(year), gp, 'R')
    session.load()
    print("Carrera cargada correctamente\n")
except Exception as e:
    print(f"\nError: {e}")
    exit()

# ============================================
# OBTENER DATOS
# ============================================

print("Procesando datos de posiciones...")

# Obtener todas las vueltas
laps = session.laps

# Obtener pilotos que terminaron la carrera (top 10)
resultados = session.results
top_drivers = resultados.head(10)['Abbreviation'].tolist()

print(f"\nPilotos a analizar (Top 10):")
for i, driver in enumerate(top_drivers, 1):
    driver_info = resultados[resultados['Abbreviation'] == driver].iloc[0]
    print(f"  {i:2d}. {driver} - {driver_info['FullName']}")

# ============================================
# CREAR GRÁFICO
# ============================================

fig, ax = plt.subplots(figsize=(16, 10))

# Colores para cada piloto
colors = plt.cm.tab20(np.linspace(0, 1, len(top_drivers)))

print("\nGenerando grafico...")

for idx, driver in enumerate(top_drivers):
    # Obtener vueltas del piloto
    driver_laps = laps.pick_drivers(driver)  # ← Cambié pick_driver por pick_drivers
    
    if len(driver_laps) == 0:
        continue
    
    # Extraer número de vuelta y posición
    lap_numbers = driver_laps['LapNumber'].to_numpy()
    positions = driver_laps['Position'].to_numpy()
    
    # Plotear línea de posición
    ax.plot(lap_numbers, positions, 
            marker='o', markersize=4, linewidth=2,
            label=driver, color=colors[idx])

# Obtener pit stops
print("\nDetectando pit stops...")
try:
    pit_stops = session.laps[session.laps['PitInTime'].notna()]
    
    if len(pit_stops) > 0:
        # Agrupar por piloto
        for driver in top_drivers:
            driver_pits = pit_stops[pit_stops['Driver'] == driver]
            
            if len(driver_pits) > 0:
                for _, pit in driver_pits.iterrows():
                    lap_num = pit['LapNumber']
                    position = pit['Position']
                    
                    # Marcar pit stop con X grande
                    ax.plot(lap_num, position, 'rX', 
                           markersize=15, markeredgewidth=3,
                           zorder=10)
        
        print(f"  -> {len(pit_stops)} pit stops detectados")
    else:
        print("  -> No se detectaron pit stops en los datos")
        
except Exception as e:
    print(f"  -> No se pudieron obtener pit stops: {e}")

# ============================================
# CONFIGURAR GRÁFICO
# ============================================

ax.set_xlabel('Numero de Vuelta', fontsize=14, fontweight='bold')
ax.set_ylabel('Posicion', fontsize=14, fontweight='bold')
ax.set_title(f'Evolucion de Posiciones - {gp} GP {year}\n(X roja = Pit Stop)', 
             fontsize=16, fontweight='bold', pad=20)

# Invertir eje Y (1º arriba, 20º abajo)
ax.invert_yaxis()

# Grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Leyenda fuera del gráfico
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
          fontsize=10, framealpha=0.9)

# Limites del eje Y
ax.set_ylim(21, 0)

plt.tight_layout()

# Guardar
filename = f'graficos_posiciones_carrera/{gp}_{year}_posiciones_carrera.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nGrafico guardado: {filename}")

print("\n" + "=" * 60)
print("Abriendo ventana con el grafico...")
print("(Cierra la ventana para continuar)")
print("=" * 60)

plt.show(block=True)  # ← block=True asegura que espere

print("\n" + "=" * 60)
print("ANALISIS COMPLETADO")
print("=" * 60)
print("\nInterpretacion:")
print("  - Lineas = trayectoria de cada piloto")
print("  - X rojas = Pit stops")
print("  - Caidas bruscas tras X = tiempo perdido en boxes")
print("  - Subidas tras X = undercut exitoso")
print("\n" + "=" * 60)