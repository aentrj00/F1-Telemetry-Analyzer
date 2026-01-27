import fastf1 as ff1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ============================================
# CONFIGURACIÓN
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('graficos_tiempos_sesion'):
    os.makedirs('graficos_tiempos_sesion')

ff1.Cache.enable_cache('cache')

# ============================================
# LISTA DE PILOTOS
# ============================================

pilotos_info = {
    'VER': 'Max Verstappen',
    'PER': 'Sergio Perez',
    'LEC': 'Charles Leclerc',
    'SAI': 'Carlos Sainz',
    'NOR': 'Lando Norris',
    'PIA': 'Oscar Piastri',
    'HAM': 'Lewis Hamilton',
    'RUS': 'George Russell',
    'ALO': 'Fernando Alonso',
    'STR': 'Lance Stroll',
    'OCO': 'Esteban Ocon',
    'GAS': 'Pierre Gasly',
    'TSU': 'Yuki Tsunoda',
    'RIC': 'Daniel Ricciardo',
    'HUL': 'Nico Hulkenberg',
    'MAG': 'Kevin Magnussen',
    'ALB': 'Alexander Albon',
    'SAR': 'Logan Sargeant',
    'BOT': 'Valtteri Bottas',
    'ZHO': 'Zhou Guanyu'
}

# ============================================
# PROGRAMA PRINCIPAL
# ============================================

print("=" * 60)
print("EVOLUCION DE TIEMPOS DE VUELTA POR PILOTO")
print("=" * 60)

print("\nPilotos disponibles:")
for code, name in sorted(pilotos_info.items()):
    print(f"  {code} - {name}")

print("\n" + "=" * 60)

# Pedir datos
year = input("\nYear (ej: 2024): ").strip()
if not year:
    year = '2024'

gp = input("Gran Premio (ej: Bahrain, Monaco, Monza): ").strip()
if not gp:
    gp = 'Bahrain'

print("\nTipos de sesion:")
print("  R   - Carrera")
print("  Q   - Clasificacion")
print("  S   - Sprint")
print("  FP1 - Libres 1")
print("  FP2 - Libres 2")
print("  FP3 - Libres 3")

session_type = input("\nTipo de sesion (ej: FP1, Q, R): ").strip().upper()
if not session_type:
    session_type = 'FP1'

driver = input("\nPiloto (codigo 3 letras, ej: HAM): ").strip().upper()

if driver not in pilotos_info:
    print("\nError: Codigo de piloto no valido")
    exit()

print("\n" + "=" * 60)
print(f"Cargando sesion: {year} - {gp} GP ({session_type})")
print("=" * 60)

# Cargar sesión
try:
    session = ff1.get_session(int(year), gp, session_type)
    session.load()
    print("Sesion cargada correctamente\n")
except Exception as e:
    print(f"\nError: {e}")
    exit()

# ============================================
# OBTENER DATOS DEL PILOTO
# ============================================

print(f"Procesando vueltas de {pilotos_info[driver]}...")

driver_laps = session.laps.pick_drivers(driver)  # ← Cambiado pick_driver por pick_drivers

if len(driver_laps) == 0:
    print(f"\nError: No hay datos de {driver} en esta sesion")
    exit()

# Filtrar vueltas válidas
driver_laps = driver_laps[driver_laps['LapTime'].notna()]

print(f"  -> {len(driver_laps)} vueltas validas encontradas")

# Extraer datos
lap_times_seconds = driver_laps['LapTime'].dt.total_seconds().to_numpy()
lap_start_times = driver_laps['LapStartTime'].to_numpy()

# Convertir lap_start_times a minutos desde inicio de sesión
session_start = lap_start_times[0]

# ← ESTO ES LO QUE ESTABA FALLANDO - Conversión correcta
time_minutes = []
for t in lap_start_times:
    delta = pd.Timedelta(t - session_start)  # Convertir a pandas Timedelta
    time_minutes.append(delta.total_seconds() / 60)

time_minutes = np.array(time_minutes)

# Detectar compound de neumático si está disponible
try:
    compounds = driver_laps['Compound'].to_numpy()
    has_compound = True
except:
    has_compound = False
    print("  -> No hay datos de compuestos de neumaticos")

# ============================================
# CREAR GRÁFICO
# ============================================

fig, ax = plt.subplots(figsize=(16, 9))

# Colores por compuesto
compound_colors = {
    'SOFT': 'red',
    'MEDIUM': 'yellow',
    'HARD': 'white',
    'INTERMEDIATE': 'green',
    'WET': 'blue'
}

if has_compound:
    # Plotear con colores por compuesto
    for compound in set(compounds):
        if pd.isna(compound):  # Ignorar valores NaN
            continue
            
        mask = compounds == compound
        color = compound_colors.get(compound, 'gray')
        edge_color = 'black' if compound in ['MEDIUM', 'HARD'] else color
        
        ax.scatter(time_minutes[mask], 
                  lap_times_seconds[mask],
                  c=color, edgecolors=edge_color, linewidths=1.5,
                  s=100, alpha=0.7, label=compound, zorder=5)
    
    # Línea conectando todos los puntos
    ax.plot(time_minutes, lap_times_seconds, 
            color='gray', linewidth=1, alpha=0.3, zorder=1)
else:
    # Plotear sin distinción de compuesto
    ax.scatter(time_minutes, lap_times_seconds, 
              c='blue', s=100, alpha=0.7, zorder=5)
    ax.plot(time_minutes, lap_times_seconds, 
            color='blue', linewidth=2, alpha=0.5)

# Marcar vuelta más rápida
fastest_idx = np.argmin(lap_times_seconds)
ax.scatter(time_minutes[fastest_idx], lap_times_seconds[fastest_idx],
          c='lime', s=300, marker='*', edgecolors='black', 
          linewidths=2, zorder=10, label='Vuelta mas rapida')

# ============================================
# CONFIGURAR GRÁFICO
# ============================================

ax.set_xlabel('Tiempo de sesion (minutos)', fontsize=14, fontweight='bold')
ax.set_ylabel('Tiempo de vuelta (segundos)', fontsize=14, fontweight='bold')

session_names = {
    'R': 'Carrera',
    'Q': 'Clasificacion',
    'S': 'Sprint',
    'FP1': 'Libres 1',
    'FP2': 'Libres 2',
    'FP3': 'Libres 3'
}
session_text = session_names.get(session_type, session_type)

ax.set_title(f'{pilotos_info[driver]} ({driver}) - {gp} GP {year} - {session_text}\n' +
             f'Evolucion de tiempos de vuelta',
             fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Leyenda
ax.legend(loc='best', fontsize=11, framealpha=0.9)

# Info adicional
fastest_time = lap_times_seconds[fastest_idx]
avg_time = np.mean(lap_times_seconds)

info_text = (f'Vuelta mas rapida: {fastest_time:.3f}s\n'
             f'Media: {avg_time:.3f}s\n'
             f'Total vueltas: {len(lap_times_seconds)}')

ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Guardar
filename = f'graficos_tiempos_sesion/{gp}_{year}_{session_type}_{driver}_tiempos.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nGrafico guardado: {filename}")

print("\n" + "=" * 60)
print("Abriendo ventana con el grafico...")
print("(Cierra la ventana para continuar)")
print("=" * 60)

plt.show(block=True)

print("\n" + "=" * 60)
print("ANALISIS COMPLETADO")
print("=" * 60)
print("\nInterpretacion:")
print("  - Eje X: Tiempo transcurrido de sesion")
print("  - Eje Y: Tiempo de cada vuelta")
print("  - Green Star: Vuelta mas rapida")
if has_compound:
    print("  - Colores: Compuesto de neumatico usado")
    print("    * Rojo = Blando | Amarillo = Medio | Blanco = Duro")
print("\n" + "=" * 60)