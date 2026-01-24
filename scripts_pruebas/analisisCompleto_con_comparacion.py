import fastf1 as ff1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np
import os

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('graficos_vuelta_carrera'):
    os.makedirs('graficos_vuelta_carrera')

ff1.Cache.enable_cache('cache')

# ============================================
# FUNCIÓN PARA CREAR GRÁFICO DE UN PILOTO
# ============================================

def crear_grafico_piloto(session, driver_code, driver_name, gp_name, session_type):
    """
    Crea gráfico del circuito con velocidades y puntos de frenada
    """
    print(f"\n  Procesando datos de {driver_name}...")
    
    # Obtener vuelta rápida
    lap = session.laps.pick_driver(driver_code).pick_fastest()
    telemetry = lap.get_telemetry()
    
    # Extraer datos
    x = telemetry['X'].to_numpy()
    y = telemetry['Y'].to_numpy()
    speed = telemetry['Speed'].to_numpy()
    brake = telemetry['Brake'].to_numpy()
    distance = telemetry['Distance'].to_numpy()
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # ===== CIRCUITO COLOREADO POR VELOCIDAD =====
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    cmap = cm.plasma
    norm = plt.Normalize(speed.min(), speed.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=6)
    lc.set_array(speed)
    
    ax.add_collection(lc)
    
    # ===== PUNTOS DE FRENADA =====
    frenadas = []
    for i in range(1, len(brake)):
        if brake[i] > 0 and brake[i-1] == 0:
            frenadas.append(i)
    
    print(f"    -> {len(frenadas)} puntos de frenada detectados")
    
    # Plotear puntos de frenada
    for idx in frenadas:
        ax.plot(x[idx], y[idx], 'ro', markersize=12, 
                markeredgewidth=2, markeredgecolor='white', 
                zorder=10, label='Punto de frenada' if idx == frenadas[0] else '')
    
    # ===== MARCADORES DE DISTANCIA =====
    marcadores = range(0, int(distance.max()), 500)
    for dist in marcadores:
        idx = (np.abs(distance - dist)).argmin()
        ax.plot(x[idx], y[idx], 'ko', markersize=6, zorder=9)
        ax.text(x[idx], y[idx], f'{int(dist)}m', 
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='yellow', alpha=0.7),
                zorder=9)
    
    # ===== CONFIGURACIÓN DEL GRÁFICO =====
    ax.set_xlim(x.min() - 300, x.max() + 300)
    ax.set_ylim(y.min() - 300, y.max() + 300)
    ax.set_xlabel('Posicion X (metros)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Posicion Y (metros)', fontsize=12, fontweight='bold')
    
    # Tipo de sesión en texto
    session_names = {
        'R': 'Carrera',
        'Q': 'Clasificacion',
        'S': 'Sprint',
        'FP1': 'Libres 1',
        'FP2': 'Libres 2',
        'FP3': 'Libres 3'
    }
    session_text = session_names.get(session_type, session_type)
    
    ax.set_title(f'{driver_name} ({driver_code}) - {gp_name} GP - {session_text}\n' + 
                 f'Vuelta rapida | Tiempo: {lap["LapTime"]} | Puntos de frenada: {len(frenadas)}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Colorbar
    cbar = plt.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label('Velocidad (km/h)', fontsize=12, fontweight='bold')
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Info adicional
    info_text = (f'Velocidad maxima: {speed.max():.1f} km/h\n'
                 f'Velocidad minima: {speed.min():.1f} km/h\n'
                 f'Distancia total: {distance.max():.0f} m\n'
                 f'Frenadas: {len(frenadas)}')
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar con nombre descriptivo
    filename = f'graficos/{gp_name}_{session_type}_{driver_code}_circuito.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    -> Guardado: {filename}")
    
    return lap["LapTime"], len(frenadas)

# ============================================
# PROGRAMA PRINCIPAL
# ============================================


print("ANALIZADOR DE TELEMETRIA F1 - COMPARADOR DE PILOTOS")


# Lista de pilotos
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

print("\nPilotos disponibles:")
for code, name in sorted(pilotos_info.items()):
    print(f"  {code} - {name}")



# Pedir datos
year = input("\nYear (ej: 2024): ").strip()
if not year:
    year = '2024'

gp = input("Gran Premio (ej: Bahrain, Monaco, Monza): ").strip()
if not gp:
    gp = 'Bahrain'

print("\nTipos de sesion disponibles:")
print("  R   - Carrera")
print("  Q   - Clasificacion")
print("  S   - Sprint")
print("  FP1 - Libres 1")
print("  FP2 - Libres 2")
print("  FP3 - Libres 3")

session_type = input("\nTipo de sesion (ej: R, Q, FP1): ").strip().upper()
if not session_type:
    session_type = 'R'


piloto1 = input("Piloto 1 (codigo 3 letras, ej: VER): ").strip().upper()
piloto2 = input("Piloto 2 (codigo 3 letras, ej: NOR): ").strip().upper()

# Validar
if piloto1 not in pilotos_info or piloto2 not in pilotos_info:
    print("\nError: Codigo de piloto no valido")
    exit()

if piloto1 == piloto2:
    print("\nError: Debes elegir dos pilotos diferentes")
    exit()


print(f"Cargando sesion: {year} - {gp} GP ({session_type})")


# Cargar sesión
try:
    session = ff1.get_session(int(year), gp, session_type)
    session.load()
    print("Sesion cargada correctamente\n")
except Exception as e:
    print(f"\nError al cargar sesion: {e}")
    print("Verifica que el anio, GP y tipo de sesion sean correctos")
    exit()

# Crear gráficos
print("GENERANDO GRAFICOS")


try:
    tiempo1, frenadas1 = crear_grafico_piloto(session, piloto1, pilotos_info[piloto1], gp, session_type)
    tiempo2, frenadas2 = crear_grafico_piloto(session, piloto2, pilotos_info[piloto2], gp, session_type)
    
    
    print("COMPARACION DE RESULTADOS")
    
    print(f"\n{pilotos_info[piloto1]:20} ({piloto1}):")
    print(f"  Tiempo: {tiempo1}")
    print(f"  Frenadas: {frenadas1}")
    
    print(f"\n{pilotos_info[piloto2]:20} ({piloto2}):")
    print(f"  Tiempo: {tiempo2}")
    print(f"  Frenadas: {frenadas2}")
    
    # Calcular diferencias
    diff_tiempo = abs(tiempo1.total_seconds() - tiempo2.total_seconds())
    mas_rapido = piloto1 if tiempo1 < tiempo2 else piloto2
    
    diff_frenadas = abs(frenadas1 - frenadas2)
    
    print(f"\n{'='*60}")
    print(f"Mas rapido: {pilotos_info[mas_rapido]} por {diff_tiempo:.3f} segundos")
    
    if diff_frenadas > 0:
        mas_frenadas = piloto1 if frenadas1 > frenadas2 else piloto2
        print(f"Mas frenadas: {pilotos_info[mas_frenadas]} ({diff_frenadas} frenadas extra)")
    
    
    print("ANALISIS COMPLETADO")
    
    print(f"\nGraficos guardados en 'graficos/':")
    print(f"  - {gp}_{session_type}_{piloto1}_circuito.png")
    print(f"  - {gp}_{session_type}_{piloto2}_circuito.png")
    print("\nCompara visualmente:")
    print("  - Zonas de frenada (puntos rojos)")
    print("  - Velocidades en cada sector (colores)")
    
except Exception as e:
    print(f"\nError al procesar datos: {e}")
    print("Verifica que ambos pilotos participaron en esta sesion")


