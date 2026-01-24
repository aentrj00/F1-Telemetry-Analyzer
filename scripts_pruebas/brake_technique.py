import fastf1 as ff1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import os

# ============================================
# CONFIGURACIÓN
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('grafico_brake_technique'):
    os.makedirs('grafico_brake_technique')

ff1.Cache.enable_cache('cache')

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def detectar_curvas_principales(telemetry, num_curvas=6):
    """
    Detecta las curvas principales del circuito
    Basado en velocidad mínima y uso de freno
    """
    speed = telemetry['Speed'].to_numpy()
    brake = telemetry['Brake'].to_numpy()
    distance = telemetry['Distance'].to_numpy()
    
    curvas = []
    ventana = 50
    
    for i in range(ventana, len(speed) - ventana):
        if speed[i] < np.min(speed[i-ventana:i+ventana]) + 5:
            if np.max(brake[i-ventana:i]) > 0.5:
                if not curvas or (distance[i] - curvas[-1]['apex_distance']) > 200:
                    curvas.append({
                        'apex_idx': i,
                        'apex_distance': distance[i],
                        'apex_speed': speed[i]
                    })
    
    curvas_sorted = sorted(curvas, key=lambda x: x['apex_speed'])
    return curvas_sorted[:num_curvas]

def analizar_curva_completa(telemetry, apex_idx, piloto_nombre):
    """
    Analiza en detalle una curva con TODAS las métricas
    """
    ventana = 200  # Ventana más amplia para capturar todo
    
    start_idx = max(0, apex_idx - ventana)
    end_idx = min(len(telemetry), apex_idx + ventana)
    
    section = telemetry.iloc[start_idx:end_idx].copy()
    section = section.reset_index(drop=True)
    
    speed = section['Speed'].to_numpy()
    brake = section['Brake'].to_numpy()
    throttle = section['Throttle'].to_numpy()
    distance = section['Distance'].to_numpy()
    
    apex_local = apex_idx - start_idx
    
    # ===== 1. INICIO DE FRENADA =====
    brake_start_idx = None
    for i in range(apex_local, -1, -1):
        if brake[i] > 0 and (i == 0 or brake[i-1] == 0):
            brake_start_idx = i
            break
    
    if brake_start_idx is None:
        return None
    
    # ===== 2. FIN DE FRENADA (brake vuelve a 0) =====
    brake_end_idx = None
    for i in range(apex_local, len(brake)):
        if brake[i] == 0 and i > apex_local:
            brake_end_idx = i
            break
    
    if brake_end_idx is None:
        brake_end_idx = min(apex_local + 50, len(brake) - 1)
    
    # ===== 3. VUELTA AL ACELERADOR 100% =====
    throttle_full_idx = None
    for i in range(apex_local, len(throttle)):
        if throttle[i] >= 99:  # 100% throttle
            throttle_full_idx = i
            break
    
    if throttle_full_idx is None:
        throttle_full_idx = min(apex_local + 100, len(throttle) - 1)
    
    # ===== 4. PUNTO DE MÁXIMA FRENADA =====
    max_brake_idx = brake_start_idx + np.argmax(brake[brake_start_idx:apex_local+1])
    
    # ===== 5. CALCULAR MÉTRICAS =====
    
    # Distancia 1: Hasta apex
    dist_frenada_apex = distance[apex_local] - distance[brake_start_idx]
    
    # Distancia 2: Hasta soltar freno completamente
    dist_freno_total = distance[brake_end_idx] - distance[brake_start_idx]
    
    # Distancia 3: Hasta volver a 100% throttle
    dist_sin_acelerar = distance[throttle_full_idx] - distance[brake_start_idx]
    
    # Trail braking: distancia frenando DESPUÉS del apex
    trail_braking = distance[brake_end_idx] - distance[apex_local]
    
    resultado = {
        'piloto': piloto_nombre,
        
        # Velocidades
        'velocidad_entrada': speed[brake_start_idx],
        'velocidad_apex': speed[apex_local],
        'velocidad_salida': speed[throttle_full_idx],
        
        # Distancias
        'dist_frenada_apex': dist_frenada_apex,
        'dist_freno_total': dist_freno_total,
        'dist_sin_acelerar': dist_sin_acelerar,
        'trail_braking': trail_braking,
        
        # Intensidades
        'brake_max': brake[max_brake_idx] * 100,
        
        # Posiciones absolutas
        'distancia_inicio_freno': distance[brake_start_idx],
        'distancia_apex': distance[apex_local],
        'distancia_fin_freno': distance[brake_end_idx],
        'distancia_throttle_full': distance[throttle_full_idx],
        
        # Índices para marcar en gráfico
        'idx_brake_start': brake_start_idx,
        'idx_apex': apex_local,
        'idx_brake_end': brake_end_idx,
        'idx_throttle_full': throttle_full_idx,
        
        # Datos para gráfico (normalizado a distancia relativa)
        'distance_plot': distance - distance[brake_start_idx],
        'speed_plot': speed,
        'brake_plot': brake * 100,
        'throttle_plot': throttle * 100
    }
    
    return resultado

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

print("=" * 70)
print("ANALISIS COMPLETO DE TECNICA DE FRENADA")
print("=" * 70)

print("\nPilotos disponibles:")
for code, name in sorted(pilotos_info.items()):
    print(f"  {code} - {name}")

print("\n" + "=" * 70)

# Pedir datos
year = input("\nAnio (ej: 2024): ").strip()
if not year:
    year = '2024'

gp = input("Gran Premio (ej: Bahrain, Monaco, Monza): ").strip()
if not gp:
    gp = 'Bahrain'

print("\nTipos de sesion:")
print("  R   - Carrera")
print("  Q   - Clasificacion")
print("  FP1 - Libres 1")

session_type = input("\nTipo de sesion (ej: Q, R, FP1): ").strip().upper()
if not session_type:
    session_type = 'Q'

print("\n" + "=" * 70)
piloto1 = input("Piloto 1 (ej: VER): ").strip().upper()
piloto2 = input("Piloto 2 (ej: NOR): ").strip().upper()

if piloto1 not in pilotos_info or piloto2 not in pilotos_info:
    print("\nError: Codigo invalido")
    exit()

if piloto1 == piloto2:
    print("\nError: Elige pilotos diferentes")
    exit()

print("\n" + "=" * 70)
print(f"Cargando sesion: {year} - {gp} GP ({session_type})")
print("=" * 70)

# Cargar sesión
try:
    session = ff1.get_session(int(year), gp, session_type)
    session.load()
    print("Sesion cargada\n")
except Exception as e:
    print(f"\nError: {e}")
    exit()

# Obtener vueltas rápidas
print(f"Obteniendo vueltas rapidas...")
lap1 = session.laps.pick_drivers(piloto1).pick_fastest()
lap2 = session.laps.pick_drivers(piloto2).pick_fastest()

tel1 = lap1.get_telemetry()
tel2 = lap2.get_telemetry()

tiempo_diff = (lap1['LapTime'] - lap2['LapTime']).total_seconds()
mas_rapido = piloto1 if tiempo_diff < 0 else piloto2

print(f"  {pilotos_info[piloto1]}: {lap1['LapTime']}")
print(f"  {pilotos_info[piloto2]}: {lap2['LapTime']}")
print(f"  Diferencia: {abs(tiempo_diff):.3f}s (mas rapido: {mas_rapido})")

# Detectar curvas principales
print(f"\nDetectando curvas principales del circuito...")
curvas = detectar_curvas_principales(tel1, num_curvas=6)
print(f"  -> {len(curvas)} curvas identificadas\n")

# Analizar cada curva
print("Analizando tecnica de frenada en cada curva...")
print("=" * 70)

comparaciones = []

for idx, curva in enumerate(curvas, 1):
    print(f"\nCURVA {idx} (apex a {curva['apex_distance']:.0f}m)")
    print("-" * 70)
    
    analisis1 = analizar_curva_completa(tel1, curva['apex_idx'], pilotos_info[piloto1])
    analisis2 = analizar_curva_completa(tel2, curva['apex_idx'], pilotos_info[piloto2])
    
    if analisis1 and analisis2:
        comparaciones.append({
            'curva_num': idx,
            'p1': analisis1,
            'p2': analisis2
        })
        
        # Mostrar métricas detalladas
        print(f"\n{piloto1} ({pilotos_info[piloto1]}):")
        print(f"  Velocidad entrada:       {analisis1['velocidad_entrada']:>6.1f} km/h")
        print(f"  Frenada maxima:          {analisis1['brake_max']:>6.1f} %")
        print(f"  Velocidad apex:          {analisis1['velocidad_apex']:>6.1f} km/h")
        print(f"  Velocidad salida:        {analisis1['velocidad_salida']:>6.1f} km/h")
        print(f"  Dist. frenada (apex):    {analisis1['dist_frenada_apex']:>6.1f} m")
        print(f"  Dist. freno total:       {analisis1['dist_freno_total']:>6.1f} m")
        print(f"  Dist. sin acelerar:      {analisis1['dist_sin_acelerar']:>6.1f} m")
        print(f"  Trail braking:           {analisis1['trail_braking']:>6.1f} m")
        
        print(f"\n{piloto2} ({pilotos_info[piloto2]}):")
        print(f"  Velocidad entrada:       {analisis2['velocidad_entrada']:>6.1f} km/h")
        print(f"  Frenada maxima:          {analisis2['brake_max']:>6.1f} %")
        print(f"  Velocidad apex:          {analisis2['velocidad_apex']:>6.1f} km/h")
        print(f"  Velocidad salida:        {analisis2['velocidad_salida']:>6.1f} km/h")
        print(f"  Dist. frenada (apex):    {analisis2['dist_frenada_apex']:>6.1f} m")
        print(f"  Dist. freno total:       {analisis2['dist_freno_total']:>6.1f} m")
        print(f"  Dist. sin acelerar:      {analisis2['dist_sin_acelerar']:>6.1f} m")
        print(f"  Trail braking:           {analisis2['trail_braking']:>6.1f} m")
        
        # Diferencias clave
        print(f"\nDIFERENCIAS ({piloto1} - {piloto2}):")
        diff_vel_ent = analisis1['velocidad_entrada'] - analisis2['velocidad_entrada']
        diff_vel_apex = analisis1['velocidad_apex'] - analisis2['velocidad_apex']
        diff_trail = analisis1['trail_braking'] - analisis2['trail_braking']
        diff_dist_apex = analisis1['dist_frenada_apex'] - analisis2['dist_frenada_apex']
        
        print(f"  Vel. entrada:    {diff_vel_ent:>+7.1f} km/h")
        print(f"  Vel. apex:       {diff_vel_apex:>+7.1f} km/h")
        print(f"  Trail braking:   {diff_trail:>+7.1f} m")
        print(f"  Dist. frenada:   {diff_dist_apex:>+7.1f} m")

# ============================================
# CREAR GRÁFICOS MEJORADOS
# ============================================

print("\n" + "=" * 70)
print("Generando graficos comparativos mejorados...")
print("=" * 70)

num_curvas = len(comparaciones)
fig = plt.figure(figsize=(20, 5 * num_curvas))
gs = GridSpec(num_curvas, 2, figure=fig, hspace=0.4, wspace=0.3)

for idx, comp in enumerate(comparaciones):
    p1 = comp['p1']
    p2 = comp['p2']
    
    # ========== SUBPLOT PILOTO 1 ==========
    ax1 = fig.add_subplot(gs[idx, 0])
    
    # Velocidad (línea azul)
    ax1.plot(p1['distance_plot'], p1['speed_plot'], 
             'b-', linewidth=2.5, label='Velocidad', zorder=3)
    ax1.set_ylabel('Velocidad (km/h)', color='b', fontweight='bold', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Marcadores de puntos clave
    dist_plot = p1['distance_plot']
    
    # Inicio de frenada (rojo)
    ax1.plot(dist_plot[p1['idx_brake_start']], 
             p1['speed_plot'][p1['idx_brake_start']], 
             'ro', markersize=12, markeredgewidth=2, 
             markeredgecolor='darkred', zorder=10, label='Inicio freno')
    
    # Apex (amarillo)
    ax1.plot(dist_plot[p1['idx_apex']], 
             p1['speed_plot'][p1['idx_apex']], 
             'o', color='gold', markersize=12, markeredgewidth=2, 
             markeredgecolor='orange', zorder=10, label='Apex')
    
    # Fin de freno (naranja)
    ax1.plot(dist_plot[p1['idx_brake_end']], 
             p1['speed_plot'][p1['idx_brake_end']], 
             'o', color='orange', markersize=10, markeredgewidth=2, 
             markeredgecolor='darkorange', zorder=10, label='Fin freno')
    
    # 100% throttle (verde)
    ax1.plot(dist_plot[p1['idx_throttle_full']], 
             p1['speed_plot'][p1['idx_throttle_full']], 
             'go', markersize=12, markeredgewidth=2, 
             markeredgecolor='darkgreen', zorder=10, label='100% gas')
    
    # Eje secundario - Freno y acelerador
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(dist_plot, 0, p1['brake_plot'],
                          color='red', alpha=0.3, label='Freno %')
    ax1_twin.fill_between(dist_plot, 0, p1['throttle_plot'],
                          color='green', alpha=0.2, label='Acelerador %')
    ax1_twin.set_ylabel('Freno / Acelerador (%)', fontweight='bold', fontsize=11)
    ax1_twin.set_ylim(0, 105)
    
    ax1.set_xlabel('Distancia desde inicio frenada (m)', fontweight='bold', fontsize=11)
    ax1.set_title(f'Curva {comp["curva_num"]} - {p1["piloto"]}\n' +
                  f'Trail braking: {p1["trail_braking"]:.1f}m | ' +
                  f'Dist. total sin acelerar: {p1["dist_sin_acelerar"]:.1f}m', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # ========== SUBPLOT PILOTO 2 ==========
    ax2 = fig.add_subplot(gs[idx, 1])
    
    ax2.plot(p2['distance_plot'], p2['speed_plot'], 
             'b-', linewidth=2.5, label='Velocidad', zorder=3)
    ax2.set_ylabel('Velocidad (km/h)', color='b', fontweight='bold', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='b')
    
    dist_plot2 = p2['distance_plot']
    
    ax2.plot(dist_plot2[p2['idx_brake_start']], 
             p2['speed_plot'][p2['idx_brake_start']], 
             'ro', markersize=12, markeredgewidth=2, 
             markeredgecolor='darkred', zorder=10, label='Inicio freno')
    
    ax2.plot(dist_plot2[p2['idx_apex']], 
             p2['speed_plot'][p2['idx_apex']], 
             'o', color='gold', markersize=12, markeredgewidth=2, 
             markeredgecolor='orange', zorder=10, label='Apex')
    
    ax2.plot(dist_plot2[p2['idx_brake_end']], 
             p2['speed_plot'][p2['idx_brake_end']], 
             'o', color='orange', markersize=10, markeredgewidth=2, 
             markeredgecolor='darkorange', zorder=10, label='Fin freno')
    
    ax2.plot(dist_plot2[p2['idx_throttle_full']], 
             p2['speed_plot'][p2['idx_throttle_full']], 
             'go', markersize=12, markeredgewidth=2, 
             markeredgecolor='darkgreen', zorder=10, label='100% gas')
    
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(dist_plot2, 0, p2['brake_plot'],
                          color='red', alpha=0.3, label='Freno %')
    ax2_twin.fill_between(dist_plot2, 0, p2['throttle_plot'],
                          color='green', alpha=0.2, label='Acelerador %')
    ax2_twin.set_ylabel('Freno / Acelerador (%)', fontweight='bold', fontsize=11)
    ax2_twin.set_ylim(0, 105)
    
    ax2.set_xlabel('Distancia desde inicio frenada (m)', fontweight='bold', fontsize=11)
    ax2.set_title(f'Curva {comp["curva_num"]} - {p2["piloto"]}\n' +
                  f'Trail braking: {p2["trail_braking"]:.1f}m | ' +
                  f'Dist. total sin acelerar: {p2["dist_sin_acelerar"]:.1f}m', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Título general
fig.suptitle(f'Analisis Completo de Tecnica de Frenada - {gp} GP {year} ({session_type})\n' +
             f'{pilotos_info[piloto1]} vs {pilotos_info[piloto2]}',
             fontsize=17, fontweight='bold', y=0.998)

# Guardar
filename = f'grafico_brake_technique/{gp}_{year}_{session_type}_frenadas_{piloto1}_vs_{piloto2}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nGrafico guardado: {filename}")

plt.show(block=True)

# ============================================
# TABLA RESUMEN COMPLETA
# ============================================

print("\n" + "=" * 70)
print("TABLA RESUMEN COMPLETA")
print("=" * 70)

for comp in comparaciones:
    p1 = comp['p1']
    p2 = comp['p2']
    
    print(f"\n{'='*70}")
    print(f"CURVA {comp['curva_num']}")
    print(f"{'='*70}")
    
    print(f"\n{'Metrica':<30} {piloto1:>12} {piloto2:>12} {'Diferencia':>12}")
    print("-" * 70)
    
    metricas = [
        ('Velocidad entrada (km/h)', 'velocidad_entrada'),
        ('Frenada maxima (%)', 'brake_max'),
        ('Velocidad apex (km/h)', 'velocidad_apex'),
        ('Velocidad salida (km/h)', 'velocidad_salida'),
        ('Dist. frenada apex (m)', 'dist_frenada_apex'),
        ('Dist. freno total (m)', 'dist_freno_total'),
        ('Dist. sin acelerar (m)', 'dist_sin_acelerar'),
        ('Trail braking (m)', 'trail_braking'),
    ]
    
    for nombre, key in metricas:
        val1 = p1[key]
        val2 = p2[key]
        diff = val1 - val2
        
        print(f"{nombre:<30} {val1:>12.1f} {val2:>12.1f} {diff:>+12.1f}")

print("\n" + "=" * 70)
print("LEYENDA DE MARCADORES EN GRAFICOS")
print("=" * 70)
print("  Punto ROJO    = Inicio de frenada")
print("  Punto AMARILLO = Apex (vertice de curva)")
print("  Punto NARANJA = Fin de frenada (suelta freno)")
print("  Punto VERDE   = Vuelta a 100% acelerador")
print("\n" + "=" * 70)
print("METRICAS EXPLICADAS")
print("=" * 70)
print("  Dist. frenada apex   = Desde inicio freno hasta apex")
print("  Dist. freno total    = Desde inicio freno hasta soltar freno")
print("  Dist. sin acelerar   = Desde inicio freno hasta 100% gas")
print("  Trail braking        = Distancia frenando DENTRO de la curva")
print("                         (despues del apex)")
print("\n" + "=" * 70)
print("ANALISIS COMPLETADO")
print("=" * 70)




### **1. Métricas completas:**

# Dist. frenada apex     = hasta vértice
# Dist. freno total      = hasta soltar freno 100%
# Dist. sin acelerar     = hasta volver a 100% throttle
# Trail braking          = frenando después del apex

### **2. Marcadores visuales en gráfico:**
#- **Punto rojo** = Inicio frenada
#- **Punto amarillo** = Apex
#- **Punto naranja** = Fin frenada
#- **Punto verde** = 100% throttle

### **3. Output detallado:**
#
#CURVA 1 (apex a 1250m)
#----------------------------------------------------------------------

#VER (Max Verstappen):
#  Velocidad entrada:       285.3 km/h
#  Frenada maxima:           98.5 %
#  Velocidad apex:          123.4 km/h
#  Velocidad salida:        198.7 km/h
#  Dist. frenada (apex):    142.3 m
#  freno total:       158.7 m
#  Dist. sin acelerar:      195.2 m
#  Trail braking:            16.4 m

#NOR (Lando Norris):
#  Velocidad entrada:       283.1 km/h
#  Frenada maxima:           95.2 %
#  Velocidad apex:          125.1 km/h
#  Velocidad salida:        196.3 km/h
#  Dist. frenada (apex):    138.7 m
#  Dist. freno total:       152.1 m
#  Dist. sin acelerar:      198.5 m
#  Trail braking:            13.4 m

#DIFERENCIAS (VER - NOR):
#  Vel. entrada:      +2.2 km/h
#  Vel. apex:         -1.7 km/h
#  Trail braking:     +3.0 m
#  Dist. frenada:     +3.6 m
