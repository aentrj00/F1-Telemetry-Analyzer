import fastf1 as ff1
import matplotlib.pyplot as plt

# Habilitar cache (para no descargar los datos cada vez)
ff1.Cache.enable_cache('cache')

print("Descargando datos de F1 2024 - Bahrain GP...")

# Cargar sesión
session = ff1.get_session(2024, 'Bahrain', 'R')
session.load()

print(" Datos descargados!")

# Obtener vuelta más rápida de Verstappen
ver_lap = session.laps.pick_driver('VER').pick_fastest()
telemetry = ver_lap.get_telemetry()

# Crear gráfico
plt.figure(figsize=(12, 6))
plt.plot(telemetry['Distance'], telemetry['Speed'], color='red', linewidth=2)
plt.xlabel('Distancia (metros)', fontsize=12)
plt.ylabel('Velocidad (km/h)', fontsize=12)
plt.title('Vuelta más rápida - Max Verstappen - Bahrain 2024', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('verstappen_bahrain.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Gráfico guardado como 'verstappen_bahrain.png'")