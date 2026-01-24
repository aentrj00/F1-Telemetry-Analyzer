# F1 Telemetry Analyzer 

Sistema de análisis de telemetría de Fórmula 1 para estudiar rendimiento de pilotos, técnicas de pilotaje y estrategias de carrera.

##  Características

### Scripts Disponibles

#### 1. Comparador de Circuitos (`analisisCompleto_con_comparacion.py`)
- Visualización del circuito con velocidades
- Puntos de frenada marcados
- Comparación visual entre 2 pilotos
- Métricas de velocidad máxima/mínima

#### 2. Evolución de Posiciones (`posiciones_carrera.py`)
- Gráfico de posiciones vuelta a vuelta
- Detección de pit stops
- Análisis de undercut/overcut
- Top 10 pilotos

#### 3. Tiempos de Vuelta (`tiempos_sesion.py`)
- Evolución temporal de tiempos
- Identificación de compuestos de neumáticos
- Análisis de cualquier sesión (FP, Q, R)
- Detección de vuelta más rápida

#### 4. Análisis de Frenadas (`brake_technique.py`)
- Detección automática de curvas principales
- Comparación de técnica de frenada
- Métricas avanzadas:
  - Trail braking
  - Distancia de frenada
  - Velocidades de entrada/apex/salida
- Visualización con marcadores de puntos clave

##  Instalación
```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/F1-Telemetry-Analyzer.git
cd F1-Telemetry-Analyzer

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

##  Uso
```bash
# Ejemplo: Analizar frenadas
python scripts/analisis_frenadas_completo.py

# Seguir instrucciones interactivas
Año: 2024
Gran Premio: Bahrain
Tipo de sesión: Q
Piloto 1: VER
Piloto 2: NOR
```

##  Tecnologías

- **FastF1**: API de datos de F1
- **Matplotlib**: Visualizaciones
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **Scikit-learn**: Machine Learning (próximamente)

##  Estructura
```
F1-Telemetry-Analyzer/
├── scripts_pruebas/              # Scripts for testing
├── cache/                        # Cache of F1 data (auto-generated)
└── grafico.../                   # Graphs (auto-generated)
```



