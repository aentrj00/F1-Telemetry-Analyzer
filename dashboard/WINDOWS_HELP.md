# 🪟 WINDOWS TROUBLESHOOTING

## [SUCCESS] SOLUCIONES RÁPIDAS PARA WINDOWS

### 1️⃣ Firewall de Windows

**Al ejecutar primera vez, Windows mostrará:**
```
"Firewall de Windows Defender bloqueó algunas características de Python"
```

**SOLUCIÓN:**
```
[SUCCESS] Marca: "Redes privadas, como las domésticas o del trabajo"
[SUCCESS] Click: "Permitir acceso"
```

**¿Por qué?** Streamlit abre un servidor web local (localhost:8501)

---

### 2️⃣ Error "ModuleNotFoundError: No module named 'components'"

**SOLUCIÓN RÁPIDA:**

Ejecuta desde el directorio correcto:
```bash
# Método 1: Entrar a la carpeta dashboard
cd F1-Telemetry-Analyzer\dashboard
streamlit run app.py

# Método 2: Desde raíz con ruta completa
cd F1-Telemetry-Analyzer
streamlit run dashboard\app.py
```

---

### 3️⃣ Error "No module named 'streamlit'"

**SOLUCIÓN:**
```bash
pip install streamlit
```

Si tienes múltiples versiones de Python:
```bash
python -m pip install streamlit
```

---

### 4️⃣ Puerto ya en uso

**Error:**
```
Address already in use
```

**SOLUCIÓN:**
```bash
# Usar otro puerto
streamlit run dashboard\app.py --server.port 8502
```

O cerrar Streamlit anterior (Ctrl+C en la terminal)

---

### 5️⃣ Scripts no encontrados

**Error:**
```
Script not found: tyre_degradation_ml.py
```

**SOLUCIÓN:**

Verifica estructura:
```
F1-Telemetry-Analyzer\
├── dashboard\
│   ├── app.py
│   └── components\
└── scripts\              ← IMPORTANTE: Deben estar aquí
    ├── tyre_degradation_ml.py
    ├── tyre_analysis_degradation_versus.py
    ├── sector_analysis.py
    ├── race_pace_analyzer.py
    └── consistency_heatmap.py
```

---

### 6️⃣ Python no reconocido

**Error:**
```
'python' no se reconoce como un comando interno o externo
```

**SOLUCIÓN:**

Usa `py` en lugar de `python`:
```bash
py -m pip install streamlit
py -m streamlit run dashboard\app.py
```

---

### 7️⃣ Permisos de ejecución

**Error:**
```
No se puede cargar porque la ejecución de scripts está deshabilitada
```

**SOLUCIÓN (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

O usa Command Prompt (cmd) en lugar de PowerShell

---

### 8️⃣ Terminal se cierra al ejecutar

**SOLUCIÓN:**

No hagas doble clic en el archivo .py

Ejecuta desde terminal:
```bash
# Abrir Command Prompt o PowerShell
cd F1-Telemetry-Analyzer
streamlit run dashboard\app.py
```

---

### 9️⃣ Cache de FastF1 muy grande

**SOLUCIÓN:**

Puedes borrar el cache sin problemas:
```bash
rmdir /s cache
```

Se regenerará automáticamente

---

### 🔟 matplotlib backend error

**Error:**
```
backend TkAgg not available
```

**SOLUCIÓN:**
```bash
pip install tk
```

---

## 🚀 COMANDOS ÚTILES WINDOWS

### Verificar instalación:
```bash
python --version
pip --version
streamlit --version
```

### Listar paquetes instalados:
```bash
pip list
```

### Actualizar pip:
```bash
python -m pip install --upgrade pip
```

### Ver puertos en uso:
```bash
netstat -ano | findstr :8501
```

---

## 📝 TIPS PARA WINDOWS

### 1. Usa el directorio correcto
```bash
# Siempre verifica dónde estás
cd

# Ve al proyecto
cd C:\Users\TuUsuario\Desktop\F1-Telemetry-Analyzer
```

### 2. Barras en Windows
```bash
# Windows usa \ (backslash)
dashboard\app.py

# NO usar / (forward slash) en cmd
```

### 3. Espacios en rutas
```bash
# Si la ruta tiene espacios, usa comillas
cd "C:\Users\Tu Nombre\Desktop\F1-Telemetry-Analyzer"
```

### 4. Terminal recomendado
```
[SUCCESS] Command Prompt (cmd)
[SUCCESS] PowerShell
[SUCCESS] Windows Terminal
[ERROR] Git Bash (puede dar problemas con rutas)
```

---

## 🎯 CHECKLIST PRE-EJECUCIÓN

Antes de ejecutar `streamlit run dashboard\app.py`:

- [ ] Estás en el directorio `F1-Telemetry-Analyzer`
- [ ] Existe la carpeta `dashboard\`
- [ ] Existe la carpeta `scripts\` con los 5 scripts
- [ ] Streamlit está instalado (`pip list | findstr streamlit`)
- [ ] Python versión 3.8+ (`python --version`)
- [ ] Puerto 8501 libre

---

## ❓ ¿SIGUE SIN FUNCIONAR?

### Prueba esto paso a paso:

```bash
# 1. Ve al directorio
cd F1-Telemetry-Analyzer

# 2. Verifica estructura
dir
dir dashboard
dir scripts

# 3. Activa entorno virtual (si tienes)
venv\Scripts\activate

# 4. Instala dependencias
pip install -r dashboard\requirements.txt

# 5. Ejecuta
streamlit run dashboard\app.py

# 6. Permite en Firewall cuando pregunte

# 7. Abre navegador en: http://localhost:8501
```

---

**Si nada funciona:** Pega el error completo y lo revisamos 🔍
