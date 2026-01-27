# 🚀 QUICK START GUIDE

## ⚡ 3 Steps to Launch

### 1️⃣ Copy Dashboard to Your Project
```bash
# Copy the entire dashboard/ folder to your F1-Telemetry-Analyzer directory
cp -r dashboard/ /path/to/F1-Telemetry-Analyzer/
```

**Your project structure should look like:**
```
F1-Telemetry-Analyzer/
├── dashboard/          ← NEW FOLDER
│   ├── app.py
│   ├── components/
│   └── utils/
├── scripts/           ← YOUR EXISTING SCRIPTS
│   ├── tyre_degradation_ml.py
│   ├── tyre_analysis_degradation_versus.py
│   ├── sector_analysis.py
│   ├── race_pace_analyzer.py
│   └── consistency_heatmap.py
├── cache/
└── output_*/
```

### 2️⃣ Install Streamlit
```bash
pip install streamlit
```

### 3️⃣ Launch Dashboard
```bash
cd F1-Telemetry-Analyzer
streamlit run dashboard/app.py
```

**That's it!** 🎉 Your browser will open automatically at `http://localhost:8501`

---

## 📱 First Time Using the Dashboard?

### Step-by-Step:

1. **Sidebar (Left)**: 
   - Select Year: `2024`
   - Select GP: `Spain`
   - Session: `Race`

2. **Main Area (Center)**:
   - Click on a tab (e.g., "🤖 ML Strategy Optimizer")

3. **Configuration**:
   - Select driver(s)
   - Click "Analyze" button

4. **Wait**:
   - First run: ~3-4 minutes (downloading data)
   - Later runs: 30-60 seconds (cached)

5. **View Results**:
   - Metrics at top
   - Full output in expandable section
   - Visualization displayed automatically

---

## 🎯 Try These First!

### Example 1: ML Strategy (Easy)
```
Tab: 🤖 ML Strategy Optimizer
Driver: VER
Train with all drivers: ✓
Train only with Race: ✓
Click: "🚀 Optimize Strategy"
```

### Example 2: Consistency (Fast)
```
Tab: 📊 Consistency Heatmap
Driver: VER
Session: Qualifying
Click: "📊 Analyze Consistency"
```

### Example 3: Sector Comparison (Visual)
```
Tab: 📍 Sector Analysis
Driver 1: VER
Driver 2: NOR
Sector Length: 100m
Click: "🔍 Analyze Sectors"
```

---

## ❓ Common Issues

### "Script not found"
**Problem**: Dashboard can't find your scripts
**Solution**: 
```bash
# Make sure you're in F1-Telemetry-Analyzer directory
pwd
# Should show: /path/to/F1-Telemetry-Analyzer

# Check scripts exist
ls scripts/
```

### "Module not found"
**Problem**: Missing dependencies
**Solution**:
```bash
pip install -r dashboard/requirements.txt
```

### "Analysis timeout"
**Problem**: Taking too long
**Solution**: Try a different GP/year, or wait a bit longer on first run

---

## 💡 Pro Tips

1. **First Run**: Start with Spain 2024 Race - it's well documented and fast
2. **Cache**: First analysis takes longer, subsequent are faster
3. **Tabs**: Each tab is independent - try them all!
4. **Images**: All visualizations are saved to `output_*` folders
5. **Keyboard**: Use `Ctrl+C` in terminal to stop dashboard

---

## 🎓 Need Help?

Check the full `README.md` in the dashboard folder for:
- Detailed feature explanations
- Configuration options
- Troubleshooting guide
- Advanced usage

---

**Enjoy analyzing F1 data!** [RACING][SPEED]