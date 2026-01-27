# [RACING] F1 Telemetry Analyzer Dashboard

Professional web-based dashboard for F1 race engineering analysis.

## 📋 Features

### 🤖 ML Strategy Optimizer
- Machine Learning based tire strategy prediction
- Random Forest model with 3.8% prediction accuracy
- Tests all compound combinations (SOFT/MEDIUM/HARD)
- Evaluates 1-STOP, 2-STOP, 3-STOP strategies
- Multi-session training for better accuracy

### 🔄 Tire Degradation Comparison
- Compare tire wear between two drivers
- Stint-by-stint degradation analysis
- Compound-specific performance tracking
- Visual degradation curves

### 📍 Sector Analysis
- Mini-sector performance breakdown
- Circuit heatmap visualization
- Cumulative delta tracking
- Corner-by-corner comparison
- Configurable sector length (50-500m)

### ⚡ Race Pace Analyzer
- Fuel-corrected lap time analysis
- Undercut/overcut potential calculation
- Stint-by-stint pace comparison
- Consistency metrics (standard deviation)
- Configurable fuel effect parameter

### 📊 Consistency Heatmap
- Visual lap quality grid
- Purple sector detection (THE fastest)
- Color-coded performance (Purple/Blue/Green/Yellow/Red)
- Qualifying progression analysis
- Consistency rating system

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
```

### Setup
```bash
# 1. Navigate to project directory
cd F1-Telemetry-Analyzer

# 2. Install dependencies (if not already done)
pip install -r requirements.txt

# 3. Install Streamlit
pip install streamlit

# 4. Verify installation
streamlit --version
```

## 🎮 Usage

### Launch Dashboard
```bash
# From project root directory
streamlit run dashboard/app.py

# Or with custom port
streamlit run dashboard/app.py --server.port 8502
```

The dashboard will automatically open in your default browser at `http://localhost:8501`

### Using the Dashboard

1. **Select Parameters (Sidebar)**
   - Year: 2020-2024
   - Grand Prix: Select from calendar
   - Session Type: Race, Qualifying, or Practice

2. **Choose Analysis Tab**
   - 🤖 ML Strategy Optimizer
   - 🔄 Tire Degradation vs
   - 📍 Sector Analysis
   - ⚡ Race Pace Analyzer
   - 📊 Consistency Heatmap

3. **Configure Analysis**
   - Select driver(s)
   - Adjust parameters (if applicable)
   - Click "Analyze" button

4. **View Results**
   - Key metrics displayed at top
   - Full analysis output in expandable section
   - Visualizations shown automatically
   - Images saved to respective output folders

## 📁 Project Structure

```
F1-Telemetry-Analyzer/
├── dashboard/
│   ├── app.py                 # Main dashboard application
│   ├── components/            # Analysis modules
│   │   ├── ml_optimizer.py
│   │   ├── tire_comparison.py
│   │   ├── sector_analyzer.py
│   │   ├── pace_analyzer.py
│   │   └── consistency.py
│   └── utils/                 # Helper functions
│
├── scripts/                   # Original analysis scripts
│   ├── tyre_degradation_ml.py
│   ├── tyre_analysis_degradation_versus.py
│   ├── sector_analysis.py
│   ├── race_pace_analyzer.py
│   └── consistency_heatmap.py
│
├── cache/                     # FastF1 cache directory
├── output_*/                  # Generated visualizations
└── requirements.txt
```

## ⚙️ Configuration

### Fuel Effect (Race Pace Analyzer)

| Circuit Type | Recommended Value |
|-------------|-------------------|
| Monaco, Singapore, Hungary | 0.045 s/kg |
| Normal circuits | 0.035 s/kg (default) |
| Monza, Spa, Silverstone | 0.030 s/kg |
| Mexico, Brazil | 0.028 s/kg |

### Sector Length (Sector Analysis)

| Length | Use Case |
|--------|----------|
| 50-100m | Very detailed analysis |
| 150-200m | Balanced detail (recommended) |
| 300-500m | Overview analysis |

## 🎯 Tips & Best Practices

### ML Strategy Optimizer
- [SUCCESS] Use "Train with all drivers" for more data
- [SUCCESS] Use "Race only" for better accuracy
- ⏱️ First run takes ~3-4 minutes (downloads data)
- 📊 Subsequent runs are faster (cached)

### Tire Degradation Comparison
- 👥 Compare drivers with different strategies
- 🔄 Use Race session for best results
- 📈 Look for crossover points in degradation

### Sector Analysis
- 🏁 Use Qualifying for clean lap comparison
- 📏 Use 100-150m sectors for best detail/clarity balance
- 🎯 Focus on significant corners (delta > 0.03s)

### Race Pace Analyzer
- ⛽ Adjust fuel effect based on circuit type
- 📊 Compare similar stints for fairness
- 🔍 Look at fuel-corrected pace, not raw times

### Consistency Heatmap
- 🟣 Only ONE purple lap (the fastest)
- 🔵 Blue laps are very close (<2% off)
- 📊 Consistency rating based on std deviation
- ✨ Use for Qualifying progression analysis

## ❗ Troubleshooting

### "Script not found" error
```bash
# Verify scripts location
ls scripts/

# Scripts should be in ../scripts/ relative to dashboard/
```

### Analysis timeout
```bash
# Increase timeout in component files
# Or try different GP/year combination
```

### No visualization displayed
```bash
# Check output directories
ls output_tire_ml/
ls output_consistency/
# etc.

# Images should be generated automatically
```

### Cache issues
```bash
# Clear FastF1 cache
rm -rf cache/

# Let dashboard rebuild cache on next run
```

## 🔧 Advanced Usage

### Custom Port
```bash
streamlit run dashboard/app.py --server.port 8080
```

### Network Access
```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

### Headless Mode
```bash
streamlit run dashboard/app.py --server.headless true
```

## 📊 Performance

- **ML Optimizer**: 2-4 minutes first run, 30-60s cached
- **Tire Comparison**: 30-60 seconds
- **Sector Analysis**: 60-90 seconds
- **Race Pace**: 60-90 seconds
- **Consistency**: 30-60 seconds

## 🎓 Academic Use

This dashboard is perfect for:
- 📚 University projects (TFG/TFM)
- 🎤 Presentations and demos
- 💼 Portfolio demonstrations
- 🔬 Research and analysis

## 📝 License

Part of F1-Telemetry-Analyzer project.

## 🙋 Support

For issues or questions:
1. Check troubleshooting section
2. Verify scripts are in correct location
3. Ensure all dependencies installed
4. Check FastF1 cache status

## 🚀 Future Enhancements

Potential additions:
- [ ] Real-time analysis during live sessions
- [ ] Multi-driver comparison (3+ drivers)
- [ ] Export reports to PDF
- [ ] Cloud deployment (Streamlit Cloud)
- [ ] Database for historical analysis
- [ ] Advanced filtering options

---

**Built with** ❤️ **for F1 race engineering analysis**
