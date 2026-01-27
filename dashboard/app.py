import streamlit as st
import sys
import os

# Add both scripts and dashboard directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add dashboard directory
sys.path.append(os.path.join(current_dir, '..', 'scripts'))  # Add scripts directory

# Page config
st.set_page_config(
    page_title="F1 Telemetry Analyzer",
    page_icon=":)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E10600;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 4px;
        font-weight: 600;
        color: #262730;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E10600;
        color: white !important;
    }
    
    /* Button styling - MEJOR CONTRASTE */
    .stButton > button {
        width: 100%;
        background-color: #E10600;
        color: white !important;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
        border: none;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #B30500;
        color: white !important;
        box-shadow: 0 4px 12px rgba(225, 6, 0, 0.3);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Primary button specific */
    .stButton > button[kind="primary"] {
        background-color: #E10600;
        color: white !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #f0f2f6;
        border-left: 4px solid #E10600;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #262730;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
        color: #E10600;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">[RACING] F1 Telemetry Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Race Engineering Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title(" Configuration")
st.sidebar.markdown("---")

# Year selection
year = st.sidebar.selectbox(
    " Year",
    options=[2024, 2023, 2022, 2021, 2020],
    index=0,
    help="Select the Formula 1 season"
)

# Grand Prix selection
gp_list = [
    "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami",
    "Monaco", "Spain", "Canada", "Austria", "Britain", 
    "Hungary", "Belgium", "Netherlands", "Italy", "Singapore",
    "Japan", "Qatar", "United States", "Mexico", "Brazil",
    "Las Vegas", "Abu Dhabi"
]

gp = st.sidebar.selectbox(
    "Grand Prix",
    options=gp_list,
    index=6,  # Spain by default
    help="Select the Grand Prix"
)

# Driver list
drivers_list = [
    "VER", "PER", "LEC", "SAI", "NOR", "PIA",
    "HAM", "RUS", "ALO", "STR", "OCO", "GAS",
    "TSU", "RIC", "HUL", "MAG", "ALB", "SAR",
    "BOT", "ZHO"
]

# Session type
session_type = st.sidebar.selectbox(
    " Session Type",
    options=["Race", "Qualifying", "Practice 1", "Practice 2", "Practice 3"],
    index=0,
    help="Select the session type"
)

# Map to FastF1 session codes
session_map = {
    "Race": "R",
    "Qualifying": "Q",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3"
}
session_code = session_map[session_type]

st.sidebar.markdown("---")
st.sidebar.info(" **Tip:** Select parameters above, then choose an analysis tab below")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " ML Strategy Optimizer",
    " Tire Degradation vs",
    " Sector Analysis",
    " Race Pace Analyzer",
    " Consistency Heatmap"
])

# Import components
from components import ml_optimizer, tire_comparison, sector_analyzer, pace_analyzer, consistency

# Tab 1: ML Strategy Optimizer
with tab1:
    st.header(" ML Tire Strategy Optimizer")
    st.markdown("**Machine Learning based race strategy prediction with Random Forest**")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        driver_ml = st.selectbox(
            "Select Driver",
            options=drivers_list,
            key="driver_ml",
            help="Driver to analyze strategy for"
        )
    
    with col2:
        train_all = st.checkbox(
            "Train with all drivers",
            value=True,
            key="train_all_ml",
            help="Use data from all drivers (recommended)"
        )
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        race_only = st.checkbox(
            "Train ONLY with Race data",
            value=True,
            key="race_only_ml",
            help="More accurate predictions (recommended)"
        )
    
    if st.button(" Optimize Strategy", type="primary", key="btn_ml"):
        ml_optimizer.run_analysis(year, gp, driver_ml, train_all, race_only)

# Tab 2: Tire Degradation Comparison
with tab2:
    st.header(" Tire Degradation Comparison")
    st.markdown("**Compare tire degradation between two drivers**")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        driver1_tire = st.selectbox(
            "Driver 1",
            options=drivers_list,
            index=0,
            key="driver1_tire",
            help="First driver to compare"
        )
    
    with col2:
        driver2_tire = st.selectbox(
            "Driver 2",
            options=drivers_list,
            index=4,
            key="driver2_tire",
            help="Second driver to compare"
        )
    
    if driver1_tire == driver2_tire:
        st.error(" Please select different drivers")
    elif st.button(" Analyze Degradation", type="primary", key="btn_tire"):
        tire_comparison.run_analysis(year, gp, driver1_tire, driver2_tire)

# Tab 3: Sector Analysis
with tab3:
    st.header(" Sector-by-Sector Analysis")
    st.markdown("**Detailed mini-sector performance comparison**")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        driver1_sector = st.selectbox(
            "Driver 1",
            options=drivers_list,
            index=0,
            key="driver1_sector"
        )
    
    with col2:
        driver2_sector = st.selectbox(
            "Driver 2",
            options=drivers_list,
            index=4,
            key="driver2_sector"
        )
    
    with col3:
        sector_length = st.slider(
            "Sector Length (m)",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            key="sector_length",
            help="Length of mini-sectors for analysis"
        )
    
    if driver1_sector == driver2_sector:
        st.error(" Please select different drivers")
    elif st.button(" Analyze Sectors", type="primary", key="btn_sector"):
        sector_analyzer.run_analysis(year, gp, session_code, driver1_sector, driver2_sector, sector_length)

# Tab 4: Race Pace Analyzer
with tab4:
    st.header("⚡ Race Pace Analyzer")
    st.markdown("**Fuel-corrected pace comparison with undercut analysis**")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        driver1_pace = st.selectbox(
            "Driver 1",
            options=drivers_list,
            index=0,
            key="driver1_pace"
        )
    
    with col2:
        driver2_pace = st.selectbox(
            "Driver 2",
            options=drivers_list,
            index=4,
            key="driver2_pace"
        )
    
    with col3:
        fuel_effect = st.number_input(
            "Fuel Effect (s/kg)",
            min_value=0.020,
            max_value=0.050,
            value=0.035,
            step=0.005,
            key="fuel_effect",
            help="Time effect per kg of fuel (default: 0.035)"
        )
    
    st.info(" **Fuel Effect Guide:** Monaco/Singapore: 0.045 | Normal: 0.035 | Monza: 0.030")
    
    if driver1_pace == driver2_pace:
        st.error(" Please select different drivers")
    elif st.button(" Analyze Pace", type="primary", key="btn_pace"):
        pace_analyzer.run_analysis(year, gp, driver1_pace, driver2_pace, fuel_effect)

# Tab 5: Consistency Heatmap
with tab5:
    st.header(" Consistency Heatmap")
    st.markdown("**Visual lap quality analysis with purple sector detection**")
    st.markdown("---")
    
    driver_consistency = st.selectbox(
        "Select Driver",
        options=drivers_list,
        index=0,
        key="driver_consistency",
        help="Driver to analyze consistency"
    )
    
    st.info(" Purple = THE fastest |  Blue = <2% off |  Green = 2-5% off |  Yellow = 5-10% off |  Red = >10% off")
    
    if st.button(" Analyze Consistency", type="primary", key="btn_consistency"):
        consistency.run_analysis(year, gp, session_code, driver_consistency)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>[RACING] <strong>F1 Telemetry Analyzer</strong> | Professional Race Engineering Analysis</p>
        <p>Powered by FastF1 • Developed with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
