import fastf1 as ff1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('output_tire_ml'):
    os.makedirs('output_tire_ml')

ff1.Cache.enable_cache('cache')

# ============================================
# CIRCUIT CHARACTERISTICS DATABASE
# ============================================

CIRCUIT_DATA = {
    'Bahrain': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 15, 'length_km': 5.412, 'typical_weather': 'hot_dry', 'surface': 'abrasive'},
    'Jeddah': {'type': 'street', 'avg_speed': 'high', 'corners': 27, 'length_km': 6.174, 'typical_weather': 'hot_dry', 'surface': 'smooth'},
    'Australia': {'type': 'semi-permanent', 'avg_speed': 'medium', 'corners': 14, 'length_km': 5.278, 'typical_weather': 'temperate', 'surface': 'medium'},
    'Baku': {'type': 'street', 'avg_speed': 'high', 'corners': 20, 'length_km': 6.003, 'typical_weather': 'temperate', 'surface': 'smooth'},
    'Miami': {'type': 'street', 'avg_speed': 'medium', 'corners': 19, 'length_km': 5.412, 'typical_weather': 'hot_humid', 'surface': 'medium'},
    'Monaco': {'type': 'street', 'avg_speed': 'low', 'corners': 19, 'length_km': 3.337, 'typical_weather': 'temperate', 'surface': 'smooth'},
    'Spain': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 16, 'length_km': 4.675, 'typical_weather': 'hot_dry', 'surface': 'abrasive'},
    'Canada': {'type': 'semi-permanent', 'avg_speed': 'high', 'corners': 14, 'length_km': 4.361, 'typical_weather': 'variable', 'surface': 'smooth'},
    'Austria': {'type': 'permanent', 'avg_speed': 'high', 'corners': 10, 'length_km': 4.318, 'typical_weather': 'temperate', 'surface': 'medium'},
    'Britain': {'type': 'permanent', 'avg_speed': 'high', 'corners': 18, 'length_km': 5.891, 'typical_weather': 'variable', 'surface': 'medium'},
    'Hungary': {'type': 'permanent', 'avg_speed': 'low', 'corners': 14, 'length_km': 4.381, 'typical_weather': 'hot_dry', 'surface': 'abrasive'},
    'Belgium': {'type': 'permanent', 'avg_speed': 'high', 'corners': 19, 'length_km': 7.004, 'typical_weather': 'variable', 'surface': 'medium'},
    'Netherlands': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 14, 'length_km': 4.259, 'typical_weather': 'cool', 'surface': 'medium'},
    'Italy': {'type': 'permanent', 'avg_speed': 'high', 'corners': 11, 'length_km': 5.793, 'typical_weather': 'hot_dry', 'surface': 'smooth'},
    'Singapore': {'type': 'street', 'avg_speed': 'low', 'corners': 23, 'length_km': 5.063, 'typical_weather': 'hot_humid', 'surface': 'smooth'},
    'Japan': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 18, 'length_km': 5.807, 'typical_weather': 'temperate', 'surface': 'medium'},
    'Qatar': {'type': 'permanent', 'avg_speed': 'high', 'corners': 16, 'length_km': 5.380, 'typical_weather': 'hot_dry', 'surface': 'abrasive'},
    'United States': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 20, 'length_km': 5.513, 'typical_weather': 'hot_dry', 'surface': 'abrasive'},
    'Mexico': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 17, 'length_km': 4.304, 'typical_weather': 'temperate', 'surface': 'smooth'},
    'Brazil': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 15, 'length_km': 4.309, 'typical_weather': 'variable', 'surface': 'abrasive'},
    'Las Vegas': {'type': 'street', 'avg_speed': 'high', 'corners': 17, 'length_km': 6.120, 'typical_weather': 'cool', 'surface': 'smooth'},
    'Abu Dhabi': {'type': 'permanent', 'avg_speed': 'medium', 'corners': 16, 'length_km': 5.281, 'typical_weather': 'hot_dry', 'surface': 'smooth'}
}

def encode_circuit_features(gp_name):
    circuit = None
    for key in CIRCUIT_DATA.keys():
        if key.lower() in gp_name.lower():
            circuit = CIRCUIT_DATA[key]
            break
    
    if circuit is None:
        circuit = {'type': 'permanent', 'avg_speed': 'medium', 'corners': 15, 'length_km': 5.0, 'typical_weather': 'temperate', 'surface': 'medium'}
    
    type_encoding = {'street': 0, 'semi-permanent': 1, 'permanent': 2}
    speed_encoding = {'low': 0, 'medium': 1, 'high': 2}
    weather_encoding = {'cool': 0, 'temperate': 1, 'hot_dry': 2, 'hot_humid': 3, 'variable': 4, 'wet': 5}
    surface_encoding = {'smooth': 0, 'medium': 1, 'abrasive': 2}
    
    return {
        'circuit_type': type_encoding.get(circuit['type'], 2),
        'circuit_speed': speed_encoding.get(circuit['avg_speed'], 1),
        'circuit_corners': circuit['corners'],
        'circuit_length': circuit['length_km'],
        'circuit_weather_type': weather_encoding.get(circuit['typical_weather'], 1),
        'circuit_surface': surface_encoding.get(circuit['surface'], 1)
    }

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_all_sessions(year, gp):
    sessions = []
    session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']
    
    print(f"\nLoading all sessions for {gp} GP {year}...")
    
    for session_type in session_types:
        try:
            session = ff1.get_session(int(year), gp, session_type)
            session.load()
            sessions.append({'type': session_type, 'session': session})
            print(f"  [SUCCESS] {session_type} loaded")
        except Exception as e:
            print(f"  [ERROR] {session_type} not available: {str(e)[:50]}")
    
    return sessions

def detect_stints(laps):
    stints = []
    current_stint = []
    
    for idx, lap in laps.iterrows():
        if len(current_stint) == 0:
            current_stint = [lap]
        elif pd.notna(lap['PitInTime']):
            if len(current_stint) > 3:
                stints.append(pd.DataFrame(current_stint))
            current_stint = []
        else:
            current_stint.append(lap)
    
    if len(current_stint) > 3:
        stints.append(pd.DataFrame(current_stint))
    
    return stints

def calculate_pit_stop_time(laps):
    pit_times = []
    
    for idx, lap in laps.iterrows():
        if pd.notna(lap['PitInTime']) and pd.notna(lap['PitOutTime']):
            pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
            if 15 < pit_duration < 60:
                pit_times.append(pit_duration)
    
    return np.mean(pit_times) if pit_times else 23.0

def prepare_ml_features_from_sessions(sessions_data, gp_name, target_driver=None, sessions_to_use=None):
    """
    Prepares ML features from multiple sessions
    
    Args:
        sessions_data: List of session dictionaries
        gp_name: Grand Prix name
        target_driver: If specified, only use this driver's data
        sessions_to_use: List of session types to use (e.g., ['R'] for race only)
                        If None, uses all sessions
    """
    all_features = []
    circuit_features = encode_circuit_features(gp_name)
    
    for session_data in sessions_data:
        session = session_data['session']
        session_type = session_data['type']
        
        # Filter by session type if specified
        if sessions_to_use is not None and session_type not in sessions_to_use:
            print(f"\n  Skipping {session_type} (not in filter)")
            continue
        
        print(f"\n  Processing {session_type}...")
        
        if target_driver:
            try:
                laps = session.laps.pick_drivers(target_driver)
            except:
                print(f"    [ERROR] Driver {target_driver} not found in {session_type}")
                continue
        else:
            laps = session.laps
        
        laps = laps[laps['LapTime'].notna()]
        
        if len(laps) == 0:
            print(f"    [ERROR] No valid laps in {session_type}")
            continue
        
        print(f"    -> {len(laps)} valid laps")
        
        weather_data = session.weather_data
        total_laps = laps['LapNumber'].max() if session_type == 'R' else 60
        
        for idx, lap in laps.iterrows():
            lap_number = lap['LapNumber']
            lap_time = lap['LapTime']
            
            if pd.isna(lap_time):
                continue
            
            lap_time_seconds = lap_time.total_seconds()
            
            if lap_time_seconds > 200:
                continue
            
            lap_start_time = lap['LapStartTime']
            
            try:
                weather_at_lap = weather_data[weather_data['Time'] <= lap_start_time].iloc[-1]
                track_temp = weather_at_lap['TrackTemp']
                air_temp = weather_at_lap['AirTemp']
                try:
                    wind_speed = weather_at_lap['WindSpeed']
                    if pd.isna(wind_speed):
                        wind_speed = 0.0
                except:
                    wind_speed = 0.0
            except:
                track_temp = 30.0
                air_temp = 25.0
                wind_speed = 0.0
            
            try:
                compound = lap['Compound']
                if pd.isna(compound):
                    compound = 'MEDIUM'
            except:
                compound = 'MEDIUM'
            
            compound_encoded = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}.get(compound, 1)
            
            try:
                tire_life = lap['TyreLife']
                if pd.isna(tire_life):
                    tire_life = 1
            except:
                tire_life = 1
            
            fuel_load = 110 - (lap_number / total_laps) * 110 if session_type == 'R' else 10.0
            track_evolution = lap_number / total_laps
            session_encoding = {'FP1': 0, 'FP2': 1, 'FP3': 2, 'Q': 3, 'R': 4}.get(session_type, 4)
            
            all_features.append({
                'tire_age': tire_life,
                'compound': compound_encoded,
                'track_temp': track_temp,
                'air_temp': air_temp,
                'wind_speed': wind_speed,
                'fuel_load': fuel_load,
                'track_evolution': track_evolution,
                'session_type': session_encoding,
                **circuit_features,
                'lap_time': lap_time_seconds,
                'compound_name': compound,
                'session_name': session_type
            })
    
    df = pd.DataFrame(all_features)
    
    if len(df) == 0:
        return None, None
    
    mean_time = df['lap_time'].mean()
    std_time = df['lap_time'].std()
    df['outlier'] = np.abs(df['lap_time'] - mean_time) > 2.5 * std_time
    
    df_clean = df[~df['outlier']].copy()
    
    print(f"\n  Total samples: {len(df)}")
    print(f"  After removing outliers: {len(df_clean)}")
    
    return df_clean, df

def train_ml_model(features_df):
    feature_columns = [
        'tire_age', 'compound', 
        'track_temp', 'air_temp', 'wind_speed',
        'fuel_load', 'track_evolution', 'session_type',
        'circuit_type', 'circuit_speed', 'circuit_corners', 
        'circuit_length', 'circuit_weather_type', 'circuit_surface'
    ]
    
    X = features_df[feature_columns]
    y = features_df['lap_time']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'feature_importance': feature_importance
    }
    
    return model, metrics, feature_columns

def optimize_race_strategy(model, feature_columns, avg_conditions, circuit_features, 
                          total_race_laps, pit_stop_time, max_tire_age_seen):
    """
    Optimizes race strategy - Tests ALL compound combinations
    Includes 1-STOP, 2-STOP, and 3-STOP strategies
    
    F1 Rules enforced:
    - Minimum 1 pit stop required
    - Must use at least 2 different compounds
    """
    strategies = []
    
    # CRITICAL: Limit stint length to what we've actually seen in data
    max_stint_length = min(int(max_tire_age_seen * 0.95), int(total_race_laps * 0.7))
    
    print(f"\n  [!] IMPORTANT: Limiting strategies to tire_age <= {max_stint_length} laps")
    print(f"    (Max tire_age in training data: {max_tire_age_seen} laps)")
    print(f"    Model cannot reliably predict beyond this range")
    print(f"    Testing ALL compound combinations (1-STOP, 2-STOP, 3-STOP)...\n")
    
    # Compound definitions
    compounds = {
        'SOFT': 0,
        'MEDIUM': 1,
        'HARD': 2
    }
    
    # ============================================
    # HELPER FUNCTION: Simulate stint
    # ============================================
    def simulate_stint(stint_laps, compound_code, start_lap):
        """Simulates a single stint and returns total time"""
        time = 0
        for lap_in_stint in range(1, stint_laps + 1):
            race_lap = start_lap + lap_in_stint - 1
            features = {
                'tire_age': lap_in_stint,
                'compound': compound_code,
                'track_temp': avg_conditions['track_temp'],
                'air_temp': avg_conditions['air_temp'],
                'wind_speed': avg_conditions['wind_speed'],
                'fuel_load': 110 - (race_lap / total_race_laps) * 110,
                'track_evolution': race_lap / total_race_laps,
                'session_type': 4,
                **circuit_features
            }
            features_array = np.array([[features[col] for col in feature_columns]])
            time += model.predict(features_array)[0]
        return time
    
    # ============================================
    # 1-STOP STRATEGIES
    # ============================================
    # Valid compound pairs (must be different per F1 rules)
    compound_pairs_1stop = [
        ('SOFT', 'MEDIUM'),
        ('SOFT', 'HARD'),
        ('MEDIUM', 'HARD')
    ]
    
    if max_stint_length >= total_race_laps * 0.4:
        for compound1_name, compound2_name in compound_pairs_1stop:
            compound1 = compounds[compound1_name]
            compound2 = compounds[compound2_name]
            
            for pit_lap in range(int(total_race_laps * 0.3), min(max_stint_length, int(total_race_laps * 0.7)), 2):
                stint2_laps = total_race_laps - pit_lap
                
                # Check stint lengths
                if stint2_laps > max_stint_length or pit_lap > max_stint_length:
                    continue
                
                # Simulate stints
                total_time = 0
                total_time += simulate_stint(pit_lap, compound1, 1)
                total_time += pit_stop_time
                total_time += simulate_stint(stint2_laps, compound2, pit_lap + 1)
                
                strategies.append({
                    'type': '1-STOP',
                    'pit_laps': [pit_lap],
                    'stints': [
                        {'laps': pit_lap, 'compound': compound1_name},
                        {'laps': stint2_laps, 'compound': compound2_name}
                    ],
                    'total_time': total_time,
                    'num_pit_stops': 1
                })
    
    # ============================================
    # 2-STOP STRATEGIES
    # ============================================
    # Valid compound triplets (must use at least 2 different)
    compound_triplets_2stop = [
        ('SOFT', 'MEDIUM', 'HARD'),
        ('SOFT', 'HARD', 'HARD'),
        ('SOFT', 'MEDIUM', 'MEDIUM'),
        ('MEDIUM', 'SOFT', 'HARD'),
        ('MEDIUM', 'HARD', 'HARD'),
        ('MEDIUM', 'SOFT', 'SOFT'),
        ('HARD', 'SOFT', 'SOFT'),
        ('HARD', 'MEDIUM', 'MEDIUM'),
        ('HARD', 'SOFT', 'MEDIUM')
    ]
    
    pit1_range = range(int(total_race_laps * 0.20), min(max_stint_length, int(total_race_laps * 0.40)), 3)
    pit2_range = range(int(total_race_laps * 0.45), min(max_stint_length + 15, int(total_race_laps * 0.75)), 3)
    
    for compound1_name, compound2_name, compound3_name in compound_triplets_2stop:
        compound1 = compounds[compound1_name]
        compound2 = compounds[compound2_name]
        compound3 = compounds[compound3_name]
        
        for pit1 in pit1_range:
            for pit2 in pit2_range:
                # Minimum stint length: 10 laps
                if pit2 <= pit1 + 10:
                    continue
                
                stint1_len = pit1
                stint2_len = pit2 - pit1
                stint3_len = total_race_laps - pit2
                
                # Check all stints within limits
                if (stint1_len > max_stint_length or 
                    stint2_len > max_stint_length or 
                    stint3_len > max_stint_length or
                    stint3_len < 10):
                    continue
                
                # Simulate stints
                total_time = 0
                total_time += simulate_stint(stint1_len, compound1, 1)
                total_time += pit_stop_time
                total_time += simulate_stint(stint2_len, compound2, pit1 + 1)
                total_time += pit_stop_time
                total_time += simulate_stint(stint3_len, compound3, pit2 + 1)
                
                strategies.append({
                    'type': '2-STOP',
                    'pit_laps': [pit1, pit2],
                    'stints': [
                        {'laps': stint1_len, 'compound': compound1_name},
                        {'laps': stint2_len, 'compound': compound2_name},
                        {'laps': stint3_len, 'compound': compound3_name}
                    ],
                    'total_time': total_time,
                    'num_pit_stops': 2
                })
    
    # ============================================
    # 3-STOP STRATEGIES
    # ============================================
    # Valid compound quartets (must use at least 2 different)
    compound_quartets_3stop = [
        ('SOFT', 'SOFT', 'MEDIUM', 'HARD'),
        ('SOFT', 'MEDIUM', 'MEDIUM', 'HARD'),
        ('SOFT', 'MEDIUM', 'HARD', 'HARD'),
        ('MEDIUM', 'MEDIUM', 'HARD', 'HARD'),
        ('SOFT', 'SOFT', 'SOFT', 'HARD'),
        ('SOFT', 'SOFT', 'HARD', 'HARD'),
        ('MEDIUM', 'SOFT', 'SOFT', 'HARD'),
        ('HARD', 'SOFT', 'SOFT', 'SOFT'),
        ('HARD', 'MEDIUM', 'MEDIUM', 'MEDIUM')
    ]
    
    # 3-stop only makes sense with shorter stints
    pit1_range_3 = range(int(total_race_laps * 0.15), min(max_stint_length, int(total_race_laps * 0.30)), 4)
    pit2_range_3 = range(int(total_race_laps * 0.35), min(max_stint_length, int(total_race_laps * 0.55)), 4)
    pit3_range_3 = range(int(total_race_laps * 0.60), min(max_stint_length, int(total_race_laps * 0.80)), 4)
    
    for compound1_name, compound2_name, compound3_name, compound4_name in compound_quartets_3stop:
        compound1 = compounds[compound1_name]
        compound2 = compounds[compound2_name]
        compound3 = compounds[compound3_name]
        compound4 = compounds[compound4_name]
        
        for pit1 in pit1_range_3:
            for pit2 in pit2_range_3:
                for pit3 in pit3_range_3:
                    # Minimum stint length: 8 laps
                    if pit2 <= pit1 + 8 or pit3 <= pit2 + 8:
                        continue
                    
                    stint1_len = pit1
                    stint2_len = pit2 - pit1
                    stint3_len = pit3 - pit2
                    stint4_len = total_race_laps - pit3
                    
                    # Check all stints within limits
                    if (stint1_len > max_stint_length or 
                        stint2_len > max_stint_length or 
                        stint3_len > max_stint_length or
                        stint4_len > max_stint_length or
                        stint4_len < 8):
                        continue
                    
                    # Simulate stints
                    total_time = 0
                    total_time += simulate_stint(stint1_len, compound1, 1)
                    total_time += pit_stop_time
                    total_time += simulate_stint(stint2_len, compound2, pit1 + 1)
                    total_time += pit_stop_time
                    total_time += simulate_stint(stint3_len, compound3, pit2 + 1)
                    total_time += pit_stop_time
                    total_time += simulate_stint(stint4_len, compound4, pit3 + 1)
                    
                    strategies.append({
                        'type': '3-STOP',
                        'pit_laps': [pit1, pit2, pit3],
                        'stints': [
                            {'laps': stint1_len, 'compound': compound1_name},
                            {'laps': stint2_len, 'compound': compound2_name},
                            {'laps': stint3_len, 'compound': compound3_name},
                            {'laps': stint4_len, 'compound': compound4_name}
                        ],
                        'total_time': total_time,
                        'num_pit_stops': 3
                    })
    
    # ============================================
    # RESULTS
    # ============================================
    if len(strategies) == 0:
        print("\n  [!] WARNING: No valid strategies found within tire_age limits!")
        print("    Model needs more data with longer stints to optimize strategy.")
        return None
    
    best_strategy = min(strategies, key=lambda x: x['total_time'])
    
    # Get top 5 overall and top 3 per category
    top_strategies_overall = sorted(strategies, key=lambda x: x['total_time'])[:10]
    
    strategies_1stop = [s for s in strategies if s['type'] == '1-STOP']
    strategies_2stop = [s for s in strategies if s['type'] == '2-STOP']
    strategies_3stop = [s for s in strategies if s['type'] == '3-STOP']
    
    print(f"  [SUCCESS] Tested {len(strategies)} strategy combinations:")
    print(f"      - 1-STOP: {len(strategies_1stop)} strategies")
    print(f"      - 2-STOP: {len(strategies_2stop)} strategies")
    print(f"      - 3-STOP: {len(strategies_3stop)} strategies")
    
    return {
        'best_strategy': best_strategy,
        'all_strategies': top_strategies_overall,
        'by_type': {
            '1-STOP': sorted(strategies_1stop, key=lambda x: x['total_time'])[:3] if strategies_1stop else [],
            '2-STOP': sorted(strategies_2stop, key=lambda x: x['total_time'])[:3] if strategies_2stop else [],
            '3-STOP': sorted(strategies_3stop, key=lambda x: x['total_time'])[:3] if strategies_3stop else []
        }
    }

def simulate_actual_strategy(stints, model, feature_columns, avg_conditions, 
                            circuit_features, total_race_laps, pit_stop_time):
    total_time = 0
    actual_stints_info = []
    
    for stint_idx, stint in enumerate(stints):
        stint_laps = len(stint)
        
        try:
            compound_name = stint.iloc[0]['Compound']
            if pd.isna(compound_name):
                compound_name = 'MEDIUM'
        except:
            compound_name = 'MEDIUM'
        
        compound_code = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}.get(compound_name, 1)
        
        try:
            start_lap = stint.iloc[0]['LapNumber']
        except:
            start_lap = sum([len(s) for s in stints[:stint_idx]]) + stint_idx + 1
        
        for lap_in_stint in range(1, stint_laps + 1):
            race_lap = start_lap + lap_in_stint - 1
            
            features = {
                'tire_age': lap_in_stint,
                'compound': compound_code,
                'track_temp': avg_conditions['track_temp'],
                'air_temp': avg_conditions['air_temp'],
                'wind_speed': avg_conditions['wind_speed'],
                'fuel_load': 110 - (race_lap / total_race_laps) * 110,
                'track_evolution': race_lap / total_race_laps,
                'session_type': 4,
                **circuit_features
            }
            
            features_array = np.array([[features[col] for col in feature_columns]])
            lap_time = model.predict(features_array)[0]
            total_time += lap_time
        
        if stint_idx < len(stints) - 1:
            total_time += pit_stop_time
        
        actual_stints_info.append({
            'stint_number': stint_idx + 1,
            'laps': stint_laps,
            'compound': compound_name,
            'start_lap': start_lap
        })
    
    return {
        'total_time': total_time,
        'stints': actual_stints_info,
        'num_pit_stops': len(stints) - 1
    }

def analyze_stint_with_ml(stint_df, model, feature_columns):
    X = stint_df[feature_columns]
    y_true = stint_df['lap_time'].values
    y_pred = model.predict(X)
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        stint_df['tire_age'], stint_df['lap_time']
    )
    
    return {
        'predictions': y_pred,
        'r2_score': r2,
        'mae': mae,
        'linear_slope': slope,
        'compound': stint_df['compound_name'].iloc[-1]
    }

# ============================================
# DRIVER LIST
# ============================================

drivers_info = {
    'VER': 'Max Verstappen', 'PER': 'Sergio Perez', 'LEC': 'Charles Leclerc', 'SAI': 'Carlos Sainz',
    'NOR': 'Lando Norris', 'PIA': 'Oscar Piastri', 'HAM': 'Lewis Hamilton', 'RUS': 'George Russell',
    'ALO': 'Fernando Alonso', 'STR': 'Lance Stroll', 'OCO': 'Esteban Ocon', 'GAS': 'Pierre Gasly',
    'TSU': 'Yuki Tsunoda', 'RIC': 'Daniel Ricciardo', 'HUL': 'Nico Hulkenberg', 'MAG': 'Kevin Magnussen',
    'ALB': 'Alexander Albon', 'SAR': 'Logan Sargeant', 'BOT': 'Valtteri Bottas', 'ZHO': 'Zhou Guanyu'
}

# ============================================
# MAIN PROGRAM
# ============================================

print("=" * 70)
print("ML TIRE STRATEGY OPTIMIZER - PROFESSIONAL EDITION")
print("Multi-session training with circuit characteristics")
print("=" * 70)

print("\nAvailable drivers:")
for code, name in sorted(drivers_info.items()):
    print(f"  {code} - {name}")

print("\n" + "=" * 70)

year = input("\nYear (e.g., 2024): ").strip() or '2024'
gp = input("Grand Prix (e.g., Bahrain, Hungary, Monaco): ").strip() or 'Bahrain'
driver = input("Driver to analyze (e.g., VER): ").strip().upper()

if driver not in drivers_info:
    print("\nError: Invalid driver code")
    exit()

use_all_drivers = input("\nTrain model with ALL drivers? (recommended) [Y/n]: ").strip().upper()
train_driver = None if use_all_drivers != 'N' else driver

if train_driver is None:
    print(f"  -> Training with ALL drivers' data (better predictions)")
else:
    print(f"  -> Training with only {train_driver}'s data")

# CRITICAL: Ask if user wants to train only with race data
use_race_only = input("\nTrain ONLY with Race data? (more accurate predictions) [Y/n]: ").strip().upper()
if use_race_only != 'N':
    print(f"  -> Training ONLY with Race session data (excludes FP/Q)")
    print(f"  -> This gives more accurate race predictions")
    sessions_to_use = ['R']
else:
    print(f"  -> Training with ALL sessions (FP1/FP2/FP3/Q/R)")
    sessions_to_use = None

print("\n" + "=" * 70)
print(f"Loading data: {year} - {gp} GP")
print(f"Analyzing: {drivers_info[driver]}")
print("=" * 70)

# Load sessions
sessions_data = load_all_sessions(year, gp)

if len(sessions_data) == 0:
    print("\nError: No sessions could be loaded")
    exit()

print(f"\n[SUCCESS] Successfully loaded {len(sessions_data)} sessions")

# Data preparation
print("\n" + "=" * 70)
print("STEP 1: DATA PREPARATION FROM ALL SESSIONS")
print("=" * 70)

features_df, full_df = prepare_ml_features_from_sessions(
    sessions_data, gp, target_driver=train_driver, sessions_to_use=sessions_to_use
)

if features_df is None or len(features_df) < 50:
    print("\nError: Not enough data to train model")
    exit()

# CRITICAL: Calculate max tire_age seen in data
max_tire_age_seen = int(features_df['tire_age'].max())

print(f"\n[SUCCESS] Total training samples: {len(features_df)}")
print(f"  Maximum tire_age in data: {max_tire_age_seen} laps")
print(f"  Sessions breakdown:")
for session_name in features_df['session_name'].unique():
    count = len(features_df[features_df['session_name'] == session_name])
    print(f"    {session_name}: {count} laps")

# Get race session
race_session = next((s['session'] for s in sessions_data if s['type'] == 'R'), None)

if race_session is None:
    print("\nError: Race session not found")
    exit()

laps = race_session.laps.pick_drivers(driver)
laps = laps[laps['LapTime'].notna()]
total_race_laps = int(race_session.laps['LapNumber'].max())
pit_stop_time = calculate_pit_stop_time(laps)
stints = detect_stints(laps)

print(f"\nRace parameters:")
print(f"  Total laps: {total_race_laps}")
print(f"  Pit stop time: {pit_stop_time:.1f}s")
print(f"  Stints by {driver}: {len(stints)}")

avg_conditions = {
    'track_temp': features_df[features_df['session_name'] == 'R']['track_temp'].mean(),
    'air_temp': features_df[features_df['session_name'] == 'R']['air_temp'].mean(),
    'wind_speed': features_df[features_df['session_name'] == 'R']['wind_speed'].mean()
}

circuit_features = encode_circuit_features(gp)

# ML training
print("\n" + "=" * 70)
print("STEP 2: TRAINING MACHINE LEARNING MODEL")
print("=" * 70)

model, metrics, feature_columns = train_ml_model(features_df)

print(f"\nModel Performance:")
print(f"  Training R²:   {metrics['train_r2']:.4f}")
print(f"  Testing R²:    {metrics['test_r2']:.4f}")
print(f"  Training MAE:  {metrics['train_mae']:.4f} seconds")
print(f"  Testing MAE:   {metrics['test_mae']:.4f} seconds")

quality = "EXCELLENT" if metrics['test_r2'] > 0.8 else "GOOD" if metrics['test_r2'] > 0.6 else "MODERATE" if metrics['test_r2'] > 0.4 else "LOW"
print(f"  -> {quality} model quality")

print(f"\nTop 10 Most Important Features:")
for i, (feat, imp) in enumerate(sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"  {i:2d}. {feat:25} {imp*100:5.1f}%")

# Strategy optimization
print("\n" + "=" * 70)
print("STEP 3: OPTIMIZING RACE STRATEGY")
print("=" * 70)

print(f"\nSimulating strategies for {total_race_laps}-lap race...")
print(f"Conditions: Track {avg_conditions['track_temp']:.1f}°C, Air {avg_conditions['air_temp']:.1f}°C, Wind {avg_conditions['wind_speed']:.1f} km/h")

optimal_strategies = optimize_race_strategy(
    model, feature_columns, avg_conditions, circuit_features,
    total_race_laps, pit_stop_time, max_tire_age_seen
)

if optimal_strategies is None:
    print("\nCannot optimize strategy - insufficient tire degradation data")
    exit()

best = optimal_strategies['best_strategy']

print(f"\nBEST STRATEGY: {best['type']}")
for i, stint in enumerate(best['stints'], 1):
    print(f"  Stint {i}: {stint['laps']} laps ({stint['compound']})")
    if i < len(best['stints']):
        cumulative_laps = sum([s['laps'] for s in best['stints'][:i]])
        print(f"  PIT STOP {i} (after lap {cumulative_laps}) -> +{pit_stop_time:.1f}s")

total_minutes = int(best['total_time'] // 60)
total_seconds = best['total_time'] % 60
print(f"\nPredicted total race time: {best['total_time']:.1f}s ({total_minutes}:{total_seconds:05.2f})")
print(f"Pit time lost: {best['num_pit_stops'] * pit_stop_time:.1f}s ({best['num_pit_stops']} stops)")

# Actual strategy
print(f"\nACTUAL STRATEGY USED:")
actual_simulation = simulate_actual_strategy(
    stints, model, feature_columns, avg_conditions, circuit_features,
    total_race_laps, pit_stop_time
)

for stint in actual_simulation['stints']:
    print(f"  Stint {stint['stint_number']}: {stint['laps']} laps ({stint['compound']})")
    if stint['stint_number'] < len(actual_simulation['stints']):
        end_lap = stint['start_lap'] + stint['laps'] - 1
        print(f"  PIT STOP {stint['stint_number']} (after lap {end_lap}) -> +{pit_stop_time:.1f}s")

actual_minutes = int(actual_simulation['total_time'] // 60)
actual_seconds = actual_simulation['total_time'] % 60
print(f"\nPredicted total race time: {actual_simulation['total_time']:.1f}s ({actual_minutes}:{actual_seconds:05.2f})")
print(f"Pit time lost: {actual_simulation['num_pit_stops'] * pit_stop_time:.1f}s ({actual_simulation['num_pit_stops']} stops)")

# Comparison
time_diff = actual_simulation['total_time'] - best['total_time']

print(f"\n{'='*70}")
print("STRATEGY COMPARISON")
print(f"{'='*70}")

if abs(time_diff) < 5:
    print(f"[SUCCESS] Strategy was OPTIMAL (difference: {abs(time_diff):.1f}s)")
elif time_diff > 0:
    print(f"[!] Suboptimal strategy")
    print(f"  Time lost: {time_diff:.1f}s compared to best strategy")
    print(f"  Could have finished {time_diff:.1f}s faster with {best['type']}")
else:
    print(f"[i] Actual strategy was {abs(time_diff):.1f}s faster than prediction")
    print(f"\n  WHY? The model is conservative with long stints:")
    print(f"    - Best strategy has stint length: {max([s['laps'] for s in best['stints']])} laps")
    print(f"    - Max tire_age in training: {max_tire_age_seen} laps")
    print(f"    - Actual strategy kept all stints <= {max([s['laps'] for s in actual_simulation['stints']])} laps")
    print(f"    - Fresh tires every {np.mean([s['laps'] for s in actual_simulation['stints']]):.0f} laps is faster!")

print(f"\n{'='*70}")
print("STRATEGY BREAKDOWN BY TYPE")
print(f"{'='*70}")

# Show top 3 of each type
for strategy_type in ['1-STOP', '2-STOP', '3-STOP']:
    strategies_of_type = optimal_strategies['by_type'][strategy_type]
    
    if len(strategies_of_type) == 0:
        print(f"\n{strategy_type}: Not feasible with current tire_age limits")
        continue
    
    best_of_type = strategies_of_type[0]
    time_vs_overall_best = best_of_type['total_time'] - best['total_time']
    
    print(f"\n{strategy_type}:")
    print(f"  Best: {best_of_type['total_time']:.1f}s (+{time_vs_overall_best:.1f}s vs overall best)")
    
    for i, strat in enumerate(strategies_of_type[:3], 1):
        time_vs_best = strat['total_time'] - best_of_type['total_time']
        stints_str = " -> ".join([f"{s['laps']}L {s['compound']}" for s in strat['stints']])
        print(f"    {i}. {stints_str:50} | {strat['total_time']:.1f}s (+{time_vs_best:.1f}s)")

print(f"\nTOP 10 OVERALL STRATEGIES (All Types):")
for i, strat in enumerate(optimal_strategies['all_strategies'][:10], 1):
    time_vs_best = strat['total_time'] - best['total_time']
    stints_str = " -> ".join([f"{s['laps']}L {s['compound']}" for s in strat['stints']])
    marker = "estrella" if i == 1 else " "
    print(f"  {marker}{i:2}. {strat['type']:7} | {stints_str:45} | {strat['total_time']:.1f}s (+{time_vs_best:.1f}s)")

# ============================================
# STINT ANALYSIS & VISUALIZATION
# ============================================

print("\n" + "=" * 70)
print("STEP 4: ANALYZING STINTS")
print("=" * 70)

stint_analyses = []

for idx, stint in enumerate(stints, 1):
    print(f"\nStint {idx}:")
    
    # Prepare features for this stint
    stint_features_list = []
    for _, lap in stint.iterrows():
        try:
            tire_life = lap['TyreLife'] if not pd.isna(lap['TyreLife']) else 1
            compound = lap['Compound'] if not pd.isna(lap['Compound']) else 'MEDIUM'
            compound_encoded = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}.get(compound, 1)
            lap_number = lap['LapNumber']
            lap_time_seconds = lap['LapTime'].total_seconds()
            
            # Get weather
            lap_start_time = lap['LapStartTime']
            try:
                weather_at_lap = race_session.weather_data[race_session.weather_data['Time'] <= lap_start_time].iloc[-1]
                track_temp = weather_at_lap['TrackTemp']
                air_temp = weather_at_lap['AirTemp']
                wind_speed = weather_at_lap['WindSpeed'] if not pd.isna(weather_at_lap['WindSpeed']) else 0.0
            except:
                track_temp = avg_conditions['track_temp']
                air_temp = avg_conditions['air_temp']
                wind_speed = avg_conditions['wind_speed']
            
            stint_features_list.append({
                'tire_age': tire_life,
                'compound': compound_encoded,
                'track_temp': track_temp,
                'air_temp': air_temp,
                'wind_speed': wind_speed,
                'fuel_load': 110 - (lap_number / total_race_laps) * 110,
                'track_evolution': lap_number / total_race_laps,
                'session_type': 4,
                **circuit_features,
                'lap_time': lap_time_seconds,
                'compound_name': compound
            })
        except:
            continue
    
    if len(stint_features_list) < 5:
        print(f"  Skipped (not enough laps)")
        continue
    
    stint_df = pd.DataFrame(stint_features_list)
    
    analysis = analyze_stint_with_ml(stint_df, model, feature_columns)
    
    stint_analyses.append({
        'stint_number': idx,
        'stint_data': stint_df,
        'analysis': analysis
    })
    
    print(f"  Compound: {analysis['compound']}")
    print(f"  Laps: {len(stint_df)}")
    print(f"  ML R² score: {analysis['r2_score']:.4f}")
    print(f"  ML MAE: {analysis['mae']:.4f} seconds")
    print(f"  Linear degradation: {analysis['linear_slope']:.4f} s/lap")

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "=" * 70)
print("STEP 5: GENERATING VISUALIZATIONS")
print("=" * 70)

compound_colors = {
    'SOFT': 'red',
    'MEDIUM': 'yellow',
    'HARD': 'white',
    'INTERMEDIATE': 'green',
    'WET': 'blue'
}

num_stints = len(stint_analyses)

if num_stints > 0:
    fig_height = 6 * num_stints
    fig, axes = plt.subplots(num_stints, 1, figsize=(18, fig_height))
    
    if num_stints == 1:
        axes = [axes]
    
    for idx, stint_info in enumerate(stint_analyses):
        ax = axes[idx]
        
        stint_data = stint_info['stint_data']
        analysis = stint_info['analysis']
        stint_num = stint_info['stint_number']
        
        tire_age = stint_data['tire_age'].values
        actual_times = stint_data['lap_time'].values
        predicted_times = analysis['predictions']
        
        compound = analysis['compound']
        color = compound_colors.get(compound, 'gray')
        edge_color = 'black' if color in ['yellow', 'white'] else color
        
        # Actual lap times
        ax.scatter(tire_age, actual_times, c=color, edgecolors=edge_color,
                  s=150, alpha=0.8, linewidth=2, zorder=5, label='Actual lap times')
        
        # ML prediction
        ax.plot(tire_age, predicted_times, 'b--', linewidth=2.5, 
               alpha=0.7, label='ML prediction', zorder=4)
        
        # Confidence interval
        residuals = actual_times - predicted_times
        std_residuals = np.std(residuals)
        
        ax.fill_between(tire_age, 
                        predicted_times - 1.96 * std_residuals,
                        predicted_times + 1.96 * std_residuals,
                        alpha=0.2, color='blue', label='95% confidence')
        
        ax.set_xlabel('Tire Age (laps)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Lap Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_title(f'Stint {stint_num} - {compound} Tires | ML R²={analysis["r2_score"]:.3f} | MAE={analysis["mae"]:.3f}s',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    fig.suptitle(f'ML Tire Degradation Analysis - {gp} GP {year}\n{drivers_info[driver]} ({driver})',
                 fontsize=17, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    filename = f'output_tire_ml/{gp}_{year}_{driver}_ML_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Visualization saved: {filename}")
    
    plt.show(block=True)
else:
    print("\n[!] No stints to visualize")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nRace: {gp} GP {year} ({total_race_laps} laps)")
print(f"Driver: {drivers_info[driver]} ({driver})")
print(f"Weather: Track {avg_conditions['track_temp']:.1f}°C, Air {avg_conditions['air_temp']:.1f}°C")

circuit_info = next((CIRCUIT_DATA[k] for k in CIRCUIT_DATA if k.lower() in gp.lower()), None)
if circuit_info:
    print(f"\nCircuit Characteristics:")
    print(f"  Type: {circuit_info['type']}")
    print(f"  Average speed: {circuit_info['avg_speed']}")
    print(f"  Corners: {circuit_info['corners']}")
    print(f"  Length: {circuit_info['length_km']:.3f} km")
    print(f"  Surface: {circuit_info['surface']}")

print(f"\nML Model:")
print(f"  Quality: R² = {metrics['test_r2']:.3f}")
print(f"  Training samples: {len(features_df)}")
print(f"  Max tire_age seen: {max_tire_age_seen} laps")
print(f"  Sessions used: {', '.join(features_df['session_name'].unique())}")

print(f"\nStrategy:")
print(f"  Optimal (ML): {best['type']} with max stint {max([s['laps'] for s in best['stints']])} laps")
print(f"  Actual:  {len(stints)}-STOP with max stint {max([s['laps'] for s in actual_simulation['stints']])} laps")

# Calculate prediction quality
time_diff_abs = abs(time_diff)
time_diff_pct = (time_diff_abs / actual_simulation['total_time']) * 100

print(f"\nPrediction Quality:")
print(f"  Time difference: {time_diff_abs:.1f}s ({time_diff_pct:.1f}% error)")

if time_diff_pct < 1.0:
    print(f"  Quality: [SUCCESS] EXCELLENT (< 1% error)")
    print(f"  Result: Predictions are highly reliable")
elif time_diff_pct < 3.0:
    print(f"  Quality: [SUCCESS] GOOD (< 3% error)")
    print(f"  Result: Predictions are reliable")
elif time_diff_pct < 5.0:
    print(f"  Quality:  MODERATE (< 5% error)")
    print(f"  Result: Predictions are acceptable")
else:
    print(f"  Quality: [ERROR] POOR (> 5% error)")
    print(f"  Result: [!] Model needs improvement")
    print(f"\n  RECOMMENDATIONS:")
    if len(features_df[features_df['session_name'] != 'R']) > len(features_df[features_df['session_name'] == 'R']):
        print(f"    - Try training ONLY with Race data (exclude FP/Q)")
        print(f"    - Practice sessions have slower, inconsistent times")
    if len(features_df) < 1000:
        print(f"    - More training data needed (currently {len(features_df)} samples)")
    if metrics['test_r2'] < 0.7:
        print(f"    - Model R² is low ({metrics['test_r2']:.3f})")
        print(f"    - Try using only race data for more consistent patterns")

if abs(time_diff) < 5:
    print(f"\n  Verdict: [SUCCESS] OPTIMAL")
elif time_diff > 0:
    print(f"\n  Verdict: [!] Actual strategy was suboptimal by {time_diff:.1f}s")
else:
    print(f"\n  Verdict: [SUCCESS] Actual was {abs(time_diff):.1f}s faster")
    if time_diff_pct < 3.0:
        print(f"           (Within acceptable margin)")
    else:
        print(f"           (Model underestimated - see recommendations above)")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}\n")
