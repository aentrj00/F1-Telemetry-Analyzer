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
# HELPER FUNCTIONS
# ============================================

def detect_stints(laps):
    """
    Detects different stints (runs with same tire)
    Each pit stop = new stint
    Returns list of DataFrames, one per stint
    """
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
    """
    Calculates average pit stop time from actual race data
    Uses difference between PitInTime and PitOutTime
    """
    pit_times = []
    
    for idx, lap in laps.iterrows():
        if pd.notna(lap['PitInTime']) and pd.notna(lap['PitOutTime']):
            pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
            if 15 < pit_duration < 60:  # Sanity check (15-60 seconds)
                pit_times.append(pit_duration)
    
    if pit_times:
        return np.mean(pit_times)
    else:
        # Fallback: estimate based on circuit
        return 23.0  # Typical F1 pit stop with pit lane time

def prepare_ml_features(laps, session, total_race_laps):
    """
    Prepares features for machine learning model
    Returns DataFrame with features and target
    """
    features_list = []
    
    weather_data = session.weather_data
    
    for idx, lap in laps.iterrows():
        lap_number = lap['LapNumber']
        lap_time = lap['LapTime']
        
        if pd.isna(lap_time):
            continue
        
        lap_time_seconds = lap_time.total_seconds()
        lap_start_time = lap['LapStartTime']
        
        # Get weather at this lap
        try:
            weather_at_lap = weather_data[weather_data['Time'] <= lap_start_time].iloc[-1]
            track_temp = weather_at_lap['TrackTemp']
            air_temp = weather_at_lap['AirTemp']
        except:
            track_temp = 30.0
            air_temp = 25.0
        
        # Get compound
        try:
            compound = lap['Compound']
            if pd.isna(compound):
                compound = 'MEDIUM'
        except:
            compound = 'MEDIUM'
        
        # Encode compound: SOFT=0, MEDIUM=1, HARD=2
        compound_encoded = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 
                           'INTERMEDIATE': 3, 'WET': 4}.get(compound, 1)
        
        # Get tire age
        try:
            tire_life = lap['TyreLife']
            if pd.isna(tire_life):
                tire_life = 1
        except:
            tire_life = 1
        
        # Estimate fuel load (starts at ~110kg, decreases linearly)
        fuel_load = 110 - (lap_number / total_race_laps) * 110
        
        # Track evolution (rubber buildup makes track faster)
        track_evolution = lap_number / total_race_laps
        
        features_list.append({
            'lap_number': lap_number,
            'tire_age': tire_life,
            'compound': compound_encoded,
            'track_temp': track_temp,
            'air_temp': air_temp,
            'fuel_load': fuel_load,
            'track_evolution': track_evolution,
            'lap_time': lap_time_seconds,
            'compound_name': compound
        })
    
    df = pd.DataFrame(features_list)
    
    # Remove outliers (traffic, incidents)
    mean_time = df['lap_time'].mean()
    std_time = df['lap_time'].std()
    df['outlier'] = np.abs(df['lap_time'] - mean_time) > 2 * std_time
    
    df_clean = df[~df['outlier']].copy()
    
    return df_clean, df

def train_ml_model(features_df):
    """
    Trains Random Forest model for lap time prediction
    Returns trained model and performance metrics
    """
    feature_columns = ['tire_age', 'compound', 'track_temp', 
                      'air_temp', 'fuel_load', 'track_evolution']
    
    X = features_df[feature_columns]
    y = features_df['lap_time']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
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
    
    return model, metrics

def optimize_race_strategy(model, avg_conditions, total_race_laps, pit_stop_time):
    """
    Optimizes complete race strategy
    Compares 1-stop, 2-stop, 3-stop strategies
    Returns best strategy with total race time
    """
    
    strategies = []
    
    # ============================================
    # STRATEGY 1: ONE-STOP
    # ============================================
    
    # Try different pit windows for 1-stop
    for pit_lap in range(int(total_race_laps * 0.3), int(total_race_laps * 0.7), 2):
        
        total_time = 0
        
        # Stint 1: Start to pit (MEDIUM)
        for lap in range(1, pit_lap + 1):
            features = np.array([[
                lap,  # tire_age
                1,    # MEDIUM compound
                avg_conditions['track_temp'],
                avg_conditions['air_temp'],
                110 - (lap / total_race_laps) * 110,
                lap / total_race_laps
            ]])
            lap_time = model.predict(features)[0]
            total_time += lap_time
        
        # Pit stop
        total_time += pit_stop_time
        
        # Stint 2: After pit to finish (HARD)
        stint2_laps = total_race_laps - pit_lap
        for lap_in_stint in range(1, stint2_laps + 1):
            race_lap = pit_lap + lap_in_stint
            features = np.array([[
                lap_in_stint,  # tire_age (reset after pit)
                2,    # HARD compound
                avg_conditions['track_temp'],
                avg_conditions['air_temp'],
                110 - (race_lap / total_race_laps) * 110,
                race_lap / total_race_laps
            ]])
            lap_time = model.predict(features)[0]
            total_time += lap_time
        
        strategies.append({
            'type': '1-STOP',
            'pit_laps': [pit_lap],
            'stints': [
                {'laps': pit_lap, 'compound': 'MEDIUM', 'compound_code': 1},
                {'laps': stint2_laps, 'compound': 'HARD', 'compound_code': 2}
            ],
            'total_time': total_time,
            'num_pit_stops': 1
        })
    
    # ============================================
    # STRATEGY 2: TWO-STOP
    # ============================================
    
    # Try different pit windows for 2-stop
    pit1_range = range(int(total_race_laps * 0.25), int(total_race_laps * 0.4), 3)
    pit2_range_base = range(int(total_race_laps * 0.5), int(total_race_laps * 0.75), 3)
    
    for pit1 in pit1_range:
        for pit2 in pit2_range_base:
            if pit2 <= pit1 + 10:  # Minimum stint length
                continue
                
            total_time = 0
            
            # Stint 1: MEDIUM
            for lap in range(1, pit1 + 1):
                features = np.array([[
                    lap, 1,
                    avg_conditions['track_temp'],
                    avg_conditions['air_temp'],
                    110 - (lap / total_race_laps) * 110,
                    lap / total_race_laps
                ]])
                total_time += model.predict(features)[0]
            
            total_time += pit_stop_time
            
            # Stint 2: HARD
            stint2_length = pit2 - pit1
            for lap_in_stint in range(1, stint2_length + 1):
                race_lap = pit1 + lap_in_stint
                features = np.array([[
                    lap_in_stint, 2,
                    avg_conditions['track_temp'],
                    avg_conditions['air_temp'],
                    110 - (race_lap / total_race_laps) * 110,
                    race_lap / total_race_laps
                ]])
                total_time += model.predict(features)[0]
            
            total_time += pit_stop_time
            
            # Stint 3: HARD
            stint3_length = total_race_laps - pit2
            for lap_in_stint in range(1, stint3_length + 1):
                race_lap = pit2 + lap_in_stint
                features = np.array([[
                    lap_in_stint, 2,
                    avg_conditions['track_temp'],
                    avg_conditions['air_temp'],
                    110 - (race_lap / total_race_laps) * 110,
                    race_lap / total_race_laps
                ]])
                total_time += model.predict(features)[0]
            
            strategies.append({
                'type': '2-STOP',
                'pit_laps': [pit1, pit2],
                'stints': [
                    {'laps': pit1, 'compound': 'MEDIUM', 'compound_code': 1},
                    {'laps': stint2_length, 'compound': 'HARD', 'compound_code': 2},
                    {'laps': stint3_length, 'compound': 'HARD', 'compound_code': 2}
                ],
                'total_time': total_time,
                'num_pit_stops': 2
            })
    
    # Find best strategy
    best_strategy = min(strategies, key=lambda x: x['total_time'])
    
    # Get top 5 alternatives
    top_strategies = sorted(strategies, key=lambda x: x['total_time'])[:5]
    
    return {
        'best_strategy': best_strategy,
        'all_strategies': top_strategies
    }

def simulate_actual_strategy(stints, model, avg_conditions, total_race_laps, pit_stop_time):
    """
    Simulates the actual strategy the driver used
    Returns estimated total race time
    """
    total_time = 0
    actual_stints_info = []
    
    for stint_idx, stint in enumerate(stints):
        stint_laps = len(stint)
        
        # Get compound
        try:
            compound_name = stint.iloc[0]['Compound']
            if pd.isna(compound_name):
                compound_name = 'MEDIUM'
        except:
            compound_name = 'MEDIUM'
        
        compound_code = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}.get(compound_name, 1)
        
        # Get starting lap number
        try:
            start_lap = stint.iloc[0]['LapNumber']
        except:
            start_lap = sum([len(s) for s in stints[:stint_idx]]) + stint_idx + 1
        
        # Simulate this stint
        stint_time = 0
        for lap_in_stint in range(1, stint_laps + 1):
            race_lap = start_lap + lap_in_stint - 1
            
            features = np.array([[
                lap_in_stint,
                compound_code,
                avg_conditions['track_temp'],
                avg_conditions['air_temp'],
                110 - (race_lap / total_race_laps) * 110,
                race_lap / total_race_laps
            ]])
            
            lap_time = model.predict(features)[0]
            stint_time += lap_time
        
        total_time += stint_time
        
        # Add pit stop time (except after last stint)
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

def predict_future_laps(model, last_lap_features, num_laps=10):
    """
    Predicts lap times for future laps
    """
    predictions = []
    
    for i in range(1, num_laps + 1):
        future_features = last_lap_features.copy()
        future_features['tire_age'] += i
        future_features['fuel_load'] = max(0, future_features['fuel_load'] - 2.0 * i)
        future_features['track_evolution'] = min(1.0, future_features['track_evolution'] + 0.01 * i)
        
        features_array = np.array([[
            future_features['tire_age'],
            future_features['compound'],
            future_features['track_temp'],
            future_features['air_temp'],
            future_features['fuel_load'],
            future_features['track_evolution']
        ]])
        
        pred = model.predict(features_array)[0]
        
        # Estimate uncertainty
        std_dev = 0.15
        
        predictions.append({
            'lap_offset': i,
            'predicted_time': pred,
            'lower_bound': pred - 1.96 * std_dev,
            'upper_bound': pred + 1.96 * std_dev
        })
    
    return predictions

def analyze_stint_with_ml(stint_df, model):
    """
    Complete ML analysis of a single stint
    """
    feature_columns = ['tire_age', 'compound', 'track_temp', 
                      'air_temp', 'fuel_load', 'track_evolution']
    
    X = stint_df[feature_columns]
    y_true = stint_df['lap_time'].values
    y_pred = model.predict(X)
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Also calculate linear degradation for reference
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        stint_df['tire_age'], stint_df['lap_time']
    )
    
    # Future predictions
    last_lap = stint_df.iloc[-1]
    last_features = {
        'tire_age': last_lap['tire_age'],
        'compound': last_lap['compound'],
        'track_temp': last_lap['track_temp'],
        'air_temp': last_lap['air_temp'],
        'fuel_load': last_lap['fuel_load'],
        'track_evolution': last_lap['track_evolution']
    }
    
    future_predictions = predict_future_laps(model, last_features, num_laps=15)
    
    return {
        'predictions': y_pred,
        'r2_score': r2,
        'mae': mae,
        'linear_slope': slope,
        'linear_intercept': intercept,
        'future_predictions': future_predictions,
        'compound': last_lap['compound_name']
    }

# ============================================
# DRIVER LIST
# ============================================

drivers_info = {
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
# MAIN PROGRAM
# ============================================

print("=" * 70)
print("ML TIRE STRATEGY OPTIMIZER")
print("Advanced tire degradation analysis with Machine Learning")
print("=" * 70)

print("\nAvailable drivers:")
for code, name in sorted(drivers_info.items()):
    print(f"  {code} - {name}")

print("\n" + "=" * 70)

year = input("\nYear (e.g., 2024): ").strip()
if not year:
    year = '2024'

gp = input("Grand Prix (e.g., Bahrain, Monaco, Monza): ").strip()
if not gp:
    gp = 'Bahrain'

driver = input("Driver to analyze (e.g., VER): ").strip().upper()

if driver not in drivers_info:
    print("\nError: Invalid driver code")
    exit()

print("\n" + "=" * 70)
print(f"Loading race: {year} - {gp} GP")
print(f"Analyzing: {drivers_info[driver]}")
print("=" * 70)

try:
    session = ff1.get_session(int(year), gp, 'R')
    session.load()
    print("Race loaded successfully\n")
except Exception as e:
    print(f"\nError loading session: {e}")
    exit()

# ============================================
# EXTRACT RACE PARAMETERS
# ============================================

# Get total race laps (actual from session)
all_laps = session.laps
total_race_laps = int(all_laps['LapNumber'].max())

print(f"Race parameters:")
print(f"  Total laps: {total_race_laps}")

# Calculate pit stop time from data
laps = session.laps.pick_drivers(driver)
pit_stop_time = calculate_pit_stop_time(laps)
print(f"  Average pit stop time: {pit_stop_time:.1f}s")

# ============================================
# DATA PREPARATION
# ============================================

print("\n" + "=" * 70)
print("STEP 1: DATA PREPARATION")
print("=" * 70)

laps = laps[laps['LapTime'].notna()]
print(f"Total laps by {driver}: {len(laps)}")

features_df, full_df = prepare_ml_features(laps, session, total_race_laps)

outliers_removed = len(full_df) - len(features_df)
print(f"Outliers removed (traffic/incidents): {outliers_removed}")
print(f"Clean laps for ML training: {len(features_df)}")

stints = detect_stints(laps)
print(f"Stints detected: {len(stints)}")

# Calculate average conditions
avg_conditions = {
    'track_temp': features_df['track_temp'].mean(),
    'air_temp': features_df['air_temp'].mean()
}

# ============================================
# ML MODEL TRAINING
# ============================================

print("\n" + "=" * 70)
print("STEP 2: TRAINING MACHINE LEARNING MODEL")
print("=" * 70)

model, metrics = train_ml_model(features_df)

print(f"\nModel Performance:")
print(f"  Training R²:   {metrics['train_r2']:.4f}")
print(f"  Testing R²:    {metrics['test_r2']:.4f}")
print(f"  Training MAE:  {metrics['train_mae']:.4f} seconds")
print(f"  Testing MAE:   {metrics['test_mae']:.4f} seconds")

print(f"\nFeature Importance:")
for feature, importance in sorted(metrics['feature_importance'].items(), 
                                  key=lambda x: x[1], reverse=True):
    print(f"  {feature:20} {importance:.4f} ({importance*100:.1f}%)")

# ============================================
# STRATEGY OPTIMIZATION
# ============================================

print("\n" + "=" * 70)
print("STEP 3: OPTIMIZING RACE STRATEGY")
print("=" * 70)

print(f"\nSimulating different strategies for {total_race_laps}-lap race...")
print(f"Pit stop time: {pit_stop_time:.1f}s")

optimal_strategies = optimize_race_strategy(
    model, avg_conditions, total_race_laps, pit_stop_time
)

best = optimal_strategies['best_strategy']

print(f"\nBEST STRATEGY: {best['type']}")
for i, stint in enumerate(best['stints'], 1):
    print(f"  Stint {i}: {stint['laps']} laps ({stint['compound']})")
    if i < len(best['stints']):
        print(f"  PIT STOP {i} (after lap {sum([s['laps'] for s in best['stints'][:i]])}) → +{pit_stop_time:.1f}s")

total_minutes = int(best['total_time'] // 60)
total_seconds = best['total_time'] % 60
print(f"\nEstimated total race time: {best['total_time']:.1f}s ({total_minutes}:{total_seconds:05.2f})")
print(f"Pit time lost: {best['num_pit_stops'] * pit_stop_time:.1f}s ({best['num_pit_stops']} stops)")

# ============================================
# ACTUAL STRATEGY ANALYSIS
# ============================================

print(f"\nACTUAL STRATEGY USED:")
actual_simulation = simulate_actual_strategy(
    stints, model, avg_conditions, total_race_laps, pit_stop_time
)

for stint in actual_simulation['stints']:
    print(f"  Stint {stint['stint_number']}: {stint['laps']} laps ({stint['compound']})")
    if stint['stint_number'] < len(actual_simulation['stints']):
        print(f"  PIT STOP {stint['stint_number']} (after lap {stint['start_lap'] + stint['laps'] - 1}) → +{pit_stop_time:.1f}s")

actual_minutes = int(actual_simulation['total_time'] // 60)
actual_seconds = actual_simulation['total_time'] % 60
print(f"\nEstimated total race time: {actual_simulation['total_time']:.1f}s ({actual_minutes}:{actual_seconds:05.2f})")
print(f"Pit time lost: {actual_simulation['num_pit_stops'] * pit_stop_time:.1f}s ({actual_simulation['num_pit_stops']} stops)")

# ============================================
# COMPARISON
# ============================================

time_diff = actual_simulation['total_time'] - best['total_time']

print(f"\n{'='*70}")
print("STRATEGY COMPARISON")
print(f"{'='*70}")

if abs(time_diff) < 5:
    print(f"✓ Strategy was OPTIMAL (difference: {abs(time_diff):.1f}s)")
elif time_diff > 0:
    print(f"⚠ Suboptimal strategy")
    print(f"  Time lost: {time_diff:.1f}s compared to best strategy")
    print(f"  Could have finished {time_diff:.1f}s faster with {best['type']}")
else:
    print(f"ℹ Actual strategy was {abs(time_diff):.1f}s faster than ML prediction")
    print(f"  (Model may be conservative)")

# Show alternative strategies
print(f"\nTOP 5 ALTERNATIVE STRATEGIES:")
for i, strat in enumerate(optimal_strategies['all_strategies'][:5], 1):
    time_vs_best = strat['total_time'] - best['total_time']
    stints_str = " → ".join([f"{s['laps']}L {s['compound']}" for s in strat['stints']])
    print(f"  {i}. {strat['type']:7} | {stints_str:40} | {strat['total_time']:.1f}s (+{time_vs_best:.1f}s)")

# ============================================
# STINT ANALYSIS
# ============================================

print("\n" + "=" * 70)
print("STEP 4: ANALYZING EACH STINT")
print("=" * 70)

stint_analyses = []

for idx, stint in enumerate(stints, 1):
    print(f"\nStint {idx}:")
    
    stint_features, _ = prepare_ml_features(stint, session, total_race_laps)
    
    if len(stint_features) < 5:
        print(f"  Skipped (not enough clean laps)")
        continue
    
    analysis = analyze_stint_with_ml(stint_features, model)
    
    stint_analyses.append({
        'stint_number': idx,
        'stint_data': stint_features,
        'analysis': analysis
    })
    
    print(f"  Compound: {analysis['compound']}")
    print(f"  Laps: {len(stint_features)}")
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
    
    # Future predictions
    if analysis['future_predictions']:
        future_data = analysis['future_predictions']
        future_ages = tire_age[-1] + np.array([p['lap_offset'] for p in future_data])
        future_times = [p['predicted_time'] for p in future_data]
        future_lower = [p['lower_bound'] for p in future_data]
        future_upper = [p['upper_bound'] for p in future_data]
        
        ax.plot(future_ages, future_times, 'r-', linewidth=3, 
               alpha=0.8, label='Future prediction', zorder=6)
        
        ax.fill_between(future_ages, future_lower, future_upper,
                       alpha=0.2, color='red')
    
    ax.set_xlabel('Tire Age (laps)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Lap Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'Stint {stint_num} - {compound} Tires | ML R²={analysis["r2_score"]:.3f} | MAE={analysis["mae"]:.3f}s',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.95)

fig.suptitle(f'ML Tire Degradation Analysis - {gp} GP {year}\n' +
             f'{drivers_info[driver]} ({driver})',
             fontsize=17, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.99])

filename = f'output_tire_ml/{gp}_{year}_{driver}_ML_analysis.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved: {filename}")

plt.show(block=True)

# ============================================
# FINAL REPORT
# ============================================

print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

print(f"\nRace: {gp} GP {year} ({total_race_laps} laps)")
print(f"Driver: {drivers_info[driver]} ({driver})")
print(f"Weather: Track {avg_conditions['track_temp']:.1f}°C, Air {avg_conditions['air_temp']:.1f}°C")

print(f"\nML Model Quality:")
print(f"  Test R²: {metrics['test_r2']:.3f}")
if metrics['test_r2'] > 0.7:
    print(f"  → Reliable predictions")
elif metrics['test_r2'] > 0.4:
    print(f"  → Moderate reliability")
else:
    print(f"  → Low reliability (high variance in data)")

print(f"\nStrategy Analysis:")
print(f"  Optimal: {best['type']}")
print(f"  Actual:  {actual_simulation['num_pit_stops'] + 1}-STOP")
if abs(time_diff) < 5:
    print(f"  Result:  ✓ OPTIMAL")
elif time_diff > 0:
    print(f"  Result:  ⚠ Lost {time_diff:.1f}s")
else:
    print(f"  Result:  ✓ Better than predicted")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}\n")