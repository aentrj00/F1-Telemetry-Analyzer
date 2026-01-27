import fastf1 as ff1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats

# ============================================
# CONFIGURATION
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('output_race_pace'):
    os.makedirs('output_race_pace')

ff1.Cache.enable_cache('cache')

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_fuel_corrected_time(lap_time, lap_number, total_laps, fuel_effect=0.035):
    """
    Corrects lap time for fuel load
    
    Args:
        lap_time: Actual lap time (seconds)
        lap_number: Current lap number
        total_laps: Total race laps
        fuel_effect: Time effect per kg of fuel (default 0.035s/kg)
    
    Returns:
        Fuel-corrected lap time (seconds)
    """
    # F1 starts with ~110kg fuel, burns ~1.5kg per lap
    fuel_remaining = 110 - (lap_number / total_laps) * 110
    fuel_at_start = 110
    
    # Time advantage from lighter car
    fuel_delta = fuel_at_start - fuel_remaining
    time_advantage = fuel_delta * fuel_effect
    
    # Corrected time = actual time + time advantage
    return lap_time + time_advantage

def detect_traffic(lap, gap_ahead_threshold=2.0):
    """
    Detects if lap was affected by traffic
    
    Args:
        lap: FastF1 lap object
        gap_ahead_threshold: Gap threshold to consider traffic (seconds)
    
    Returns:
        True if likely affected by traffic
    """
    # Method 1: Check if there's a position change
    if 'Position' in lap.index and pd.notna(lap['Position']):
        # If position changed, might be traffic/overtaking
        pass
    
    # Method 2: Check lap time anomaly
    # This is basic - more sophisticated would use telemetry
    return False  # Conservative - mark traffic manually in real use

def calculate_stint_pace(laps_df, total_race_laps, fuel_effect=0.035):
    """
    Calculates average pace for a stint with fuel correction
    
    Returns:
        Dict with stint statistics
    """
    if len(laps_df) == 0:
        return None
    
    lap_times = []
    fuel_corrected_times = []
    lap_numbers = []
    
    for idx, lap in laps_df.iterrows():
        if pd.isna(lap['LapTime']):
            continue
        
        lap_time = lap['LapTime'].total_seconds()
        lap_num = lap['LapNumber']
        
        # Skip outliers (pit laps, traffic, etc.)
        if lap_time > 200:  # Sanity check
            continue
        
        corrected_time = calculate_fuel_corrected_time(lap_time, lap_num, total_race_laps, fuel_effect)
        
        lap_times.append(lap_time)
        fuel_corrected_times.append(corrected_time)
        lap_numbers.append(lap_num)
    
    if len(lap_times) == 0:
        return None
    
    # Remove outliers (2 sigma)
    lap_times_array = np.array(lap_times)
    mean = np.mean(lap_times_array)
    std = np.std(lap_times_array)
    
    clean_laps = lap_times_array[np.abs(lap_times_array - mean) < 2 * std]
    clean_corrected = np.array(fuel_corrected_times)[np.abs(lap_times_array - mean) < 2 * std]
    
    return {
        'avg_raw': np.mean(clean_laps),
        'avg_corrected': np.mean(clean_corrected),
        'std_raw': np.std(clean_laps),
        'std_corrected': np.std(clean_corrected),
        'min_raw': np.min(clean_laps),
        'min_corrected': np.min(clean_corrected),
        'num_laps': len(clean_laps)
    }

def detect_stints(laps):
    """
    Detects tire stints in race
    """
    stints = []
    current_stint = []
    
    for idx, lap in laps.iterrows():
        if len(current_stint) == 0:
            current_stint = [lap]
        elif pd.notna(lap['PitInTime']):
            if len(current_stint) >= 3:
                stints.append(pd.DataFrame(current_stint))
            current_stint = []
        else:
            current_stint.append(lap)
    
    if len(current_stint) >= 3:
        stints.append(pd.DataFrame(current_stint))
    
    return stints

def calculate_undercut_potential(driver1_pace, driver2_pace, pit_loss=23.0):
    """
    Calculates undercut potential
    
    Args:
        driver1_pace: Corrected pace of driver planning undercut
        driver2_pace: Corrected pace of driver ahead
        pit_loss: Time lost in pit stop (seconds)
    
    Returns:
        Laps needed to overcome pit stop delta
    """
    pace_advantage = driver2_pace - driver1_pace
    
    if pace_advantage <= 0:
        return float('inf')  # No advantage
    
    laps_needed = pit_loss / pace_advantage
    
    return laps_needed

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
print("RACE PACE ANALYZER")
print("Fuel-corrected pace analysis with stint comparison")
print("=" * 70)

print("\nAvailable drivers:")
for code, name in sorted(drivers_info.items()):
    print(f"  {code} - {name}")

print("\n" + "=" * 70)

# Get input
year = input("\nYear (e.g., 2024): ").strip()
if not year:
    year = '2024'

gp = input("Grand Prix (e.g., Bahrain, Monaco, Monza): ").strip()
if not gp:
    gp = 'Bahrain'

print("\n" + "=" * 70)
driver1 = input("Driver 1 (e.g., VER): ").strip().upper()
driver2 = input("Driver 2 (e.g., NOR): ").strip().upper()

if driver1 not in drivers_info or driver2 not in drivers_info:
    print("\nError: Invalid driver code")
    exit()

if driver1 == driver2:
    print("\nError: Choose different drivers")
    exit()

fuel_effect = input("\nFuel effect (seconds per kg) [0.035]: ").strip()
if not fuel_effect:
    fuel_effect = 0.035
else:
    fuel_effect = float(fuel_effect)

print("\n" + "=" * 70)
print(f"Loading race: {year} - {gp} GP")
print(f"Comparing: {drivers_info[driver1]} vs {drivers_info[driver2]}")
print(f"Fuel effect: {fuel_effect}s/kg")
print("=" * 70)

# Load session
try:
    session = ff1.get_session(int(year), gp, 'R')
    session.load()
    print("\n✓ Race session loaded successfully")
except Exception as e:
    print(f"\n✗ Error loading session: {e}")
    exit()

# Get driver laps
try:
    laps1 = session.laps.pick_drivers(driver1)
    laps1 = laps1[laps1['LapTime'].notna()]
    
    laps2 = session.laps.pick_drivers(driver2)
    laps2 = laps2[laps2['LapTime'].notna()]
    
    if len(laps1) == 0 or len(laps2) == 0:
        print(f"\n✗ Not enough valid laps")
        exit()
    
    print(f"✓ {driver1}: {len(laps1)} laps")
    print(f"✓ {driver2}: {len(laps2)} laps")
except Exception as e:
    print(f"\n✗ Error getting laps: {e}")
    exit()

total_race_laps = int(session.laps['LapNumber'].max())
print(f"✓ Total race laps: {total_race_laps}")

# ============================================
# STINT ANALYSIS
# ============================================

print("\n" + "=" * 70)
print("ANALYZING STINTS")
print("=" * 70)

stints1 = detect_stints(laps1)
stints2 = detect_stints(laps2)

print(f"\n{driver1}: {len(stints1)} stints detected")
print(f"{driver2}: {len(stints2)} stints detected")

# Analyze each stint
stint_analysis1 = []
stint_analysis2 = []

print(f"\n{driver1} Stint Analysis:")
for i, stint in enumerate(stints1, 1):
    compound = stint.iloc[0]['Compound'] if 'Compound' in stint.columns else 'UNKNOWN'
    start_lap = int(stint.iloc[0]['LapNumber'])
    end_lap = int(stint.iloc[-1]['LapNumber'])
    
    stats = calculate_stint_pace(stint, total_race_laps, fuel_effect)
    
    if stats:
        stint_analysis1.append({
            'stint': i,
            'compound': compound,
            'start_lap': start_lap,
            'end_lap': end_lap,
            'laps': stats['num_laps'],
            **stats
        })
        
        print(f"  Stint {i} ({compound}): Laps {start_lap}-{end_lap}")
        print(f"    Raw pace: {stats['avg_raw']:.3f}s ± {stats['std_raw']:.3f}s")
        print(f"    Fuel-corrected: {stats['avg_corrected']:.3f}s ± {stats['std_corrected']:.3f}s")

print(f"\n{driver2} Stint Analysis:")
for i, stint in enumerate(stints2, 1):
    compound = stint.iloc[0]['Compound'] if 'Compound' in stint.columns else 'UNKNOWN'
    start_lap = int(stint.iloc[0]['LapNumber'])
    end_lap = int(stint.iloc[-1]['LapNumber'])
    
    stats = calculate_stint_pace(stint, total_race_laps, fuel_effect)
    
    if stats:
        stint_analysis2.append({
            'stint': i,
            'compound': compound,
            'start_lap': start_lap,
            'end_lap': end_lap,
            'laps': stats['num_laps'],
            **stats
        })
        
        print(f"  Stint {i} ({compound}): Laps {start_lap}-{end_lap}")
        print(f"    Raw pace: {stats['avg_raw']:.3f}s ± {stats['std_raw']:.3f}s")
        print(f"    Fuel-corrected: {stats['avg_corrected']:.3f}s ± {stats['std_corrected']:.3f}s")

# ============================================
# PACE COMPARISON
# ============================================

print("\n" + "=" * 70)
print("OVERALL PACE COMPARISON")
print("=" * 70)

# Calculate overall race pace
overall1 = calculate_stint_pace(laps1, total_race_laps, fuel_effect)
overall2 = calculate_stint_pace(laps2, total_race_laps, fuel_effect)

if overall1 and overall2:
    print(f"\n{driver1} ({drivers_info[driver1]}):")
    print(f"  Average raw pace: {overall1['avg_raw']:.3f}s")
    print(f"  Fuel-corrected pace: {overall1['avg_corrected']:.3f}s")
    print(f"  Consistency (σ): {overall1['std_corrected']:.3f}s")
    
    print(f"\n{driver2} ({drivers_info[driver2]}):")
    print(f"  Average raw pace: {overall2['avg_raw']:.3f}s")
    print(f"  Fuel-corrected pace: {overall2['avg_corrected']:.3f}s")
    print(f"  Consistency (σ): {overall2['std_corrected']:.3f}s")
    
    pace_diff = overall1['avg_corrected'] - overall2['avg_corrected']
    faster_driver = driver1 if pace_diff < 0 else driver2
    
    print(f"\n{'='*70}")
    print(f"RESULT: {drivers_info[faster_driver]} is faster by {abs(pace_diff):.3f}s/lap")
    print(f"Over {total_race_laps} laps: {abs(pace_diff) * total_race_laps:.1f}s ({abs(pace_diff) * total_race_laps / 60:.1f} minutes)")
    print(f"{'='*70}")

# ============================================
# UNDERCUT/OVERCUT ANALYSIS
# ============================================

print("\n" + "=" * 70)
print("UNDERCUT/OVERCUT POTENTIAL")
print("=" * 70)

if overall1 and overall2:
    # Assume pit stop = 23 seconds
    pit_loss = 23.0
    
    # Undercut: Driver 1 pits first
    if overall1['avg_corrected'] < overall2['avg_corrected']:
        laps_needed = calculate_undercut_potential(
            overall1['avg_corrected'], 
            overall2['avg_corrected'], 
            pit_loss
        )
        
        print(f"\n{driver1} Undercut Potential:")
        print(f"  Pace advantage: {overall2['avg_corrected'] - overall1['avg_corrected']:.3f}s/lap")
        print(f"  Laps needed to overcome pit loss: {laps_needed:.1f} laps")
        
        if laps_needed < 10:
            print(f"  → STRONG undercut potential (< 10 laps)")
        elif laps_needed < 15:
            print(f"  → MODERATE undercut potential")
        else:
            print(f"  → WEAK undercut potential (> 15 laps)")
    
    # Overcut: Driver 2 stays out
    if overall2['avg_corrected'] < overall1['avg_corrected']:
        print(f"\n{driver2} could use overcut strategy:")
        print(f"  By staying out, gains {overall1['avg_corrected'] - overall2['avg_corrected']:.3f}s/lap")
        print(f"  But must manage tire degradation")

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)

# ========== 1. LAP TIME COMPARISON (RAW) ==========
ax1 = fig.add_subplot(gs[0, 0])

for idx, lap in laps1.iterrows():
    if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200:
        lap_num = lap['LapNumber']
        lap_time = lap['LapTime'].total_seconds()
        
        # Color by compound
        compound = lap['Compound'] if 'Compound' in lap.index else None
        color = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white'}.get(compound, 'gray')
        edge = 'black' if color in ['yellow', 'white'] else color
        
        ax1.scatter(lap_num, lap_time, c=color, edgecolors=edge, s=80, alpha=0.7, linewidth=1.5)

for idx, lap in laps2.iterrows():
    if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200:
        lap_num = lap['LapNumber']
        lap_time = lap['LapTime'].total_seconds()
        
        compound = lap['Compound'] if 'Compound' in lap.index else None
        color = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white'}.get(compound, 'gray')
        edge = 'black' if color in ['yellow', 'white'] else color
        
        ax1.scatter(lap_num, lap_time, c=color, edgecolors=edge, s=80, alpha=0.4, 
                   linewidth=1.5, marker='s')

# Moving averages
window = 5
if len(laps1) >= window:
    times1 = [lap['LapTime'].total_seconds() for _, lap in laps1.iterrows() 
              if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200]
    nums1 = [lap['LapNumber'] for _, lap in laps1.iterrows() 
             if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200]
    
    if len(times1) >= window:
        moving_avg1 = pd.Series(times1).rolling(window=window, center=True).mean()
        ax1.plot(nums1, moving_avg1, 'b-', linewidth=3, alpha=0.8, label=f'{driver1} trend')

if len(laps2) >= window:
    times2 = [lap['LapTime'].total_seconds() for _, lap in laps2.iterrows() 
              if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200]
    nums2 = [lap['LapNumber'] for _, lap in laps2.iterrows() 
             if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200]
    
    if len(times2) >= window:
        moving_avg2 = pd.Series(times2).rolling(window=window, center=True).mean()
        ax1.plot(nums2, moving_avg2, 'orange', linewidth=3, alpha=0.8, label=f'{driver2} trend')

ax1.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Lap Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title(f'Raw Lap Times - {driver1} (circles) vs {driver2} (squares)', 
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=10)

# ========== 2. FUEL-CORRECTED PACE ==========
ax2 = fig.add_subplot(gs[0, 1])

corrected1 = []
nums1_corr = []
for idx, lap in laps1.iterrows():
    if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200:
        lap_num = lap['LapNumber']
        lap_time = lap['LapTime'].total_seconds()
        corrected = calculate_fuel_corrected_time(lap_time, lap_num, total_race_laps, fuel_effect)
        corrected1.append(corrected)
        nums1_corr.append(lap_num)
        
        compound = lap['Compound'] if 'Compound' in lap.index else None
        color = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white'}.get(compound, 'gray')
        edge = 'black' if color in ['yellow', 'white'] else color
        
        ax2.scatter(lap_num, corrected, c=color, edgecolors=edge, s=80, alpha=0.7, linewidth=1.5)

corrected2 = []
nums2_corr = []
for idx, lap in laps2.iterrows():
    if pd.notna(lap['LapTime']) and lap['LapTime'].total_seconds() < 200:
        lap_num = lap['LapNumber']
        lap_time = lap['LapTime'].total_seconds()
        corrected = calculate_fuel_corrected_time(lap_time, lap_num, total_race_laps, fuel_effect)
        corrected2.append(corrected)
        nums2_corr.append(lap_num)
        
        compound = lap['Compound'] if 'Compound' in lap.index else None
        color = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white'}.get(compound, 'gray')
        edge = 'black' if color in ['yellow', 'white'] else color
        
        ax2.scatter(lap_num, corrected, c=color, edgecolors=edge, s=80, alpha=0.4, 
                   linewidth=1.5, marker='s')

# Averages
if overall1:
    ax2.axhline(y=overall1['avg_corrected'], color='blue', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'{driver1} avg: {overall1["avg_corrected"]:.3f}s')
if overall2:
    ax2.axhline(y=overall2['avg_corrected'], color='orange', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'{driver2} avg: {overall2["avg_corrected"]:.3f}s')

ax2.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
ax2.set_ylabel('Fuel-Corrected Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Fuel-Corrected Pace Comparison', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=10)

# ========== 3. STINT-BY-STINT COMPARISON ==========
ax3 = fig.add_subplot(gs[1, :])

x_positions = []
colors_list = []
labels_list = []
values_raw = []
values_corrected = []

x = 0
for stint in stint_analysis1:
    x_positions.append(x)
    colors_list.append('blue')
    labels_list.append(f"{driver1}\nS{stint['stint']}\n{stint['compound']}")
    values_raw.append(stint['avg_raw'])
    values_corrected.append(stint['avg_corrected'])
    x += 1

for stint in stint_analysis2:
    x_positions.append(x)
    colors_list.append('orange')
    labels_list.append(f"{driver2}\nS{stint['stint']}\n{stint['compound']}")
    values_raw.append(stint['avg_raw'])
    values_corrected.append(stint['avg_corrected'])
    x += 1

width = 0.35
x_array = np.arange(len(x_positions))

bars1 = ax3.bar(x_array - width/2, values_raw, width, label='Raw pace', 
               color=colors_list, alpha=0.5, edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x_array + width/2, values_corrected, width, label='Fuel-corrected', 
               color=colors_list, alpha=0.9, edgecolor='black', linewidth=1.5)

ax3.set_xlabel('Stint', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average Lap Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('Stint-by-Stint Pace Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x_array)
ax3.set_xticklabels(labels_list, fontsize=9)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# ========== 4. CONSISTENCY COMPARISON ==========
ax4 = fig.add_subplot(gs[2, 0])

if len(corrected1) > 0 and len(corrected2) > 0:
    ax4.hist(corrected1, bins=20, alpha=0.6, color='blue', edgecolor='black', 
            label=f'{driver1}', density=True)
    ax4.hist(corrected2, bins=20, alpha=0.6, color='orange', edgecolor='black', 
            label=f'{driver2}', density=True)
    
    # Add vertical lines for means
    if overall1:
        ax4.axvline(x=overall1['avg_corrected'], color='blue', linestyle='--', 
                   linewidth=2, label=f'{driver1} mean')
    if overall2:
        ax4.axvline(x=overall2['avg_corrected'], color='orange', linestyle='--', 
                   linewidth=2, label=f'{driver2} mean')
    
    ax4.set_xlabel('Fuel-Corrected Lap Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax4.set_title('Pace Distribution (Consistency)', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# ========== 5. PACE DELTA EVOLUTION ==========
ax5 = fig.add_subplot(gs[2, 1])

# Calculate delta per lap
delta_laps = []
delta_values = []

for lap_num in range(1, total_race_laps + 1):
    lap1 = laps1[laps1['LapNumber'] == lap_num]
    lap2 = laps2[laps2['LapNumber'] == lap_num]
    
    if len(lap1) > 0 and len(lap2) > 0:
        time1 = lap1.iloc[0]['LapTime']
        time2 = lap2.iloc[0]['LapTime']
        
        if pd.notna(time1) and pd.notna(time2):
            t1 = time1.total_seconds()
            t2 = time2.total_seconds()
            
            if t1 < 200 and t2 < 200:  # Sanity check
                # Fuel correct both
                corr1 = calculate_fuel_corrected_time(t1, lap_num, total_race_laps, fuel_effect)
                corr2 = calculate_fuel_corrected_time(t2, lap_num, total_race_laps, fuel_effect)
                
                delta = corr1 - corr2  # Positive = driver1 slower
                
                delta_laps.append(lap_num)
                delta_values.append(delta)

if len(delta_laps) > 0:
    ax5.scatter(delta_laps, delta_values, c='purple', s=50, alpha=0.6, edgecolors='black')
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Equal pace')
    
    # Moving average
    if len(delta_values) >= 5:
        moving_avg = pd.Series(delta_values).rolling(window=5, center=True).mean()
        ax5.plot(delta_laps, moving_avg, 'b-', linewidth=3, alpha=0.8, label='Trend')
    
    ax5.fill_between(delta_laps, 0, delta_values, 
                     where=np.array(delta_values) >= 0, color='orange', alpha=0.3, 
                     label=f'{driver2} faster')
    ax5.fill_between(delta_laps, 0, delta_values, 
                     where=np.array(delta_values) < 0, color='blue', alpha=0.3, 
                     label=f'{driver1} faster')
    
    ax5.set_xlabel('Lap Number', fontsize=11, fontweight='bold')
    ax5.set_ylabel(f'Delta (seconds)\n{driver1} - {driver2}', fontsize=11, fontweight='bold')
    ax5.set_title('Pace Delta Evolution (Fuel-Corrected)', fontsize=13, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle='--')

# Overall title
fig.suptitle(f'Race Pace Analysis: {drivers_info[driver1]} vs {drivers_info[driver2]}\n' +
            f'{gp} GP {year}',
            fontsize=18, fontweight='bold', y=0.995)

# Save
filename = f'output_race_pace/{gp}_{year}_{driver1}_vs_{driver2}_pace.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: {filename}")

plt.show(block=True)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nKey Findings:")
if overall1 and overall2:
    pace_diff = overall1['avg_corrected'] - overall2['avg_corrected']
    faster = driver1 if pace_diff < 0 else driver2
    print(f"  • {drivers_info[faster]} had better fuel-corrected pace")
    print(f"  • Pace difference: {abs(pace_diff):.3f}s per lap")
    print(f"  • Over full race: {abs(pace_diff) * total_race_laps:.1f}s advantage")
    
    if overall1['std_corrected'] < overall2['std_corrected']:
        print(f"  • {driver1} was more consistent (σ={overall1['std_corrected']:.3f}s)")
    else:
        print(f"  • {driver2} was more consistent (σ={overall2['std_corrected']:.3f}s)")

print(f"\n{'='*70}\n")
