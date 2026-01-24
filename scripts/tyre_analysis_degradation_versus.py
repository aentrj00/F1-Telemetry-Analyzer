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
if not os.path.exists('output_tire_degradation'):
    os.makedirs('output_tire_degradation')

ff1.Cache.enable_cache('cache')

# ============================================
# FUNCTIONS
# ============================================

def detect_stints(laps):
    """
    Detects different stints (runs with same tire)
    Each pit stop = new stint
    """
    stints = []
    current_stint = []
    
    for idx, lap in laps.iterrows():
        if len(current_stint) == 0:
            current_stint = [lap]
        elif pd.notna(lap['PitInTime']):
            if len(current_stint) > 3:  # Minimum 3 laps to be valid
                stints.append(pd.DataFrame(current_stint))
            current_stint = []
        else:
            current_stint.append(lap)
    
    if len(current_stint) > 3:
        stints.append(pd.DataFrame(current_stint))
    
    return stints

def calculate_degradation(stint_df):
    """
    Calculates degradation rate of a stint
    Uses linear regression: lap_time = a + b * tire_age
    """
    lap_times = stint_df['LapTime'].dt.total_seconds().values
    
    # Remove outliers (very slow laps due to traffic, etc)
    mean_time = np.mean(lap_times)
    std_time = np.std(lap_times)
    mask = np.abs(lap_times - mean_time) < 2 * std_time
    
    lap_times_clean = lap_times[mask]
    tire_age = np.arange(1, len(lap_times) + 1)[mask]
    
    if len(lap_times_clean) < 3:
        return None
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(tire_age, lap_times_clean)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'lap_times': lap_times_clean,
        'tire_age': tire_age,
        'compound': stint_df.iloc[0]['Compound'] if 'Compound' in stint_df.columns else 'Unknown',
        'num_laps': len(tire_age)
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
print("TIRE DEGRADATION ANALYSIS")
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

print("\n" + "=" * 70)
print(f"Loading race: {year} - {gp} GP")
print("=" * 70)

# Load session (RACE ONLY)
try:
    session = ff1.get_session(int(year), gp, 'R')
    session.load()
    print("Race loaded\n")
except Exception as e:
    print(f"\nError: {e}")
    exit()

# ============================================
# ANALYZE DRIVER 1
# ============================================

print(f"Analyzing {drivers_info[driver1]}...")
print("-" * 70)

laps1 = session.laps.pick_drivers(driver1)
laps1 = laps1[laps1['LapTime'].notna()]

stints1 = detect_stints(laps1)
print(f"  Stints detected: {len(stints1)}")

degradation1 = []
for idx, stint in enumerate(stints1, 1):
    deg = calculate_degradation(stint)
    if deg:
        degradation1.append(deg)
        print(f"  Stint {idx}: {deg['num_laps']} laps | "
              f"Compound: {deg['compound']} | "
              f"Degradation: {deg['slope']:.4f} s/lap | "
              f"R²: {deg['r_squared']:.3f}")

# ============================================
# ANALYZE DRIVER 2
# ============================================

print(f"\nAnalyzing {drivers_info[driver2]}...")
print("-" * 70)

laps2 = session.laps.pick_drivers(driver2)
laps2 = laps2[laps2['LapTime'].notna()]

stints2 = detect_stints(laps2)
print(f"  Stints detected: {len(stints2)}")

degradation2 = []
for idx, stint in enumerate(stints2, 1):
    deg = calculate_degradation(stint)
    if deg:
        degradation2.append(deg)
        print(f"  Stint {idx}: {deg['num_laps']} laps | "
              f"Compound: {deg['compound']} | "
              f"Degradation: {deg['slope']:.4f} s/lap | "
              f"R²: {deg['r_squared']:.3f}")

# ============================================
# CREATE ENHANCED GRAPHS
# ============================================

print("\n" + "=" * 70)
print("Generating enhanced graphs...")
print("=" * 70)

# Colors by compound
compound_colors = {
    'SOFT': 'red',
    'MEDIUM': 'yellow',
    'HARD': 'white',
    'INTERMEDIATE': 'green',
    'WET': 'blue',
    'Unknown': 'gray'
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# ========== DRIVER 1 ==========

# Calculate y-axis range with extra margins
all_times_1 = []
for deg in degradation1:
    all_times_1.extend(deg['lap_times'])

if all_times_1:
    y_min_1 = min(all_times_1)
    y_max_1 = max(all_times_1)
    y_range_1 = y_max_1 - y_min_1
    
    # Add margins
    y_axis_min_1 = y_min_1 - y_range_1 * 0.10
    y_axis_max_1 = y_max_1 + y_range_1 * (0.15 + 0.10 * len(degradation1))
else:
    y_axis_min_1 = 60
    y_axis_max_1 = 120
    y_range_1 = 60

for idx, deg in enumerate(degradation1, 1):
    color = compound_colors.get(deg['compound'], 'gray')
    edge_color = 'black' if color in ['yellow', 'white'] else color
    
    # ALL POINTS ARE CIRCLES
    ax1.scatter(deg['tire_age'], deg['lap_times'],
               c=color, edgecolors=edge_color, s=150, alpha=0.7,
               marker='o', linewidth=2, zorder=5,
               label=f"Stint {idx} ({deg['compound']})")
    
    # Add stint number label INSIDE each point
    for i, (age, time) in enumerate(zip(deg['tire_age'], deg['lap_times'])):
        ax1.text(age, time, f"S{idx}", fontsize=8, ha='center', va='center',
                fontweight='bold', color='white' if color == 'red' else 'black',
                zorder=6)
    
    # Regression line
    x_line = np.array([deg['tire_age'].min(), deg['tire_age'].max()])
    y_line = deg['slope'] * x_line + deg['intercept']
    line_color = 'darkred' if deg['slope'] > 0.025 else 'darkgreen' if deg['slope'] < 0.015 else 'orange'
    ax1.plot(x_line, y_line, '--', color=line_color, linewidth=3, alpha=0.8, zorder=4)
    
    # Arrow showing degradation direction
    start_y = deg['intercept'] + deg['slope'] * deg['tire_age'].min()
    end_y = deg['intercept'] + deg['slope'] * deg['tire_age'].max()
    
    ax1.annotate('', xy=(deg['tire_age'].max(), end_y), 
                xytext=(deg['tire_age'].min(), start_y),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=line_color, alpha=0.7))
    
    # Position info box ABOVE the data
    vertical_offset = y_max_1 + (y_range_1 * 0.08) + (idx - 1) * (y_range_1 * 0.10)
    mid_x = np.mean(deg['tire_age'])
    
    deg_category = "HIGH DEG" if deg['slope'] > 0.025 else "NORMAL" if deg['slope'] > 0.015 else "LOW DEG"
    
    ax1.text(mid_x, vertical_offset, 
            f"Stint {idx}: {deg['slope']:.4f} s/lap\n{deg_category}",
            fontsize=9, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=line_color, linewidth=2.5, alpha=0.95))

ax1.set_xlabel('Tire Age (laps)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Lap Time (seconds)', fontsize=13, fontweight='bold')
ax1.set_title(f'{drivers_info[driver1]} - Tire Degradation\n' +
             'Arrow up ↗ = Tires degrading (getting slower)',
             fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')

# Set y-limits
ax1.set_ylim(y_axis_min_1, y_axis_max_1)

# ========== DRIVER 2 ==========

all_times_2 = []
for deg in degradation2:
    all_times_2.extend(deg['lap_times'])

if all_times_2:
    y_min_2 = min(all_times_2)
    y_max_2 = max(all_times_2)
    y_range_2 = y_max_2 - y_min_2
    
    y_axis_min_2 = y_min_2 - y_range_2 * 0.10
    y_axis_max_2 = y_max_2 + y_range_2 * (0.15 + 0.10 * len(degradation2))
else:
    y_axis_min_2 = 60
    y_axis_max_2 = 120
    y_range_2 = 60

for idx, deg in enumerate(degradation2, 1):
    color = compound_colors.get(deg['compound'], 'gray')
    edge_color = 'black' if color in ['yellow', 'white'] else color
    
    ax2.scatter(deg['tire_age'], deg['lap_times'],
               c=color, edgecolors=edge_color, s=150, alpha=0.7,
               marker='o', linewidth=2, zorder=5,
               label=f"Stint {idx} ({deg['compound']})")
    
    for i, (age, time) in enumerate(zip(deg['tire_age'], deg['lap_times'])):
        ax2.text(age, time, f"S{idx}", fontsize=8, ha='center', va='center',
                fontweight='bold', color='white' if color == 'red' else 'black',
                zorder=6)
    
    x_line = np.array([deg['tire_age'].min(), deg['tire_age'].max()])
    y_line = deg['slope'] * x_line + deg['intercept']
    line_color = 'darkred' if deg['slope'] > 0.025 else 'darkgreen' if deg['slope'] < 0.015 else 'orange'
    ax2.plot(x_line, y_line, '--', color=line_color, linewidth=3, alpha=0.8, zorder=4)
    
    start_y = deg['intercept'] + deg['slope'] * deg['tire_age'].min()
    end_y = deg['intercept'] + deg['slope'] * deg['tire_age'].max()
    
    ax2.annotate('', xy=(deg['tire_age'].max(), end_y), 
                xytext=(deg['tire_age'].min(), start_y),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=line_color, alpha=0.7))
    
    vertical_offset = y_max_2 + (y_range_2 * 0.08) + (idx - 1) * (y_range_2 * 0.10)
    mid_x = np.mean(deg['tire_age'])
    
    deg_category = "HIGH DEG" if deg['slope'] > 0.025 else "NORMAL" if deg['slope'] > 0.015 else "LOW DEG"
    
    ax2.text(mid_x, vertical_offset, 
            f"Stint {idx}: {deg['slope']:.4f} s/lap\n{deg_category}",
            fontsize=9, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=line_color, linewidth=2.5, alpha=0.95))

ax2.set_xlabel('Tire Age (laps)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Lap Time (seconds)', fontsize=13, fontweight='bold')
ax2.set_title(f'{drivers_info[driver2]} - Tire Degradation\n' +
             'Arrow up ↗ = Tires degrading (getting slower)',
             fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')

ax2.set_ylim(y_axis_min_2, y_axis_max_2)

# ============================================
# LEGENDS AND EXPLANATIONS - ALL OUTSIDE PLOTS
# ============================================

# Main title with more space
fig.suptitle(f'Tire Degradation Analysis - {gp} GP {year}\n' +
             f'{drivers_info[driver1]} vs {drivers_info[driver2]}',
             fontsize=17, fontweight='bold', y=0.97)

# Create space at bottom for legends
plt.subplots_adjust(bottom=0.18, top=0.93, left=0.06, right=0.98, wspace=0.15)

# ===== DRIVER 1 LEGEND - Below plot, horizontal =====
legend1 = ax1.legend(loc='upper center', fontsize=9, framealpha=0.98,
                    bbox_to_anchor=(0.5, -0.12), ncol=min(len(degradation1), 3),
                    fancybox=True, shadow=True, title="Stints",
                    title_fontsize=10)

# ===== DRIVER 2 LEGEND - Below plot, horizontal =====
legend2 = ax2.legend(loc='upper center', fontsize=9, framealpha=0.98,
                    bbox_to_anchor=(0.5, -0.12), ncol=min(len(degradation2), 3),
                    fancybox=True, shadow=True, title="Stints",
                    title_fontsize=10)

# ===== EXPLANATION BOX - Centered below BOTH plots =====
explanation_text = (
    "HOW TO READ:\n"
    "• Each circle = 1 lap  |  S1/S2/S3 = Stint number  |  Circle shape is always the same\n"
    "• Dashed line = Degradation trend  |  Arrow ↗ pointing up = Tires degrading (slower)\n"
    "• Arrow → horizontal/flat = Tires staying stable"
)

fig.text(0.5, 0.04, explanation_text,
        ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                 edgecolor='black', linewidth=2, alpha=0.95))

# Save
filename = f'output_tire_degradation/{gp}_{year}_degradation_{driver1}_vs_{driver2}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nGraph saved: {filename}")

plt.show(block=True)

# ============================================
# COMPARATIVE SUMMARY
# ============================================

print("\n" + "=" * 70)
print("COMPARATIVE SUMMARY")
print("=" * 70)

if degradation1 and degradation2:
    avg_deg1 = np.mean([d['slope'] for d in degradation1])
    avg_deg2 = np.mean([d['slope'] for d in degradation2])
    
    print(f"\n{driver1} ({drivers_info[driver1]}):")
    print(f"  Average degradation: {avg_deg1:.4f} s/lap")
    print(f"  Stints analyzed: {len(degradation1)}")
    for idx, deg in enumerate(degradation1, 1):
        print(f"    Stint {idx}: {deg['slope']:.4f} s/lap ({deg['compound']})")
    
    print(f"\n{driver2} ({drivers_info[driver2]}):")
    print(f"  Average degradation: {avg_deg2:.4f} s/lap")
    print(f"  Stints analyzed: {len(degradation2)}")
    for idx, deg in enumerate(degradation2, 1):
        print(f"    Stint {idx}: {deg['slope']:.4f} s/lap ({deg['compound']})")
    
    diff = abs(avg_deg1 - avg_deg2)
    better = driver1 if avg_deg1 < avg_deg2 else driver2
    
    print(f"\n{'='*70}")
    print(f"Better tire management: {drivers_info[better]}")
    print(f"Difference: {diff:.4f} s/lap")
    print(f"\nOver 20 laps, this represents: {diff * 20:.2f} seconds")
    
    print(f"\n{'='*70}")
    print("DEGRADATION CATEGORIES:")
    print("  < 0.015 s/lap  = LOW (excellent tire management)")
    print("  0.015-0.025    = NORMAL (typical degradation)")
    print("  > 0.025 s/lap  = HIGH (struggling with tires)")

print("\n" + "=" * 70)
print("HOW TO INTERPRET")
print("=" * 70)
print("  • Slope = seconds lost per lap as tire ages")
print("  • Lower slope = better tire management")
print("  • R² close to 1.0 = very consistent degradation")
print("  • R² low = high variation (traffic, mistakes, etc)")
print("  • Upward arrow = tire is degrading")
print("  • Flat/horizontal arrow = tire staying stable")
print("\n" + "=" * 70)