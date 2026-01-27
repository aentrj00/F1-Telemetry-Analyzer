import fastf1 as ff1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os

# ============================================
# CONFIGURATION
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('output_consistency'):
    os.makedirs('output_consistency')

ff1.Cache.enable_cache('cache')

# ============================================
# HELPER FUNCTIONS
# ============================================

def classify_lap_quality(lap_time, fastest_time, is_fastest_lap=False, threshold_fast=1.02, threshold_slow=1.05):
    """
    Classifies lap quality based on time difference from fastest lap
    
    Args:
        lap_time: Time of this lap (seconds)
        fastest_time: Fastest lap time (seconds)
        is_fastest_lap: True if this IS the fastest lap
        threshold_fast: Percentage threshold for "blue/fast" (default 2%)
        threshold_slow: Percentage threshold for "green/average" (default 5%)
    
    Returns:
        'purple': THE fastest lap (only one)
        'blue': Within 2% of fastest (close to purple)
        'green': Within 5% of fastest (good pace)
        'yellow': Within 10% of fastest (average)
        'red': Slower than 10% (traffic/mistake)
        'outlap': Out lap from pits
        'inlap': In lap to pits
    """
    if lap_time is None or fastest_time is None or fastest_time == 0:
        return 'invalid'
    
    # THE fastest lap gets purple
    if is_fastest_lap:
        return 'purple'
    
    percentage = lap_time / fastest_time
    
    if percentage <= threshold_fast:
        return 'blue'  # Close to purple
    elif percentage <= threshold_slow:
        return 'green'
    elif percentage <= 1.10:
        return 'yellow'
    else:
        return 'red'

def detect_sector_performance(lap, fastest_sectors, is_fastest_sector):
    """
    Detects sector performance in a lap
    
    Args:
        lap: FastF1 lap object
        fastest_sectors: Dict with fastest sector times
        is_fastest_sector: Dict indicating if this lap has the fastest sector {1: bool, 2: bool, 3: bool}
    
    Returns:
        List of sector colors ['purple', 'blue', 'green', 'yellow']
    """
    sector_colors = []
    
    for i in [1, 2, 3]:
        sector_col = f'Sector{i}Time'
        
        if sector_col not in lap.index or pd.isna(lap[sector_col]):
            sector_colors.append('invalid')
            continue
        
        sector_time = lap[sector_col].total_seconds()
        fastest_time = fastest_sectors[i]
        
        if fastest_time == 0 or sector_time == 0:
            sector_colors.append('invalid')
            continue
        
        # Check if THIS sector is THE fastest
        if is_fastest_sector.get(i, False):
            sector_colors.append('purple')  # Only THE fastest
        else:
            percentage = sector_time / fastest_time
            
            if percentage <= 1.02:  # Within 2% = blue (close to purple)
                sector_colors.append('blue')
            elif percentage <= 1.05:  # Within 5% = green (good)
                sector_colors.append('green')
            else:
                sector_colors.append('yellow')
    
    return sector_colors

def get_lap_compound_color(compound):
    """
    Returns color for tire compound visualization
    """
    compound_colors = {
        'SOFT': '#FF0000',      # Red
        'MEDIUM': '#FFA500',    # Orange/Yellow
        'HARD': '#FFFFFF',      # White
        'INTERMEDIATE': '#00FF00',  # Green
        'WET': '#0000FF'        # Blue
    }
    
    if pd.isna(compound):
        return '#808080'  # Gray for unknown
    
    return compound_colors.get(compound, '#808080')

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
print("CONSISTENCY HEATMAP ANALYZER")
print("Visualize all laps with color-coded performance")
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

session_type = input("Session (Q/R) [Q]: ").strip().upper()
if not session_type or session_type not in ['Q', 'R']:
    session_type = 'Q'

driver = input("Driver to analyze (e.g., VER): ").strip().upper()

if driver not in drivers_info:
    print("\nError: Invalid driver code")
    exit()

print("\n" + "=" * 70)
print(f"Loading session: {year} - {gp} GP - {session_type}")
print(f"Analyzing: {drivers_info[driver]}")
print("=" * 70)

# Load session
try:
    session = ff1.get_session(int(year), gp, session_type)
    session.load()
    print("\n✓ Session loaded successfully")
except Exception as e:
    print(f"\n✗ Error loading session: {e}")
    exit()

# Get driver laps
try:
    laps = session.laps.pick_drivers(driver)
    laps = laps[laps['LapTime'].notna()]
    
    if len(laps) == 0:
        print(f"\n✗ No valid laps found for {driver}")
        exit()
    
    print(f"✓ Found {len(laps)} valid laps")
except Exception as e:
    print(f"\n✗ Error getting laps: {e}")
    exit()

# ============================================
# ANALYZE LAPS
# ============================================

print("\n" + "=" * 70)
print("ANALYZING LAP CONSISTENCY")
print("=" * 70)

# Get fastest lap
fastest_lap = laps.pick_fastest()
fastest_time = fastest_lap['LapTime'].total_seconds()

print(f"\nFastest lap: #{int(fastest_lap['LapNumber'])} - {fastest_lap['LapTime']}")
print(f"Compound: {fastest_lap['Compound']}")

# Get fastest sectors and identify which laps have them
fastest_sectors = {}
fastest_sector_lap_nums = {}  # Track which lap has the fastest sector

for i in [1, 2, 3]:
    sector_col = f'Sector{i}Time'
    if sector_col in laps.columns:
        sector_times = laps[sector_col].dropna()
        if len(sector_times) > 0:
            fastest_sectors[i] = sector_times.min().total_seconds()
            # Find which lap has this fastest sector
            fastest_idx = sector_times.idxmin()
            fastest_sector_lap_nums[i] = laps.loc[fastest_idx, 'LapNumber']
        else:
            fastest_sectors[i] = 0
            fastest_sector_lap_nums[i] = None
    else:
        fastest_sectors[i] = 0
        fastest_sector_lap_nums[i] = None

# Analyze each lap
lap_analysis = []
fastest_lap_number = fastest_lap['LapNumber']

for idx, lap in laps.iterrows():
    lap_num = lap['LapNumber']
    lap_time = lap['LapTime']
    
    if pd.isna(lap_time):
        continue
    
    lap_time_seconds = lap_time.total_seconds()
    
    # Check if it's pit lap
    is_pit_out = pd.notna(lap['PitOutTime'])
    is_pit_in = pd.notna(lap['PitInTime'])
    
    # Check if this is THE fastest lap
    is_fastest_lap = (lap_num == fastest_lap_number)
    
    # Classify lap quality
    if is_pit_out:
        lap_quality = 'outlap'
    elif is_pit_in:
        lap_quality = 'inlap'
    else:
        lap_quality = classify_lap_quality(lap_time_seconds, fastest_time, is_fastest_lap)
    
    # Check which sectors in this lap are the fastest
    is_fastest_sector = {
        1: (lap_num == fastest_sector_lap_nums[1]) if fastest_sector_lap_nums[1] is not None else False,
        2: (lap_num == fastest_sector_lap_nums[2]) if fastest_sector_lap_nums[2] is not None else False,
        3: (lap_num == fastest_sector_lap_nums[3]) if fastest_sector_lap_nums[3] is not None else False
    }
    
    # Get sector performance
    sector_colors = detect_sector_performance(lap, fastest_sectors, is_fastest_sector)
    
    # Get compound
    compound = lap['Compound'] if 'Compound' in lap.index else None
    
    # Delta to fastest
    delta = lap_time_seconds - fastest_time
    
    lap_analysis.append({
        'lap_number': int(lap_num),
        'lap_time': lap_time_seconds,
        'lap_quality': lap_quality,
        'delta': delta,
        'compound': compound,
        'sector1': sector_colors[0] if len(sector_colors) > 0 else 'invalid',
        'sector2': sector_colors[1] if len(sector_colors) > 1 else 'invalid',
        'sector3': sector_colors[2] if len(sector_colors) > 2 else 'invalid',
        'is_pit_out': is_pit_out,
        'is_pit_in': is_pit_in
    })

analysis_df = pd.DataFrame(lap_analysis)

# Statistics
print(f"\n{'='*70}")
print("LAP QUALITY BREAKDOWN")
print(f"{'='*70}")

quality_counts = analysis_df['lap_quality'].value_counts()

quality_descriptions = {
    'purple': 'Purple (< 2% off fastest)',
    'green': 'Green (2-5% off fastest)',
    'yellow': 'Yellow (5-10% off fastest)',
    'red': 'Red (> 10% off fastest)',
    'outlap': 'Out laps',
    'inlap': 'In laps'
}

for quality, count in quality_counts.items():
    desc = quality_descriptions.get(quality, quality)
    percentage = (count / len(analysis_df)) * 100
    print(f"  {desc:30} {count:3} laps ({percentage:5.1f}%)")

# Purple sectors
print(f"\n{'='*70}")
print("FASTEST SECTORS (Purple)")
print(f"{'='*70}")

purple_s1 = len(analysis_df[analysis_df['sector1'] == 'purple'])
purple_s2 = len(analysis_df[analysis_df['sector2'] == 'purple'])
purple_s3 = len(analysis_df[analysis_df['sector3'] == 'purple'])

blue_s1 = len(analysis_df[analysis_df['sector1'] == 'blue'])
blue_s2 = len(analysis_df[analysis_df['sector2'] == 'blue'])
blue_s3 = len(analysis_df[analysis_df['sector3'] == 'blue'])

print(f"  Sector 1: {purple_s1} purple (THE fastest), {blue_s1} blue (close)")
print(f"  Sector 2: {purple_s2} purple (THE fastest), {blue_s2} blue (close)")
print(f"  Sector 3: {purple_s3} purple (THE fastest), {blue_s3} blue (close)")

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

# Color mappings
quality_colors = {
    'purple': '#9D4EDD',   # Purple - THE fastest
    'blue': '#3B82F6',     # Blue - Close to fastest (<2%)
    'green': '#10B981',    # Green - Good (2-5%)
    'yellow': '#FCD34D',   # Yellow - Average (5-10%)
    'red': '#EF4444',      # Red - Slow (>10%)
    'outlap': '#6B7280',   # Gray
    'inlap': '#374151',    # Dark gray
    'invalid': '#1F2937'   # Very dark gray
}

sector_colors_map = {
    'purple': '#9D4EDD',   # Purple - THE fastest sector
    'blue': '#3B82F6',     # Blue - Close to fastest (<2%)
    'green': '#10B981',    # Green - Good (2-5%)
    'yellow': '#FCD34D',   # Yellow - Average
    'invalid': '#374151'
}

# ========== 1. MAIN HEATMAP - LAP QUALITY ==========
ax1 = fig.add_subplot(gs[0, :])

lap_numbers = analysis_df['lap_number'].values
lap_qualities = analysis_df['lap_quality'].values

# Create bar chart
colors = [quality_colors[q] for q in lap_qualities]
bars = ax1.bar(lap_numbers, np.ones(len(lap_numbers)), color=colors, 
               edgecolor='black', linewidth=1.5, width=0.9)

# Add lap numbers on top (only for purple and blue)
for i, (lap_num, quality) in enumerate(zip(lap_numbers, lap_qualities)):
    if quality in ['purple', 'blue']:
        ax1.text(lap_num, 0.5, str(int(lap_num)), 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax1.set_ylim(0, 1)
ax1.set_xlim(lap_numbers.min() - 0.5, lap_numbers.max() + 0.5)
ax1.set_xlabel('Lap Number', fontsize=13, fontweight='bold')
ax1.set_title(f'Lap Quality Heatmap - {drivers_info[driver]}\n{gp} GP {year} - {session_type}',
             fontsize=16, fontweight='bold', pad=20)
ax1.set_yticks([])
ax1.grid(True, axis='x', alpha=0.3, linestyle='--')

# Legend
legend_elements = [
    mpatches.Patch(color=quality_colors['purple'], label='Purple (THE fastest lap)'),
    mpatches.Patch(color=quality_colors['blue'], label='Blue (< 2% off fastest)'),
    mpatches.Patch(color=quality_colors['green'], label='Green (2-5% off)'),
    mpatches.Patch(color=quality_colors['yellow'], label='Yellow (5-10% off)'),
    mpatches.Patch(color=quality_colors['red'], label='Red (> 10% off)'),
    mpatches.Patch(color=quality_colors['outlap'], label='Out lap'),
    mpatches.Patch(color=quality_colors['inlap'], label='In lap')
]
ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)

# ========== 2. SECTOR HEATMAP ==========
ax2 = fig.add_subplot(gs[1, 0])

sector_data = []
for i, row in analysis_df.iterrows():
    if row['lap_quality'] not in ['outlap', 'inlap']:
        sector_data.append([
            sector_colors_map.get(row['sector1'], '#374151'),
            sector_colors_map.get(row['sector2'], '#374151'),
            sector_colors_map.get(row['sector3'], '#374151')
        ])
    else:
        sector_data.append(['#374151', '#374151', '#374151'])

# Plot sectors as stacked bars
valid_laps = analysis_df[~analysis_df['lap_quality'].isin(['outlap', 'inlap'])]

for sector_idx in range(3):
    sector_colors_list = [sector_colors_map.get(row[f'sector{sector_idx+1}'], '#374151') 
                          for _, row in analysis_df.iterrows()]
    
    ax2.bar(lap_numbers, height=0.33, bottom=sector_idx * 0.33,
           color=sector_colors_list, edgecolor='black', linewidth=0.5, width=0.9)

ax2.set_ylim(0, 1)
ax2.set_xlim(lap_numbers.min() - 0.5, lap_numbers.max() + 0.5)
ax2.set_xlabel('Lap Number', fontsize=11, fontweight='bold')
ax2.set_ylabel('Sectors', fontsize=11, fontweight='bold')
ax2.set_yticks([0.165, 0.495, 0.825])
ax2.set_yticklabels(['S1', 'S2', 'S3'])
ax2.set_title('Sector Performance (Purple = THE Fastest Sector)', fontsize=13, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3, linestyle='--')

# ========== 3. DELTA TO FASTEST ==========
ax3 = fig.add_subplot(gs[1, 1])

# Only plot valid laps (not outlaps/inlaps)
valid_laps = analysis_df[~analysis_df['lap_quality'].isin(['outlap', 'inlap', 'invalid'])]

if len(valid_laps) > 0:
    lap_nums_valid = valid_laps['lap_number'].values
    deltas_valid = valid_laps['delta'].values
    
    # Color by quality
    colors_delta = [quality_colors[q] for q in valid_laps['lap_quality']]
    
    ax3.scatter(lap_nums_valid, deltas_valid, c=colors_delta, s=100, 
               edgecolors='black', linewidth=1.5, zorder=5)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Fastest lap', zorder=1)
    
    # Moving average
    if len(deltas_valid) > 3:
        window = min(5, len(deltas_valid))
        moving_avg = pd.Series(deltas_valid).rolling(window=window, center=True).mean()
        ax3.plot(lap_nums_valid, moving_avg, 'b-', linewidth=2, alpha=0.7, 
                label=f'{window}-lap moving average')
    
    ax3.set_xlabel('Lap Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Delta to Fastest (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Lap Time Delta Evolution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=9)

# ========== 4. COMPOUND USAGE ==========
ax4 = fig.add_subplot(gs[2, :])

compound_data = analysis_df['compound'].values
compound_colors_list = [get_lap_compound_color(c) for c in compound_data]

bars = ax4.bar(lap_numbers, np.ones(len(lap_numbers)), color=compound_colors_list,
              edgecolor='black', linewidth=1.5, width=0.9)

ax4.set_ylim(0, 1)
ax4.set_xlim(lap_numbers.min() - 0.5, lap_numbers.max() + 0.5)
ax4.set_xlabel('Lap Number', fontsize=11, fontweight='bold')
ax4.set_title('Tire Compound Usage', fontsize=13, fontweight='bold')
ax4.set_yticks([])
ax4.grid(True, axis='x', alpha=0.3, linestyle='--')

# Compound legend
unique_compounds = analysis_df['compound'].dropna().unique()
compound_legend = []
for comp in unique_compounds:
    color = get_lap_compound_color(comp)
    compound_legend.append(mpatches.Patch(color=color, label=comp, edgecolor='black'))

if compound_legend:
    ax4.legend(handles=compound_legend, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)

# Save
filename = f'output_consistency/{gp}_{year}_{session_type}_{driver}_consistency.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: {filename}")

plt.show(block=True)

# ============================================
# DETAILED REPORT
# ============================================

print("\n" + "=" * 70)
print("DETAILED CONSISTENCY REPORT")
print("=" * 70)

print(f"\nSession: {gp} GP {year} - {session_type}")
print(f"Driver: {drivers_info[driver]} ({driver})")
print(f"Total laps: {len(analysis_df)}")

print(f"\n{'='*70}")
print("FASTEST LAP")
print(f"{'='*70}")
print(f"  Lap #{int(fastest_lap['LapNumber'])}")
print(f"  Time: {fastest_lap['LapTime']}")
print(f"  Compound: {fastest_lap['Compound']}")

if fastest_sectors[1] > 0:
    print(f"\n  Sector 1: {fastest_sectors[1]:.3f}s")
if fastest_sectors[2] > 0:
    print(f"  Sector 2: {fastest_sectors[2]:.3f}s")
if fastest_sectors[3] > 0:
    print(f"  Sector 3: {fastest_sectors[3]:.3f}s")

print(f"\n{'='*70}")
print("CONSISTENCY METRICS")
print(f"{'='*70}")

valid_laps_df = analysis_df[~analysis_df['lap_quality'].isin(['outlap', 'inlap', 'invalid'])]

if len(valid_laps_df) > 0:
    mean_delta = valid_laps_df['delta'].mean()
    std_delta = valid_laps_df['delta'].std()
    
    # Consistency rating thresholds
    if std_delta < 1:
        consistency_rating = 'Excellent'
    elif std_delta < 2:
        consistency_rating = 'Good'
    elif std_delta < 3:
        consistency_rating = 'Average'
    else:
        consistency_rating = 'Poor'
    
    print(f"\nValid laps analyzed: {len(valid_laps_df)}")
    print(f"Average delta to fastest: +{mean_delta:.3f}s")
    print(f"Standard deviation: {std_delta:.3f}s")
    print(f"\nConsistency rating: {consistency_rating}")
    print(f"  Thresholds:")
    print(f"    < 0.20s = Excellent")
    print(f"    0.20-0.40s = Good")
    print(f"    0.40-0.60s = Average")
    print(f"    > 0.60s = Poor")

print(f"\n{'='*70}")
print("PROGRESSION ANALYSIS")
print(f"{'='*70}")

# If qualifying, analyze Q1/Q2/Q3 progression
if session_type == 'Q':
    print("\nQualifying segments:")
    
    # Approximate Q1/Q2/Q3 by lap groups
    total_laps = len(valid_laps_df)
    
    if total_laps > 0:
        q1_laps = valid_laps_df[valid_laps_df['lap_number'] <= valid_laps_df['lap_number'].quantile(0.4)]
        q2_laps = valid_laps_df[(valid_laps_df['lap_number'] > valid_laps_df['lap_number'].quantile(0.4)) & 
                                (valid_laps_df['lap_number'] <= valid_laps_df['lap_number'].quantile(0.7))]
        q3_laps = valid_laps_df[valid_laps_df['lap_number'] > valid_laps_df['lap_number'].quantile(0.7)]
        
        if len(q1_laps) > 0:
            print(f"  Q1 (approx): Best +{q1_laps['delta'].min():.3f}s, Avg +{q1_laps['delta'].mean():.3f}s")
        if len(q2_laps) > 0:
            print(f"  Q2 (approx): Best +{q2_laps['delta'].min():.3f}s, Avg +{q2_laps['delta'].mean():.3f}s")
        if len(q3_laps) > 0:
            print(f"  Q3 (approx): Best +{q3_laps['delta'].min():.3f}s, Avg +{q3_laps['delta'].mean():.3f}s")

# Race pace progression
if session_type == 'R':
    print("\nRace pace by phase:")
    
    race_length = len(valid_laps_df)
    
    if race_length > 0:
        phase1 = valid_laps_df[valid_laps_df['lap_number'] <= valid_laps_df['lap_number'].quantile(0.33)]
        phase2 = valid_laps_df[(valid_laps_df['lap_number'] > valid_laps_df['lap_number'].quantile(0.33)) & 
                               (valid_laps_df['lap_number'] <= valid_laps_df['lap_number'].quantile(0.66))]
        phase3 = valid_laps_df[valid_laps_df['lap_number'] > valid_laps_df['lap_number'].quantile(0.66)]
        
        if len(phase1) > 0:
            print(f"  Early (laps {int(phase1['lap_number'].min())}-{int(phase1['lap_number'].max())}): Avg +{phase1['delta'].mean():.3f}s")
        if len(phase2) > 0:
            print(f"  Mid (laps {int(phase2['lap_number'].min())}-{int(phase2['lap_number'].max())}):   Avg +{phase2['delta'].mean():.3f}s")
        if len(phase3) > 0:
            print(f"  Late (laps {int(phase3['lap_number'].min())}-{int(phase3['lap_number'].max())}):  Avg +{phase3['delta'].mean():.3f}s")

print(f"\n{'='*70}")
print("TOP 5 BEST LAPS")
print(f"{'='*70}")

top5 = valid_laps_df.nsmallest(5, 'delta')

print(f"\n{'Lap':>4} | {'Time':>10} | {'Delta':>8} | {'Compound':^10} | {'Quality':^10}")
print("-" * 70)

for _, lap in top5.iterrows():
    lap_num = int(lap['lap_number'])
    delta_str = f"+{lap['delta']:.3f}s"
    compound = lap['compound'] if pd.notna(lap['compound']) else 'N/A'
    quality = lap['lap_quality']
    
    print(f"{lap_num:4} | {lap['lap_time']:>10.3f} | {delta_str:>8} | {compound:^10} | {quality:^10}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}\n")
