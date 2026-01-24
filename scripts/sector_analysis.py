import fastf1 as ff1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import os
from scipy import interpolate

# ============================================
# CONFIGURATION
# ============================================

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('output_sector_analysis'):
    os.makedirs('output_sector_analysis')

ff1.Cache.enable_cache('cache')

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_fastest_lap_telemetry(session, driver):
    """
    Gets telemetry from the fastest lap of a driver
    """
    try:
        laps = session.laps.pick_drivers(driver)
        fastest_lap = laps.pick_fastest()
        
        if fastest_lap is None or fastest_lap.empty:
            return None
        
        telemetry = fastest_lap.get_telemetry()
        
        if telemetry is None or len(telemetry) == 0:
            return None
            
        return telemetry
        
    except Exception as e:
        print(f"  ✗ Error getting telemetry for {driver}: {str(e)[:50]}")
        return None

def create_mini_sectors(telemetry, sector_length=100):
    """
    Divides circuit into mini-sectors of specified length (meters)
    
    Args:
        telemetry: FastF1 telemetry data
        sector_length: Length of each mini-sector in meters
    
    Returns:
        DataFrame with mini-sector information
    """
    # Add distance if not present
    if 'Distance' not in telemetry.columns:
        telemetry = telemetry.add_distance()
    
    total_distance = telemetry['Distance'].max()
    num_sectors = int(total_distance / sector_length) + 1
    
    sectors = []
    
    for i in range(num_sectors):
        start_dist = i * sector_length
        end_dist = (i + 1) * sector_length
        
        # Get telemetry in this sector
        sector_data = telemetry[
            (telemetry['Distance'] >= start_dist) & 
            (telemetry['Distance'] < end_dist)
        ]
        
        if len(sector_data) == 0:
            continue
        
        # Calculate sector metrics
        min_speed = sector_data['Speed'].min()
        max_speed = sector_data['Speed'].max()
        avg_speed = sector_data['Speed'].mean()
        
        # Check if it's a braking zone (significant speed drop)
        speed_drop = max_speed - min_speed
        is_corner = speed_drop > 50  # > 50 km/h drop = corner
        
        # Get position data
        try:
            x_pos = sector_data['X'].mean()
            y_pos = sector_data['Y'].mean()
        except:
            x_pos = 0
            y_pos = 0
        
        sectors.append({
            'sector_num': i,
            'start_distance': start_dist,
            'end_distance': end_dist,
            'min_speed': min_speed,
            'max_speed': max_speed,
            'avg_speed': avg_speed,
            'speed_drop': speed_drop,
            'is_corner': is_corner,
            'x_pos': x_pos,
            'y_pos': y_pos
        })
    
    return pd.DataFrame(sectors)

def calculate_time_delta(sectors_d1, sectors_d2, sector_length=100):
    """
    Calculates time difference between two drivers in each sector
    
    Returns:
        DataFrame with sector comparisons and cumulative delta
    """
    comparisons = []
    cumulative_delta = 0
    
    for idx, sector1 in sectors_d1.iterrows():
        sector_num = sector1['sector_num']
        
        # Find corresponding sector for driver 2
        sector2 = sectors_d2[sectors_d2['sector_num'] == sector_num]
        
        if sector2.empty:
            continue
        
        sector2 = sector2.iloc[0]
        
        # Calculate time spent in sector (simplified)
        # Time = Distance / Average Speed
        # Using minimum speed for corners (more representative)
        speed1 = sector1['min_speed'] if sector1['is_corner'] else sector1['avg_speed']
        speed2 = sector2['min_speed'] if sector2['is_corner'] else sector2['avg_speed']
        
        # Avoid division by zero
        if speed1 == 0 or speed2 == 0:
            continue
        
        # Convert km/h to m/s and calculate time
        time1 = sector_length / (speed1 / 3.6)
        time2 = sector_length / (speed2 / 3.6)
        
        delta = time1 - time2  # Positive = Driver 1 slower
        cumulative_delta += delta
        
        comparisons.append({
            'sector_num': sector_num,
            'start_distance': sector1['start_distance'],
            'end_distance': sector1['end_distance'],
            'driver1_speed': speed1,
            'driver2_speed': speed2,
            'delta': delta,
            'cumulative_delta': cumulative_delta,
            'is_corner': sector1['is_corner'],
            'x_pos': sector1['x_pos'],
            'y_pos': sector1['y_pos'],
            'faster_driver': 1 if delta < 0 else 2  # 1 if driver1 faster
        })
    
    return pd.DataFrame(comparisons)

def find_significant_corners(sector_comparisons, min_delta_threshold=0.05):
    """
    Identifies corners where there's significant time difference
    
    Args:
        sector_comparisons: DataFrame from calculate_time_delta
        min_delta_threshold: Minimum time difference to be considered significant (seconds)
    
    Returns:
        List of significant corners with their details
    """
    # Only look at corners
    corners = sector_comparisons[sector_comparisons['is_corner']].copy()
    
    # Find corners with significant deltas
    corners['abs_delta'] = corners['delta'].abs()
    significant = corners[corners['abs_delta'] >= min_delta_threshold].copy()
    
    # Sort by absolute delta (most significant first)
    significant = significant.sort_values('abs_delta', ascending=False)
    
    return significant

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
print("SECTOR-BY-SECTOR ANALYSIS")
print("Compare driver performance across circuit sections")
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

print("\n" + "=" * 70)
driver1 = input("Driver 1 (e.g., VER): ").strip().upper()
driver2 = input("Driver 2 (e.g., NOR): ").strip().upper()

if driver1 not in drivers_info or driver2 not in drivers_info:
    print("\nError: Invalid driver code")
    exit()

if driver1 == driver2:
    print("\nError: Choose different drivers")
    exit()

sector_length = input("\nMini-sector length in meters [100]: ").strip()
if not sector_length:
    sector_length = 100
else:
    sector_length = int(sector_length)

print("\n" + "=" * 70)
print(f"Loading session: {year} - {gp} GP - {session_type}")
print(f"Comparing: {drivers_info[driver1]} vs {drivers_info[driver2]}")
print(f"Sector length: {sector_length}m")
print("=" * 70)

# Load session
try:
    session = ff1.get_session(int(year), gp, session_type)
    session.load()
    print("\n✓ Session loaded successfully")
except Exception as e:
    print(f"\n✗ Error loading session: {e}")
    exit()

# Get telemetry
print("\n" + "=" * 70)
print("STEP 1: LOADING TELEMETRY DATA")
print("=" * 70)

print(f"\nLoading fastest lap for {driver1}...")
tel1 = get_fastest_lap_telemetry(session, driver1)

if tel1 is None:
    print(f"✗ Could not get telemetry for {driver1}")
    exit()

print(f"✓ Loaded {len(tel1)} telemetry points")

print(f"\nLoading fastest lap for {driver2}...")
tel2 = get_fastest_lap_telemetry(session, driver2)

if tel2 is None:
    print(f"✗ Could not get telemetry for {driver2}")
    exit()

print(f"✓ Loaded {len(tel2)} telemetry points")

# Create mini-sectors
print("\n" + "=" * 70)
print("STEP 2: CREATING MINI-SECTORS")
print("=" * 70)

print(f"\nDividing circuit into {sector_length}m sectors...")

sectors1 = create_mini_sectors(tel1, sector_length)
sectors2 = create_mini_sectors(tel2, sector_length)

print(f"✓ Created {len(sectors1)} mini-sectors for {driver1}")
print(f"✓ Created {len(sectors2)} mini-sectors for {driver2}")

# Calculate deltas
print("\n" + "=" * 70)
print("STEP 3: CALCULATING TIME DIFFERENCES")
print("=" * 70)

comparisons = calculate_time_delta(sectors1, sectors2, sector_length)

if comparisons.empty:
    print("✗ Could not calculate sector comparisons")
    exit()

total_delta = comparisons['cumulative_delta'].iloc[-1]
faster_driver = driver1 if total_delta < 0 else driver2
delta_abs = abs(total_delta)

print(f"\n✓ Analyzed {len(comparisons)} sectors")
print(f"\nFINAL RESULT:")
print(f"  {drivers_info[faster_driver]} is faster by {delta_abs:.3f}s")

# Find significant corners
significant_corners = find_significant_corners(comparisons, min_delta_threshold=0.03)

print(f"\n✓ Found {len(significant_corners)} significant corners (Δ > 0.03s)")

if len(significant_corners) > 0:
    print(f"\nTOP 5 BIGGEST DIFFERENCES:")
    for idx, corner in significant_corners.head(5).iterrows():
        faster = driver1 if corner['delta'] < 0 else driver2
        delta_val = abs(corner['delta'])
        distance = corner['start_distance']
        print(f"  {distance:5.0f}m: {drivers_info[faster]:20} +{delta_val:.3f}s")

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "=" * 70)
print("STEP 4: GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

# ========== 1. CIRCUIT HEATMAP ==========
ax1 = fig.add_subplot(gs[0, :])

# Get position data
tel1_with_dist = tel1.add_distance() if 'Distance' not in tel1.columns else tel1
tel2_with_dist = tel2.add_distance() if 'Distance' not in tel2.columns else tel2

x1 = tel1_with_dist['X'].values
y1 = tel1_with_dist['Y'].values
x2 = tel2_with_dist['X'].values
y2 = tel2_with_dist['Y'].values

# Interpolate to same number of points for comparison
num_points = min(len(x1), len(x2))
x_interp = np.linspace(0, 1, num_points)

f_x1 = interpolate.interp1d(np.linspace(0, 1, len(x1)), x1)
f_y1 = interpolate.interp1d(np.linspace(0, 1, len(y1)), y1)
f_x2 = interpolate.interp1d(np.linspace(0, 1, len(x2)), x2)
f_y2 = interpolate.interp1d(np.linspace(0, 1, len(y2)), y2)

x_new = f_x1(x_interp)
y_new = f_y1(x_interp)

# Create color map based on delta
# Interpolate delta values to match circuit points
distances = comparisons['start_distance'].values
deltas = comparisons['delta'].values

# Interpolate deltas to circuit points
circuit_distances = tel1_with_dist['Distance'].values
delta_interp = np.interp(circuit_distances, distances, deltas)

# Normalize for colormap
delta_max = max(abs(delta_interp.min()), abs(delta_interp.max()))
delta_normalized = delta_interp / delta_max if delta_max > 0 else delta_interp

# Create line segments
points = np.array([x_new, y_new]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Sample delta values for segments
delta_segments = delta_normalized[:len(segments)]

# Create LineCollection
lc = LineCollection(segments, cmap='RdYlGn_r', linewidth=5)
lc.set_array(delta_segments)
lc.set_clim(-1, 1)

ax1.add_collection(lc)
ax1.set_xlim(x_new.min() - 100, x_new.max() + 100)
ax1.set_ylim(y_new.min() - 100, y_new.max() + 100)
ax1.set_aspect('equal')

# Add colorbar
cbar = plt.colorbar(lc, ax=ax1, orientation='horizontal', pad=0.05, aspect=50)
cbar.set_label(f'{driver1} faster ← → {driver2} faster', fontsize=12, fontweight='bold')

# Mark significant corners
for idx, corner in significant_corners.head(10).iterrows():
    if corner['x_pos'] != 0 and corner['y_pos'] != 0:
        marker_color = 'green' if corner['delta'] < 0 else 'red'
        marker_size = min(abs(corner['delta']) * 500, 200)
        ax1.scatter(corner['x_pos'], corner['y_pos'], 
                   s=marker_size, c=marker_color, alpha=0.6, 
                   edgecolors='black', linewidth=2, zorder=10)

ax1.set_title(f'Circuit Heatmap: {drivers_info[driver1]} vs {drivers_info[driver2]}\n' +
             f'{gp} GP {year} - {session_type}',
             fontsize=16, fontweight='bold', pad=20)
ax1.axis('off')

# ========== 2. CUMULATIVE DELTA GRAPH ==========
ax2 = fig.add_subplot(gs[1, :])

distances_plot = comparisons['start_distance'].values / 1000  # Convert to km
cumulative = comparisons['cumulative_delta'].values

ax2.plot(distances_plot, cumulative, linewidth=3, color='blue', label='Cumulative delta')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.fill_between(distances_plot, 0, cumulative, 
                 where=(cumulative >= 0), color='red', alpha=0.3, label=f'{driver2} faster')
ax2.fill_between(distances_plot, 0, cumulative, 
                 where=(cumulative < 0), color='green', alpha=0.3, label=f'{driver1} faster')

# Mark significant corners
for idx, corner in significant_corners.head(5).iterrows():
    dist_km = corner['start_distance'] / 1000
    cum_delta_at_corner = comparisons[comparisons['sector_num'] == corner['sector_num']]['cumulative_delta'].values[0]
    ax2.scatter(dist_km, cum_delta_at_corner, s=100, c='yellow', 
               edgecolors='black', linewidth=2, zorder=10)
    ax2.text(dist_km, cum_delta_at_corner + 0.02, f"{abs(corner['delta']):.2f}s", 
            fontsize=8, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax2.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Time Delta (s)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Time Difference Throughout Lap', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=10)

# ========== 3. SPEED COMPARISON IN CORNERS ==========
ax3 = fig.add_subplot(gs[2, 0])

corners_only = comparisons[comparisons['is_corner']].head(15)
corner_indices = np.arange(len(corners_only))

width = 0.35
speeds1 = corners_only['driver1_speed'].values
speeds2 = corners_only['driver2_speed'].values

bars1 = ax3.bar(corner_indices - width/2, speeds1, width, 
               label=driver1, color='blue', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(corner_indices + width/2, speeds2, width, 
               label=driver2, color='orange', alpha=0.7, edgecolor='black')

ax3.set_xlabel('Corner Number', fontsize=11, fontweight='bold')
ax3.set_ylabel('Minimum Speed (km/h)', fontsize=11, fontweight='bold')
ax3.set_title('Corner Speed Comparison (First 15 Corners)', fontsize=13, fontweight='bold')
ax3.set_xticks(corner_indices)
ax3.set_xticklabels([f"C{i+1}" for i in corner_indices], fontsize=9)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# ========== 4. SECTOR DELTA DISTRIBUTION ==========
ax4 = fig.add_subplot(gs[2, 1])

all_deltas = comparisons['delta'].values * 1000  # Convert to milliseconds

ax4.hist(all_deltas, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Equal pace')
ax4.axvline(x=np.mean(all_deltas), color='yellow', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(all_deltas):.1f}ms')

ax4.set_xlabel('Time Delta (ms)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Sectors', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Sector Time Differences', fontsize=13, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add statistics text
faster_sectors_d1 = len(comparisons[comparisons['delta'] < 0])
faster_sectors_d2 = len(comparisons[comparisons['delta'] > 0])
stats_text = f"{driver1}: {faster_sectors_d1} sectors faster\n{driver2}: {faster_sectors_d2} sectors faster"
ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save
filename = f'output_sector_analysis/{gp}_{year}_{session_type}_{driver1}_vs_{driver2}_sectors.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: {filename}")

plt.show(block=True)

# ============================================
# DETAILED REPORT
# ============================================

print("\n" + "=" * 70)
print("DETAILED SECTOR ANALYSIS REPORT")
print("=" * 70)

print(f"\nSession: {gp} GP {year} - {session_type}")
print(f"Comparison: {drivers_info[driver1]} vs {drivers_info[driver2]}")
print(f"Sector length: {sector_length}m")

print(f"\n{'='*70}")
print("OVERALL PERFORMANCE")
print(f"{'='*70}")

print(f"\nTotal time difference: {delta_abs:.3f}s")
print(f"Faster driver: {drivers_info[faster_driver]}")
print(f"Sectors analyzed: {len(comparisons)}")
print(f"  {driver1} faster in: {faster_sectors_d1} sectors ({faster_sectors_d1/len(comparisons)*100:.1f}%)")
print(f"  {driver2} faster in: {faster_sectors_d2} sectors ({faster_sectors_d2/len(comparisons)*100:.1f}%)")

print(f"\n{'='*70}")
print("TOP 10 SIGNIFICANT CORNERS")
print(f"{'='*70}")

print(f"\n{'Distance':>8} | {'Faster Driver':^20} | {'Delta':>10} | {'Corner Speed Diff':^20}")
print("-" * 70)

for idx, corner in significant_corners.head(10).iterrows():
    faster = driver1 if corner['delta'] < 0 else driver2
    delta_val = abs(corner['delta'])
    distance = corner['start_distance']
    speed_diff = abs(corner['driver1_speed'] - corner['driver2_speed'])
    
    print(f"{distance:7.0f}m | {drivers_info[faster]:^20} | {delta_val:>9.3f}s | {speed_diff:>8.1f} km/h")

print(f"\n{'='*70}")
print("SECTOR-BY-SECTOR BREAKDOWN (Every 500m)")
print(f"{'='*70}")

print(f"\n{'Distance':>10} | {driver1:^10} | {driver2:^10} | {'Delta':>8} | {'Cumulative':>11} | {'Faster':^10}")
print("-" * 70)

# Show every 5th sector (approximately every 500m if sector_length=100)
step = max(1, 500 // sector_length)
for idx, row in comparisons.iloc[::step].iterrows():
    dist = row['start_distance']
    speed1 = row['driver1_speed']
    speed2 = row['driver2_speed']
    delta = row['delta']
    cum_delta = row['cumulative_delta']
    faster = driver1 if delta < 0 else driver2
    
    delta_str = f"{delta:+.3f}s"
    cum_str = f"{cum_delta:+.3f}s"
    
    print(f"{dist:9.0f}m | {speed1:>8.1f} | {speed2:>8.1f} | {delta_str:>8} | {cum_str:>11} | {faster:^10}")

print(f"\n{'='*70}")
print("ANALYSIS SUMMARY")
print(f"{'='*70}")

# Calculate where each driver is strongest
corners_d1_faster = significant_corners[significant_corners['delta'] < 0]
corners_d2_faster = significant_corners[significant_corners['delta'] > 0]

if len(corners_d1_faster) > 0:
    avg_gain_d1 = corners_d1_faster['delta'].abs().mean()
    print(f"\n{driver1} ({drivers_info[driver1]}):")
    print(f"  Strongest corners: {len(corners_d1_faster)}")
    print(f"  Average gain: {avg_gain_d1:.3f}s per corner")
    print(f"  Total advantage: {corners_d1_faster['delta'].sum():.3f}s")

if len(corners_d2_faster) > 0:
    avg_gain_d2 = corners_d2_faster['delta'].abs().mean()
    print(f"\n{driver2} ({drivers_info[driver2]}):")
    print(f"  Strongest corners: {len(corners_d2_faster)}")
    print(f"  Average gain: {avg_gain_d2:.3f}s per corner")
    print(f"  Total advantage: {corners_d2_faster['delta'].abs().sum():.3f}s")

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

if delta_abs < 0.1:
    print("\n  → Very close performance - drivers are evenly matched")
elif delta_abs < 0.3:
    print(f"\n  → {drivers_info[faster_driver]} has slight advantage")
    print(f"     Could be driver skill, car setup, or track conditions")
else:
    print(f"\n  → {drivers_info[faster_driver]} has clear advantage")
    print(f"     Significant performance gap of {delta_abs:.3f}s")

if len(significant_corners) > 5:
    print(f"\n  → Multiple corners show differences (> 0.03s)")
    print(f"     Suggests fundamental car setup or driving style differences")
elif len(significant_corners) > 0:
    print(f"\n  → Few specific corners show differences")
    print(f"     Likely specific corner characteristics (e.g., traction, braking)")
else:
    print(f"\n  → No major corner differences found")
    print(f"     Time gained/lost distributed evenly across lap")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}\n")
