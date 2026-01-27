"""
Fix Unicode characters in Python scripts for Windows compatibility
Replaces special Unicode characters with ASCII alternatives
"""

import os
import re

# Characters to replace
REPLACEMENTS = {
    # Arrows
    '\u2192': '->',  # →
    '\u2190': '<-',  # ←
    '\u2191': '^',   # ↑
    '\u2193': 'v',   # ↓
    '\u21D2': '=>',  # ⇒
    '\u21D0': '<=',  # ⇐
    
    # Checkmarks and crosses
    '\u2713': '[SUCCESS]',     # ✓
    '\u2714': '[SUCCESS]',     # ✔
    '\u2717': '[ERROR]',      # ✗
    '\u2718': '[ERROR]',      # ✘
    '\u2716': '[ERROR]',      # ✖
    
    # Mathematical symbols
    '\u2264': '<=',       # ≤
    '\u2265': '>=',       # ≥
    '\u2260': '!=',       # ≠
    '\u00B1': '+/-',      # ±
    '\u00D7': 'x',        # ×
    '\u00F7': '/',        # ÷
    '\u2248': '~',        # ≈
    '\u221E': 'inf',      # ∞
    
    # Symbols
    '\u2022': '*',        # •
    '\u2605': '[*]',      # ★ (filled star)
    '\u2606': '[ ]',      # ☆ (empty star)
    '\u26A0': '[!]',      # ⚠
    '\u2139': '[i]',      # ℹ
    '\u2705': '[SUCCESS]',     # ✅
    '\u274C': '[ERROR]',      # ❌
    '\u231B': '[Wait]',   # ⌛
    '\u23F3': '[Wait]',   # ⏳
    
    # Greek letters (common in stats)
    '\u03C3': 'sigma',    # σ
    '\u03BC': 'mu',       # μ
    '\u0394': 'Delta',    # Δ
    '\u03B1': 'alpha',    # α
    '\u03B2': 'beta',     # β
    
    # Emojis (remove or replace)
    '\U0001F3CE\uFE0F': '[RACING]',  # 🏎️
    '\U0001F3CE': '[RACING]',         # 🏎
    '\U0001F4CA': '[Chart]',      # 📊
    '\U0001F4C8': '[Graph]',      # 📈
    '\U0001F50D': '[Search]',     # 🔍
    '\U0001F4A1': '[Tip]',        # 💡
    '\U0001F525': '[Fire]',       # 🔥
    '\U0001F680': '[Rocket]',     # 🚀
    '\U0001F3C1': '[Flag]',       # 🏁
    '\U0001F4BB': '[PC]',         # 💻
    '\U0001F6A8': '[Alert]',      # 🚨
    '\U0001F4C5': '[Calendar]',   # 📅
    '\U0001F4C2': '[Folder]',     # 📂
    '\U0001F4BE': '[Disk]',       # 💾
}

def fix_file(filepath):
    """Fix Unicode characters in a single file"""
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Replace all problematic characters
        for unicode_char, replacement in REPLACEMENTS.items():
            content = content.replace(unicode_char, replacement)
        
        # Check if anything changed
        if content != original:
            # Backup original
            backup_path = filepath + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)
            
            # Write fixed version
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[OK] Fixed: {filepath}")
            print(f"     Backup saved: {backup_path}")
            return True
        else:
            print(f"[OK] No changes needed: {filepath}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all Python files in scripts directory"""
    print("=" * 70)
    print("UNICODE CHARACTER FIX FOR WINDOWS")
    print("=" * 70)
    print()
    
    # Get scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    scripts_dir = os.path.join(parent_dir, 'scripts')
    
    if not os.path.exists(scripts_dir):
        print(f"[ERROR] Scripts directory not found: {scripts_dir}")
        print()
        print("Please run this script from the dashboard/ directory")
        input("Press Enter to exit...")
        return
    
    print(f"Scripts directory: {scripts_dir}")
    print()
    
    # List of scripts to fix
    scripts_to_fix = [
        'tyre_degradation_ml.py',
        'tyre_analysis_degradation_versus.py',
        'sector_analysis.py',
        'race_pace_analyzer.py',
        'consistency_heatmap.py'
    ]
    
    fixed_count = 0
    
    for script_name in scripts_to_fix:
        filepath = os.path.join(scripts_dir, script_name)
        
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"[!] Not found: {script_name}")
    
    print()
    print("=" * 70)
    print(f"SUMMARY: Fixed {fixed_count} file(s)")
    print("=" * 70)
    print()
    
    if fixed_count > 0:
        print("Backups saved with .backup extension")
        print("You can delete backups after confirming everything works")
    
    print()
    input("Press Enter to exit...")

if __name__ == '__main__':
    main()
