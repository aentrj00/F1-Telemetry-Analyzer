import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

def run_analysis(year, gp, driver1, driver2, fuel_effect):
    """
    Run Race Pace Analysis with fuel correction
    """
    with st.spinner(f' Analyzing race pace for {driver1} vs {driver2}...'):
        try:
            # Get script path
            script_dir = Path(__file__).parent.parent.parent / 'scripts'
            script_path = script_dir / 'race_pace_analyzer.py'
            
            if not script_path.exists():
                st.error(f" Script not found: {script_path}")
                return
            
            # Prepare inputs
            inputs = f"{year}\n{gp}\n{driver1}\n{driver2}\n{fuel_effect}\n"
            
            # Run script
            st.info(" Analyzing race pace...")
            st.info(" Applying fuel correction...")
            st.info(" Calculating undercut potential...")
            
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(script_dir.parent)
            )
            
            stdout, stderr = process.communicate(input=inputs, timeout=180)
            
            if process.returncode == 0:
                st.success(" Race pace analysis complete!")
                
                # Display results
                st.subheader(" Results")
                
                # Parse key metrics
                lines = stdout.split('\n')
                
                col1, col2, col3 = st.columns(3)
                
                for line in lines:
                    if "fuel-corrected pace:" in line.lower() and driver1 in line:
                        with col1:
                            pace = line.split(':')[-1].strip()
                            st.metric(f"{driver1} Fuel-Corrected", pace)
                    elif "fuel-corrected pace:" in line.lower() and driver2 in line:
                        with col2:
                            pace = line.split(':')[-1].strip()
                            st.metric(f"{driver2} Fuel-Corrected", pace)
                    elif "faster by" in line.lower():
                        with col3:
                            delta = line.split('by')[-1].strip()
                            st.metric("Pace Advantage", delta)
                
                # Show full output
                with st.expander(" Detailed Analysis", expanded=True):
                    st.code(stdout, language='text')
                
                # Try to find and display generated image
                output_dir = script_dir.parent / 'output_race_pace'
                if output_dir.exists():
                    images = list(output_dir.glob(f'*{gp}*{year}*{driver1}*{driver2}*.png'))
                    if not images:
                        images = list(output_dir.glob(f'*{gp}*{year}*.png'))
                    
                    if images:
                        st.subheader(" Pace Analysis Visualization")
                        latest_image = max(images, key=os.path.getctime)
                        st.image(str(latest_image), use_container_width=True)
                        
                        st.success(" Image saved to output_race_pace/")
                
            else:
                st.error(" Analysis failed")
                with st.expander(" Error Details"):
                    st.code(stderr, language='text')
                    
        except subprocess.TimeoutExpired:
            st.error(" Analysis timed out (>3 minutes)")
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.exception(e)
