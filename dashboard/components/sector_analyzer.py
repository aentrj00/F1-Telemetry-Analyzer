import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

def run_analysis(year, gp, session, driver1, driver2, sector_length):
    """
    Run Sector Analysis
    """
    with st.spinner(f' Analyzing sectors for {driver1} vs {driver2}...'):
        try:
            # Get script path
            script_dir = Path(__file__).parent.parent.parent / 'scripts'
            script_path = script_dir / 'sector_analysis.py'
            
            if not script_path.exists():
                st.error(f" Script not found: {script_path}")
                return
            
            # Prepare inputs
            inputs = f"{year}\n{gp}\n{session}\n{driver1}\n{driver2}\n{sector_length}\n"
            
            # Run script
            st.info(" Analyzing mini-sectors...")
            st.info(" Processing telemetry data...")
            
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
                st.success(" Sector analysis complete!")
                
                # Display results
                st.subheader(" Results")
                
                # Show output
                with st.expander(" Analysis Output", expanded=True):
                    st.code(stdout, language='text')
                
                # Try to find and display generated image
                output_dir = script_dir.parent / 'output_sector_analysis'
                if output_dir.exists():
                    images = list(output_dir.glob(f'*{gp}*{year}*{session}*{driver1}*{driver2}*.png'))
                    if not images:
                        images = list(output_dir.glob(f'*{gp}*{year}*.png'))
                    
                    if images:
                        st.subheader(" Sector Heatmap & Analysis")
                        latest_image = max(images, key=os.path.getctime)
                        st.image(str(latest_image), use_container_width=True)
                        
                        st.success(" Image saved to output_sector_analysis/")
                
            else:
                st.error(" Analysis failed")
                with st.expander(" Error Details"):
                    st.code(stderr, language='text')
                    
        except subprocess.TimeoutExpired:
            st.error(" Analysis timed out (>3 minutes)")
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.exception(e)
