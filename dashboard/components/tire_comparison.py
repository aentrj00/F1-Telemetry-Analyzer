import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

def run_analysis(year, gp, driver1, driver2):
    """
    Run Tire Degradation Comparison analysis
    """
    with st.spinner(f'🔄 Analyzing tire degradation for {driver1} vs {driver2}...'):
        try:
            # Get script path
            script_dir = Path(__file__).parent.parent.parent / 'scripts'
            script_path = script_dir / 'tyre_analysis_degradation_versus.py'
            
            if not script_path.exists():
                st.error(f" Script not found: {script_path}")
                return
            
            # Prepare inputs
            inputs = f"{year}\n{gp}\n{driver1}\n{driver2}\n"
            
            # Run script
            st.info(" Comparing tire degradation...")
            
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(script_dir.parent)
            )
            
            stdout, stderr = process.communicate(input=inputs, timeout=120)
            
            if process.returncode == 0:
                st.success(" Analysis complete!")
                
                # Display results
                st.subheader(" Results")
                
                # Show output
                with st.expander(" Analysis Output", expanded=True):
                    st.code(stdout, language='text')
                
                # Try to find and display generated image
                output_dir = script_dir.parent / 'output_tire_analysis'
                if output_dir.exists():
                    images = list(output_dir.glob(f'*{gp}*{year}*{driver1}*{driver2}*.png'))
                    if not images:
                        images = list(output_dir.glob(f'*{gp}*{year}*.png'))
                    
                    if images:
                        st.subheader(" Visualization")
                        latest_image = max(images, key=os.path.getctime)
                        st.image(str(latest_image), use_container_width=True)
                
            else:
                st.error(" Analysis failed")
                with st.expander("🔍 Error Details"):
                    st.code(stderr, language='text')
                    
        except subprocess.TimeoutExpired:
            st.error(" Analysis timed out (>2 minutes)")
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.exception(e)
