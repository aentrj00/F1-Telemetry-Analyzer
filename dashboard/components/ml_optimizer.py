import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

def run_analysis(year, gp, driver, train_all, race_only):
    """
    Run ML Strategy Optimizer analysis
    """
    with st.spinner(f' Loading session data for {gp} {year}...'):
        try:
            # Get script path
            script_dir = Path(__file__).parent.parent.parent / 'scripts'
            script_path = script_dir / 'tyre_degradation_ml.py'
            
            if not script_path.exists():
                st.error(f" Script not found: {script_path}")
                st.info(" Make sure your scripts are in the '../scripts/' directory")
                return
            
            # Prepare inputs
            inputs = f"{year}\n{gp}\n{driver}\n"
            inputs += "Y\n" if train_all else "N\n"
            inputs += "Y\n" if race_only else "N\n"
            
            # Run script
            st.info(" Training ML model...")
            st.info(" This may take 2-4 minutes...")
            
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(script_dir.parent)
            )
            
            stdout, stderr = process.communicate(input=inputs, timeout=900)
            
            if process.returncode == 0:
                st.success(" Analysis complete!")
                
                # Display results
                st.subheader(" Results")
                
                # Parse output for key metrics
                lines = stdout.split('\n')
                
                # Extract key information
                best_strategy = None
                prediction_error = None
                
                for i, line in enumerate(lines):
                    if "BEST STRATEGY" in line:
                        best_strategy = i
                    if "Prediction error" in line or "error:" in line.lower():
                        st.metric("Prediction Error", line.split(':')[-1].strip())
                
                # Show output in expander
                with st.expander(" Full Analysis Output", expanded=True):
                    st.code(stdout, language='text')
                
                # Try to find and display generated image
                output_dir = script_dir.parent / 'output_tire_ml'
                if output_dir.exists():
                    images = list(output_dir.glob(f'*{gp}*{year}*{driver}*.png'))
                    if images:
                        st.subheader(" Visualization")
                        latest_image = max(images, key=os.path.getctime)
                        st.image(str(latest_image), use_container_width=True)
                    else:
                        st.info(" No visualization found. Check output_tire_ml/ directory")
                
            else:
                st.error(" Analysis failed")
                with st.expander(" Error Details"):
                    st.code(stderr, language='text')
                    
        except subprocess.TimeoutExpired:
            st.error(" Analysis timed out (>5 minutes)")
            st.info(" Try with a different Grand Prix or year")
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.exception(e)
