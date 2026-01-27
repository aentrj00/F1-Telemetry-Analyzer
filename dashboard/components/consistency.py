import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

def run_analysis(year, gp, session, driver):
    """
    Run Consistency Heatmap Analysis
    """
    with st.spinner(f' Analyzing consistency for {driver}...'):
        try:
            # Get script path
            script_dir = Path(__file__).parent.parent.parent / 'scripts'
            script_path = script_dir / 'consistency_heatmap.py'
            
            if not script_path.exists():
                st.error(f" Script not found: {script_path}")
                return
            
            # Prepare inputs
            inputs = f"{year}\n{gp}\n{session}\n{driver}\n"
            
            # Run script
            st.info(" Analyzing lap consistency...")
            st.info(" Detecting purple sectors...")
            st.info(" Calculating consistency metrics...")
            
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
                st.success(" Consistency analysis complete!")
                
                # Display results
                st.subheader(" Results")
                
                # Parse key metrics
                lines = stdout.split('\n')
                
                col1, col2, col3 = st.columns(3)
                
                for line in lines:
                    if "Consistency rating:" in line:
                        with col1:
                            rating = line.split(':')[-1].strip()
                            st.metric("Consistency Rating", rating)
                    elif "Standard deviation:" in line:
                        with col2:
                            std = line.split(':')[-1].strip()
                            st.metric("Std Deviation", std)
                    elif "Valid laps analyzed:" in line:
                        with col3:
                            laps = line.split(':')[-1].strip()
                            st.metric("Valid Laps", laps)
                
                # Show lap quality breakdown
                st.subheader(" Lap Quality Breakdown")
                
                quality_section = False
                quality_lines = []
                
                for line in lines:
                    if "LAP QUALITY BREAKDOWN" in line:
                        quality_section = True
                    elif quality_section and "=====" in line:
                        quality_section = False
                    elif quality_section and line.strip():
                        quality_lines.append(line)
                
                if quality_lines:
                    for line in quality_lines[2:]:  # Skip first 2 lines (headers)
                        if line.strip():
                            st.text(line)
                
                # Show full output
                with st.expander(" Detailed Analysis", expanded=False):
                    st.code(stdout, language='text')
                
                # Try to find and display generated image
                output_dir = script_dir.parent / 'output_consistency'
                if output_dir.exists():
                    images = list(output_dir.glob(f'*{gp}*{year}*{session}*{driver}*.png'))
                    if not images:
                        images = list(output_dir.glob(f'*{gp}*{year}*.png'))
                    
                    if images:
                        st.subheader(" Consistency Heatmap")
                        latest_image = max(images, key=os.path.getctime)
                        st.image(str(latest_image), use_container_width=True)
                        
                        st.info(" Purple = THE fastest lap | Blue = <2% off | Green = 2-5% off")
                        st.success(" Image saved to output_consistency/")
                
            else:
                st.error(" Analysis failed")
                with st.expander(" Error Details"):
                    st.code(stderr, language='text')
                    
        except subprocess.TimeoutExpired:
            st.error(" Analysis timed out (>2 minutes)")
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.exception(e)
