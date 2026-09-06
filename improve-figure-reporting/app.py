import subprocess
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
OUTPUT_PNG = HERE / "porosity_permeability.png"

SCRIPTS = {
    "Polars (main_pl.py)": HERE / "main_pl.py",
    "pandas (main.py)": HERE / "main.py",
}

st.set_page_config(page_title="Porosity vs Permeability", layout="centered")
st.title("Porosity vs Permeability by Formation")
st.caption(
    "Generate the figure by running one of the plotting scripts, then view the result below."
)

choice = st.selectbox("Implementation", options=list(SCRIPTS.keys()))

if st.button("Generate figure", type="primary"):
    script = SCRIPTS[choice]
    with st.spinner(f"Running {script.name}..."):
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode == 0:
        st.success(f"{script.name} completed successfully.")
        if result.stdout.strip():
            st.code(result.stdout, language="text")
    else:
        st.error(f"{script.name} failed with exit code {result.returncode}.")
        st.code(result.stderr or result.stdout, language="text")

if OUTPUT_PNG.exists():
    st.image(str(OUTPUT_PNG), caption="porosity_permeability.png")
    with open(OUTPUT_PNG, "rb") as f:
        st.download_button(
            "Download PNG",
            data=f,
            file_name="porosity_permeability.png",
            mime="image/png",
        )
else:
    st.info("No figure generated yet. Click **Generate figure** to run the script.")
