# improve-figure-reporting

A small demo project that generates a publication-quality scatter plot of **porosity vs. permeability** for four synthetic reservoir formations (Sand A, Sand B, Shaly Sand, Tight Zone). The figure uses a clean, minimal style with a log-scaled permeability axis and a colorblind-friendly palette.

The project contains two equivalent implementations:

| Script | Data stack |
|---|---|
| [main.py](main.py) | NumPy + pandas |
| [main_pl.py](main_pl.py) | NumPy (`ndarray`) + Polars |

Both scripts synthesize the dataset (80 samples per formation), render the figure with matplotlib, and save it to `porosity_permeability.png` (300 DPI).

A Streamlit dashboard ([app.py](app.py)) is also included: it lets you pick an implementation, runs the selected script, and displays the generated figure in the browser with a download button.

## Requirements

- Python >= 3.14
- Dependencies: `matplotlib`, `pandas` (for `main.py`), `polars` (for `main_pl.py`), `streamlit` (for `app.py`)

## Setup

Using [uv](https://docs.astral.sh/uv/) (recommended):

```powershell
uv sync
uv add polars streamlit   # polars for main_pl.py, streamlit for app.py
```

Or with pip:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install matplotlib pandas polars streamlit
```

## Run

```powershell
# pandas version
python main.py

# polars version
python main_pl.py
```

Or with uv:

```powershell
uv run main.py
uv run main_pl.py
```

### Streamlit dashboard

```powershell
streamlit run app.py
# or
uv run streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501), choose an implementation, and click **Generate figure**.

## Lint & format

The project uses [Ruff](https://docs.astral.sh/ruff/) (installed as a dev dependency):

```powershell
uv run ruff check .          # lint
uv run ruff check . --fix    # lint with auto-fixes
uv run ruff format .         # format
uv run ruff format --check . # verify formatting
```

## Output

Each run writes `porosity_permeability.png` to the working directory — a scatter plot of porosity (v/v) vs. permeability (mD, log scale), grouped and colored by formation.
