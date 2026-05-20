# Python ML Tutorial

A small TensorFlow tutorial project using the Fashion MNIST dataset.

This project includes:
- Data exploration and image visualization
- Model training and evaluation
- Basic prediction inspection

## Project Structure

- `look_at_data.py`: Loads Fashion MNIST and visualizes sample images.
- `train_model_and_test.py`: Trains a neural network, evaluates it, and prints predictions.
- `pyproject.toml`: Project metadata and dependencies.
- `uv.lock`: Locked dependency versions for reproducible installs with `uv`.

## Requirements

- Python 3.12+
- `pip` (or `uv`, recommended)

## Dependencies

Defined in `pyproject.toml`:
- tensorflow
- matplotlib

## Setup

### Option 1: Setup with uv (recommended)

```bash
# Install uv if needed: https://docs.astral.sh/uv/
uv sync
```

Run commands using uv:

```bash
uv run python look_at_data.py
uv run python train_model_and_test.py
```

### Option 2: Setup with venv + pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow matplotlib
```

## Run

After environment setup, run either script:

```bash
python look_at_data.py
```

What it does:
- Prints TensorFlow and dataset details
- Displays a grid of sample Fashion MNIST images

```bash
python train_model_and_test.py
```

What it does:
- Trains a feed-forward neural network on Fashion MNIST
- Evaluates test accuracy
- Prints predicted classes for sample test images

## Notes

- The first run may download the Fashion MNIST dataset.
- Training can take a while depending on your CPU/GPU.
- If TensorFlow installation fails, verify your Python version and platform compatibility.
