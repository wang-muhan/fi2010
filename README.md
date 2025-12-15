# AdaMixer: Efficient MLP-Based Architecture for Limit Order Book Forecasting

## Introduction

**AdaMixer**, a novel all-MLP architecture built upon the **TLOB** codebase. **AdaMixer** integrates the **MLP-Mixer** framework with **Adaptive Layer Normalization (AdaLN-Zero)** to effectively handle the non-stationarity of financial data.
## Project Structure

This repository is organized as follows:

- **`config/`**: Contains configuration files using Hydra. `config.py` defines the configuration structure, default parameters, and registers available models and datasets.
- **`data/`**: Directory for storing datasets (e.g., FI-2010) and model checkpoints.
- **`models/`**: Implementation of various Deep Learning models for LOB forecasting.
  - `adaln_mlplob.py`: Implementation of **AdaMixer** (AdaLN + MLPLOB).
  - `mlplob.py`, `tlob.py`, `deeplob.py`, `binctabl.py`, etc.: Implementations of baseline and comparative models.
- **`preprocessing/`**: Scripts for data preprocessing and feature extraction from raw LOB data.
- **`utils/`**: Utility functions for model training, evaluation, and data handling.
- **`main.py`**: The main entry point for training and testing models. It handles configuration via Hydra and initiates the training process.
- **`run.py`**: Contains the core training loops and WandB integration logic.
- **`constants.py`**: Project-wide constants.

## Getting Started

### Installation

```sh
pip install -r requirements.txt
```

> **Note**: If you plan to use muon, please ensure you have the latest version of torch

## Usage

### Training AdaMixer

To train the **AdaMixer** model on the FI-2010 dataset, run the following command:

```sh
python main.py +model=adalnmlplob +dataset=fi_2010 dataset.batch_size=256 experiment.horizon=10,20,50,100 hydra.job.chdir=False --multirun
```

### Training Other Models

  You can train other models by changing the `+model` argument. Available models include:
  - `mlplob`: A pure MLP-Mixer architecture for LOB forecasting.
  - `mlpt`: A hybrid architecture combining MLP with AdaLN conditioning followed by Transformer blocks.
  - `timmlplob`: Implementation of various MLP variants from the `timm` (PyTorch Image Models) library.
  - `tlob`: A Transformer-based model for LOB forecasting.
  - `deeplob`: A deep CNN-LSTM model designed for LOB data.
  - `binctabl`: A model combining BiN normalization with TABL (Temporal Attention-Augmented Bilinear Network) layers.
  - `kanlob`: A model incorporating Kolmogorov-Arnold Networks (KAN) for LOB forecasting.
  - `convlob`: A convolutional network using depthwise separable convolutions.

Example:
```sh
python main.py +model=mlplob +dataset=fi_2010 hydra.job.chdir=False
```

### Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. You can override default parameters directly from the command line.

Check `config/config.py` for all available configuration options.

## Data

The code supports multiple datasets:
- **FI-2010**: Automatically handled. If not preprocessed, it will extract data from zip files in `data/FI_2010/`.