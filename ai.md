# Guide: Adding a New Model to the TLOB Framework

This guide outlines the systematic process for integrating a new deep learning model into the TLOB project. Follow these steps to ensure seamless integration with the existing Hydra configuration, training engine, and model registry.

## 1. Implement the Model Architecture

Create a new Python file in the `models/` directory (e.g., `models/your_new_model.py`).

*   **Requirements**:
    *   Define your model as a `torch.nn.Module`.
    *   Ensure the `__init__` method accepts parameters consistent with the project's conventions (e.g., `hidden_dim`, `num_layers`, `seq_size`, `num_features`, `dataset_type`).
    *   The `forward` method should handle input tensors of shape `(Batch, Seq, Features)` or handle `dataset_type` specific logic if necessary (e.g., splitting features for LOBSTER, though this is legacy).
    *   **Important**: Maintain consistent input/output dimensions to ensure compatibility with the `Engine` and `DataLoaders`.

## 2. Register the Model Type

Update `constants.py` to include a unique identifier for your new model.

*   **File**: `constants.py`
*   **Action**: Add a new entry to the `ModelType` Enum.

```python
class ModelType(Enum):
    # ... existing models
    YOUR_NEW_MODEL = "YOUR_NEW_MODEL"
```

## 3. Define Configuration

Create a structured configuration for your model using `dataclasses` and `hydra`.

*   **File**: `config/config.py`
*   **Action**:
    1.  Define a new dataclass inheriting from `Model`.
    2.  Specify `hyperparameters_fixed` with default values.
    3.  Specify `hyperparameters_sweep` for hyperparameter tuning ranges.
    4.  Set the `type` field to the Enum value defined in Step 2.
    5.  Register the dataclass in the `ConfigStore` at the bottom of the file.

```python
@dataclass
class YourNewModelConfig(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        "num_layers": 4, 
        "hidden_dim": 64, 
        "lr": 0.0001, 
        # ... other params
    })
    # ... sweep params
    type: ModelType = ModelType.YOUR_NEW_MODEL

# ... at the bottom of config.py
cs.store(group="model", name="your_new_model", node=YourNewModelConfig)
```

## 4. Update Model Factory

Enable the instantiation of your model in the utility factory function.

*   **File**: `utils/utils_model.py`
*   **Action**:
    1.  Import your new model class.
    2.  Update the `pick_model` function to handle the new model string identifier (matching the Enum value).

```python
from models.your_new_model import YourNewModel

def pick_model(model_type, ...):
    # ... existing checks
    elif model_type == "YOUR_NEW_MODEL":
        return YourNewModel(...)
```

## 5. Integrate with Training Engine

Update the main execution logic to initialize the `Engine` with your new model type.

*   **File**: `run.py`
*   **Action**:
    1.  In the `train` function, locate the model initialization block (chain of `if/elif` statements checking `model_type`).
    2.  Add a new branch for `cst.ModelType.YOUR_NEW_MODEL`.
    3.  Instantiate the `Engine` class, passing the relevant hyperparameters from `config`.

```python
elif model_type == cst.ModelType.YOUR_NEW_MODEL:
    model = Engine(
        # ... standard arguments
        model_type=config.model.type.value,
        # ... pass specific params from config.model.hyperparameters_fixed
    )
```

## 6. Training

Run the training command using the name registered in the `ConfigStore` (Step 3).

```bash
PYTHONPATH=. python main.py +model=your_new_model +dataset=fi_2010 hydra.job.chdir=False
```

## FI-2010 Segmented Sampling (no cross-stock leakage)

Purpose: sample sequences that stay inside a single stock segment and optionally return a stock id alongside labels.

What changed
- `constants.py`: added per-file segment metadata (`FI2010_TRAIN`, `FI2010_TEST`) with start/end.
- `preprocessing/fi_2010.py`: `fi_2010_load(..., enforce_segments=False)`; when set to True it also returns indices/stock_ids that respect segment boundaries.
- `preprocessing/dataset.py`: `Dataset` accepts optional `indices` (valid start positions) and `stock_labels`; if provided, `__getitem__` returns `(x, y, stock_id)`, otherwise legacy `(x, y)`.

How to use (no crossing segments)
```python
from preprocessing.fi_2010 import fi_2010_load
from preprocessing.dataset import Dataset
import constants as cst

res = fi_2010_load(
    path=cst.DATA_DIR + "/FI_2010",
    seq_size=seq_size,
    horizon=horizon,
    all_features=True,
    enforce_segments=True,  # key flag
)
(train_x, train_y, val_x, val_y, test_x, test_y,
 train_idx, val_idx, test_idx,
 train_sid, val_sid, test_sid) = res

train_set = Dataset(train_x, train_y, seq_size, indices=train_idx, stock_labels=train_sid)
val_set   = Dataset(val_x,   val_y,   seq_size, indices=val_idx,   stock_labels=val_sid)
test_set  = Dataset(test_x,  test_y,  seq_size, indices=test_idx,  stock_labels=test_sid)
```

Notes
- Backward compatible: with `enforce_segments=False`, `fi_2010_load` returns the original 6-tuple and `Dataset` behaves unchanged.
- Indices only include starts where a full `seq_size` window fits inside a segment, so windows cannot cross segments.
- If your training_step expects `(x, y)` only, either keep `enforce_segments=False` or ignore the third returned item when using `stock_labels`.

## Muon Optimizer Integration

We have integrated the Muon optimizer (from `torch.optim.Muon`) to improve training efficiency, especially for models with large 2D weight matrices (like Transformers and MLPs).

### Implementation Details

1.  **Wrapper Class (`models/engine.py`)**: Created `MuonAdamW`, a wrapper that combines `torch.optim.Muon` and `torch.optim.AdamW`.
    *   **Muon Group**: Applied to 2D parameters (weights of Linear layers) in hidden layers.
    *   **AdamW Group**: Applied to embeddings, output heads, and all 1D parameters (biases, normalization weights).
2.  **Configuration (`config/config.py`)**: Added Muon-specific hyperparameters to model configs (e.g., `ADALNMLPLOB`).
    *   `muon_lr`: Learning rate for Muon (typically ~0.02, higher than AdamW).
    *   `muon_momentum`: Momentum for Muon (default 0.95).
    *   `muon_weight_decay`: Weight decay for Muon (default 0.01).
3.  **Engine Logic**: `configure_optimizers` selects `MuonAdamW` if `self.optimizer == "Muon"`.

### How to Use

To use Muon, set the optimizer to "Muon" in your configuration or command line.

**1. Config File (`config/config.py`)**

To enable Muon parameters for a specific model (e.g., `ADALNMLPLOB`), add them to `hyperparameters_fixed`:

```python
@dataclass
class ADALNMLPLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        # ... standard params
        "lr": 0.0005,          # AdamW learning rate (for non-Muon params)
        "muon_lr": 0.02,       # Muon learning rate (main learning rate)
        "muon_momentum": 0.95,
        "muon_weight_decay": 0.01
    })
    # ...
```

**2. Running from Command Line**

You can switch between AdamW and Muon dynamically.

*   **Run with AdamW (Default behavior)**:
    ```bash
    python main.py +model=adalnmlplob experiment.optimizer=AdamW
    ```

*   **Run with Muon**:
    ```bash
    python main.py +model=adalnmlplob experiment.optimizer=Muon
    ```

*   **Run with Muon and Custom Hyperparameters**:
    ```bash
    python main.py +model=adalnmlplob experiment.optimizer=Muon \
        model.hyperparameters_fixed.muon_lr=0.05 \
        model.hyperparameters_fixed.muon_momentum=0.9
    ```
