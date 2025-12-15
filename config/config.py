from typing import List
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from constants import DatasetType, ModelType, SamplingType
from omegaconf import MISSING, OmegaConf


@dataclass
class Model:
    hyperparameters_fixed: dict = MISSING
    hyperparameters_sweep: dict = MISSING
    type: ModelType = MISSING
    
@dataclass
class MLPLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 3, "hidden_dim": 64, "lr": 0.0005, "seq_size": 64, "all_features": True})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [3, 6], "hidden_dim": [128], "lr": [0.0003], "seq_size": [384]})
    horizon_to_seq_size: dict = field(default_factory=lambda: {
        10: 16,
        20: 16,
        50: 16,
        100: 16
    })
    type: ModelType = ModelType.MLPLOB

@dataclass
class KANLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 1, "hidden_dim": 40, "lr": 0.0003, "seq_size": 64, "all_features": True})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [3, 6], "hidden_dim": [128], "lr": [0.0003], "seq_size": [384]})
    type: ModelType = ModelType.KANLOB
    
@dataclass
class TLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 4, "hidden_dim": 40, "num_heads": 1, "is_sin_emb": True, "lr": 0.0001, "seq_size": 32, "all_features": True, "use_pos_in_attn": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [4, 6], "hidden_dim": [128, 256], "num_heads": [1], "is_sin_emb": [True], "lr": [0.0001], "seq_size": [128], "use_pos_in_attn": [False]})
    type: ModelType = ModelType.TLOB

@dataclass
class TLOBClsToken(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 4, "hidden_dim": 40, "num_heads": 1, "is_sin_emb": True, "lr": 0.0001, "seq_size": 128, "all_features": True, "dropout": 0.1, "mlp_ratio": 4.0, "use_pos_in_attn": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [4, 6], "hidden_dim": [128, 256], "num_heads": [1], "is_sin_emb": [True], "lr": [0.0001], "seq_size": [128], "use_pos_in_attn": [False]})
    type: ModelType = ModelType.TLOB_CLS
    
@dataclass
class GLALOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 4, "hidden_dim": 40, "num_heads": 1, "is_sin_emb": True, "lr": 0.0001, "seq_size": 16, "all_features": True})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [4, 6], "hidden_dim": [128, 256], "num_heads": [1], "is_sin_emb": [True], "lr": [0.0001], "seq_size": [128]})
    type: ModelType = ModelType.GLALOB

@dataclass
class GLALOBClsToken(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 4, "hidden_dim": 40, "num_heads": 1, "is_sin_emb": True, "lr": 0.0001, "seq_size": 128, "all_features": True, "dropout": 0.1, "mlp_ratio": 4.0})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [4, 6], "hidden_dim": [128, 256], "num_heads": [1], "is_sin_emb": [True], "lr": [0.0001], "seq_size": [128]})
    type: ModelType = ModelType.GLALOB_CLS
    
    
@dataclass
class BiNCTABL(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"lr": 0.001, "seq_size": 10, "all_features": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"lr": [0.001], "seq_size": [10]})
    type: ModelType = ModelType.BINCTABL

@dataclass
class DeepLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"lr": 0.01, "seq_size": 100, "all_features": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"lr": [0.01], "seq_size": [100]})
    type: ModelType = ModelType.DEEPLOB


@dataclass
class ADALNMLPLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        "num_layers": 3, 
        "hidden_dim": 64, 
        "lr": 0.0005, 
        "seq_size": 64, 
        "all_features": True, 
        "dropout": 0.0,
        "use_dyt": False,
        # "muon_lr": 0.02,
        # "muon_momentum": 0.95,
        # "muon_weight_decay": 0.01
    })
    hyperparameters_sweep: dict = field(default_factory=lambda: {
        "num_layers": [4, 6], 
        "hidden_dim": [128, 256], 
        "lr": [0.0005, 0.001], 
        "seq_size": [64, 128],
        "use_dyt": [True, False]
    })
    horizon_to_seq_size: dict = field(default_factory=lambda: {
        10: 4,
        20: 16,
        50: 16,
        100: 16
    })
    type: ModelType = ModelType.ADALNMLPLOB

@dataclass
class MLPT(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_mlp_layers": 3, "num_trans_layers": 1, "hidden_dim": 64, "num_heads": 1, "is_sin_emb": True, "lr": 0.0005, "seq_size": 16, "all_features": True})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_mlp_layers": [4, 6], "num_trans_layers": [4, 6], "hidden_dim": [128, 256], "num_heads": [1], "is_sin_emb": [True], "lr": [0.0005, 0.001], "seq_size": [64, 128]})
    type: ModelType = ModelType.MLPT

@dataclass
class TIMMLPLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 3, "hidden_dim": 64, "lr": 0.0005, "seq_size": 64, "all_features": True, "variant": "Mlp"})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [3, 6], "hidden_dim": [128], "lr": [0.0003], "seq_size": [384], "variant": ["Mlp", "GluMlp", "SwiGLU", "GatedMlp", "ConvMlp", "GlobalResponseNormMlp"]})
    type: ModelType = ModelType.TIMMLPLOB

@dataclass
class ConvLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {
        "num_layers": 3, 
        "hidden_dim": 64, 
        "lr": 0.0005, 
        "seq_size": 64, 
        "all_features": True,
        "kernel_size": 32,
        "dropout": 0.0
    })
    hyperparameters_sweep: dict = field(default_factory=lambda: {
        "num_layers": [3, 6], 
        "hidden_dim": [64, 128], 
        "lr": [0.0005, 0.001], 
        "seq_size": [64, 128],
        "kernel_size": [16, 32]
    })
    type: ModelType = ModelType.CONVLOB
        
@dataclass
class Dataset:
    type: DatasetType = MISSING
    dates: list = MISSING
    batch_size: int = MISSING

@dataclass
class FI_2010(Dataset):
    type: DatasetType = DatasetType.FI_2010
    dates: list = field(default_factory=lambda: ["2010-01-01", "2010-12-31"])
    batch_size: int = 256

@dataclass
class Experiment:
    is_data_preprocessed: bool = False
    is_wandb: bool = False
    is_sweep: bool = False
    type: list = field(default_factory=lambda: ["TRAINING"])
    is_debug: bool = False
    checkpoint_reference: str = ""
    seed: int = 1
    horizon: int = 100
    max_epochs: int = 20
    dir_ckpt: str = "model.ckpt"
    optimizer: str = "AdamW"
    use_class_weight: bool = False
    lr_scheduler_type: str = "plateau" # "plateau" or "cosine_warmup"
    warmup_epochs: int = 1
    label_smoothing: float = 0.7
    save_attn_score: bool = True

    
defaults = [Model, Experiment, Dataset]

@dataclass
class Config:
    model: Model
    dataset: Dataset
    experiment: Experiment = field(default_factory=Experiment)
    defaults: List = field(default_factory=lambda: [
        {"hydra/job_logging": "disabled"},
        {"hydra/hydra_logging": "disabled"},
        "_self_"
    ])
    
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="mlplob", node=MLPLOB)
cs.store(group="model", name="tlob", node=TLOB)
cs.store(group="model", name="tlob_cls", node=TLOBClsToken)
cs.store(group="model", name="glalob", node=GLALOB)
cs.store(group="model", name="glalob_cls", node=GLALOBClsToken)
cs.store(group="model", name="binctabl", node=BiNCTABL)
cs.store(group="model", name="deeplob", node=DeepLOB)
cs.store(group="model", name="kanlob", node=KANLOB)
cs.store(group="model", name="adalnmlplob", node=ADALNMLPLOB)
cs.store(group="model", name="mlpt", node=MLPT)
cs.store(group="model", name="timmlplob", node=TIMMLPLOB)
cs.store(group="model", name="convlob", node=ConvLOB)
cs.store(group="dataset", name="fi_2010", node=FI_2010)