import torch
from enum import Enum
from preprocessing.dataset import Dataset  

class DatasetType(Enum):
    FI_2010 = "FI_2010"
    

class ModelType(Enum):
    MLPLOB = "MLPLOB"
    TLOB = "TLOB"
    TLOB_CLS = "TLOB_CLS"
    GLALOB = "GLALOB"
    GLALOB_CLS = "GLALOB_CLS"
    BINCTABL = "BINCTABL"
    DEEPLOB = "DEEPLOB"
    KANLOB = "KANLOB"
    ADALNMLPLOB = "ADALNMLPLOB"
    MLPT = "MLPT"
    
class SamplingType(Enum):
    TIME = "time"
    QUANTITY = "quantity"
    NONE = "none"



TSLA_LOB_MEAN_SIZE_10 = 165.44670902537212
TSLA_LOB_STD_SIZE_10 = 481.7127061897184
TSLA_LOB_MEAN_PRICE_10 = 20180.439318660694
TSLA_LOB_STD_PRICE_10 = 814.8782058033195

TSLA_EVENT_MEAN_SIZE = 88.09459295373463
TSLA_EVENT_STD_SIZE = 86.55913199110894
TSLA_EVENT_MEAN_PRICE = 20178.610720500274
TSLA_EVENT_STD_PRICE = 813.8188032145645
TSLA_EVENT_MEAN_TIME = 0.08644932804905886
TSLA_EVENT_STD_TIME = 0.3512181506722207
TSLA_EVENT_MEAN_DEPTH = 7.365325300819055
TSLA_EVENT_STD_DEPTH = 8.59342838063813

# for 15 days of INTC
INTC_LOB_MEAN_SIZE_10 = 6222.424274871972
INTC_LOB_STD_SIZE_10 = 7538.341086370264
INTC_LOB_MEAN_PRICE_10 = 3635.766219937785
INTC_LOB_STD_PRICE_10 = 44.15649995373795

INTC_EVENT_MEAN_SIZE = 324.6800802006092
INTC_EVENT_STD_SIZE = 574.5781447696605
INTC_EVENT_MEAN_PRICE = 3635.78165265669
INTC_EVENT_STD_PRICE = 43.872407609651184
INTC_EVENT_MEAN_TIME = 0.025201754040915927
INTC_EVENT_STD_TIME = 0.11013627432323592
INTC_EVENT_MEAN_DEPTH = 1.3685517399834501
INTC_EVENT_STD_DEPTH = 2.333747222206966




PRECISION = 32
N_LOB_LEVELS = 10
LEN_LEVEL = 4
LEN_ORDER = 6
LEN_SMOOTH = 10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR_EXPERIMENTS = "data/experiments"
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"
PROJECT_NAME = "EvolutionData"
SPLIT_RATES = [0.8, 0.1, 0.1]
WANDB_API = ""
WANDB_USERNAME = ""

# FI-2010 first-line segmentation metadata (5 stacked stocks)
# Change points are 0-based indices where a new segment starts.
FI2010_TRAIN = {
    "Train_Dst_NoAuction_ZScore_CF_7.txt": {
    "rows": 149,
    "cols": 254750,
        "change_points": [23336, 68210, 106759, 161434],
        "segments": [
            {"start": 0, "end": 23335, "median": 0.40367331},
            {"start": 23336, "end": 68209, "median": -1.020004},
            {"start": 68210, "end": 106758, "median": -0.55343746},
            {"start": 106759, "end": 161433, "median": -1.0190049},
            {"start": 161434, "end": 254749, "median": 1.191941},
        ],
    },
}

FI2010_TEST = {
    "Test_Dst_NoAuction_ZScore_CF_7.txt": {
        "cols": 55478,
        "change_points": [2647, 11309, 19900, 33129],
        "segments": [
            {"start": 0, "end": 2646, "median": 0.40667053},
            {"start": 2647, "end": 11308, "median": -0.99602627},
            {"start": 11309, "end": 19899, "median": -0.53145788},
            {"start": 19900, "end": 33128, "median": -1.006017},
            {"start": 33129, "end": 55477, "median": 1.3368064},
        ],
    },
    "Test_Dst_NoAuction_ZScore_CF_8.txt": {
        "cols": 52172,
        "change_points": [1873, 11144, 21180, 34060],
        "segments": [
            {"start": 0, "end": 1872, "median": 0.34801531},
            {"start": 1873, "end": 11143, "median": -0.9672815},
            {"start": 11144, "end": 21179, "median": -0.52425488},
            {"start": 21180, "end": 34059, "median": -0.98598707},
            {"start": 34060, "end": 52171, "median": 1.3187359},
        ],
    },
    "Test_Dst_NoAuction_ZScore_CF_9.txt": {
        "cols": 31937,
        "change_points": [1888, 7016, 12738, 18559],
        "segments": [
            {"start": 0, "end": 1887, "median": 0.3935748},
            {"start": 1888, "end": 7015, "median": -0.92058435},
            {"start": 7016, "end": 12737, "median": -0.48449127},
            {"start": 12738, "end": 18558, "median": -0.94116402},
            {"start": 18559, "end": 31936, "median": 1.4352983},
        ],
    },
}