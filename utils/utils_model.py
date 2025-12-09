from models.mlplob import MLPLOB
from models.kanlob import KANLOB
from models.tlob import TLOB, TLOBClsToken
from models.glalob import GLALOB, GLALOBClsToken
from models.binctabl import BiN_CTABL
from models.deeplob import DeepLOB
from models.adaln_mlplob import AdalnMLPLOB
from models.mlpt import MLPT
from transformers import AutoModelForSeq2SeqLM


def pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads=8, is_sin_emb=False, dataset_type=None):
    if model_type == "MLPLOB":
        return MLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type)
    elif model_type == "KANLOB":
        return KANLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type)
    elif model_type == "TLOB":
        return TLOB(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    elif model_type == "TLOB_CLS":
        return TLOBClsToken(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    elif model_type == "GLALOB":
        return GLALOB(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    elif model_type == "GLALOB_CLS":
        return GLALOBClsToken(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    elif model_type == "BINCTABL":
        return BiN_CTABL(60, num_features, seq_size, seq_size, 120, 5, 3, 1)
    elif model_type == "DEEPLOB":
        return DeepLOB()
    elif model_type == "ADALNMLPLOB":
        return AdalnMLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type)
    elif model_type == "MLPT":
        return MLPT(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    else:
        raise ValueError("Model not found")