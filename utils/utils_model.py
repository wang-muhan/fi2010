from models.mlplob import MLPLOB
from models.kanlob import KANLOB
from models.tlob import TLOB, TLOBClsToken
from models.glalob import GLALOB, GLALOBClsToken
from models.binctabl import BiN_CTABL
from models.deeplob import DeepLOB
from models.adaln_mlplob import AdalnMLPLOB
from models.mlpt import MLPT
from models.timmlplob import TIMMLPLOB
from models.conv import ConvLOB
from transformers import AutoModelForSeq2SeqLM


def pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads=8, is_sin_emb=False, dataset_type=None, **kwargs):
    if model_type == "MLPLOB":
        return MLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type)
    elif model_type == "KANLOB":
        return KANLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type)
    elif model_type == "TLOB":
        use_pos_in_attn = kwargs.get("use_pos_in_attn", False)
        return TLOB(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type, use_pos_in_attn=use_pos_in_attn)
    elif model_type == "TLOB_CLS":
        use_pos_in_attn = kwargs.get("use_pos_in_attn", False)
        return TLOBClsToken(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type, use_pos_in_attn=use_pos_in_attn)
    elif model_type == "GLALOB":
        return GLALOB(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    elif model_type == "GLALOB_CLS":
        return GLALOBClsToken(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    elif model_type == "BINCTABL":
        return BiN_CTABL(60, num_features, seq_size, seq_size, 120, 5, 3, 1)
    elif model_type == "DEEPLOB":
        return DeepLOB()
    elif model_type == "ADALNMLPLOB":
        dropout = kwargs.get("dropout", 0.0)
        use_dyt = kwargs.get("use_dyt", False)
        return AdalnMLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type, dropout=dropout, use_dyt=use_dyt)
    elif model_type == "MLPT":
        num_mlp_layers = kwargs.get("num_mlp_layers")
        if num_mlp_layers is None:
            num_mlp_layers = num_layers
        num_trans_layers = kwargs.get("num_trans_layers")
        if num_trans_layers is None:
            num_trans_layers = num_layers
        return MLPT(hidden_dim, num_mlp_layers, num_trans_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type)
    elif model_type == "TIMMLPLOB":
        variant = kwargs.get("variant", "Mlp")
        return TIMMLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type, variant=variant)
    elif model_type == "CONVLOB":
        dropout = kwargs.get("dropout", 0.0)
        kernel_size = kwargs.get("kernel_size", 32)
        return ConvLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type, kernel_size=kernel_size, dropout=dropout)
    else:
        raise ValueError("Model not found")