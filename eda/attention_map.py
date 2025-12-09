import argparse
import os
import sys
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import constants as cst
from preprocessing.fi_2010 import fi_2010_load
from preprocessing.dataset import Dataset
from torch.utils.data import DataLoader
from models.engine import Engine


def ensure_checkpoint(path):
    if not os.path.isfile(path):
        raise SystemExit(f"Checkpoint not found. Run: python utils/download_ckpt.py --all")


def load_engine(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    h = ckpt["hyper_parameters"]
    model = Engine.load_from_checkpoint(
        path,
        map_location=device,
        weights_only=False,
        seq_size=h["seq_size"],
        horizon=h["horizon"],
        max_epochs=h["max_epochs"],
        model_type=h["model_type"],
        is_wandb=h["is_wandb"],
        experiment_type=h["experiment_type"],
        lr=h["lr"],
        optimizer=h.get("optimizer", "Adam"),
        dir_ckpt=h.get("dir_ckpt", "model.ckpt"),
        num_features=h["num_features"],
        dataset_type=h["dataset_type"],
        num_layers=h["num_layers"],
        hidden_dim=h["hidden_dim"],
        num_heads=h["num_heads"],
        is_sin_emb=h["is_sin_emb"],
        len_test_dataloader=h.get("len_test_dataloader", None),
    )
    return model.to(device)


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def build_loaders(path, horizon, seq_size, all_features, batch_size):
    train_x, train_y, val_x, val_y, test_x, test_y = fi_2010_load(path, seq_size, horizon, all_features)
    train_ds = Dataset(train_x, train_y, seq_size)
    val_ds = Dataset(val_x, val_y, seq_size)
    test_ds = Dataset(test_x, test_y, seq_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    return train_dl, val_dl, test_dl


def collect_mean_attention(engine, loader, device):
    base = engine.model
    temporal_idx = [i for i in range(len(base.layers)) if i % 2 == 0]
    channel_idx = [i for i in range(len(base.layers)) if i % 2 == 1]
    seq_len = base.seq_size
    temp_sums = [torch.zeros(seq_len, device=device) for _ in temporal_idx]
    temp_counts = [0 for _ in temporal_idx]
    chan_sums = []
    chan_counts = []
    buffer = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            buffer.append((idx, out[1].detach()))
        return hook

    hooks = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(base.layers)]
    base.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            buffer.clear()
            with engine.ema.average_parameters():
                _ = base(xb)
            for idx, att in buffer:
                att = att.mean(dim=1)
                if idx % 2 == 0:
                    tpos = idx // 2
                    att_layer = att.mean(dim=1)
                    temp_sums[tpos] += att_layer.sum(dim=0)
                    temp_counts[tpos] += att_layer.shape[0]
                else:
                    if len(chan_sums) <= idx // 2:
                        q_len = att.shape[-2]
                        chan_sums.append(torch.zeros(q_len, device=device))
                        chan_counts.append(0)
                    chan_layer = att.mean(dim=1)
                    pos = idx // 2
                    chan_sums[pos] += chan_layer.sum(dim=0)
                    chan_counts[pos] += chan_layer.shape[0]
    for h in hooks:
        h.remove()
    temp_means = torch.stack([temp_sums[i] / temp_counts[i] for i in range(len(temp_sums))]).cpu().numpy()

    # Handle varying sizes for channel attention (due to downsampling in TLOB last layers)
    # Filter out layers with reduced dimensions (e.g. the last layer with 36 channels instead of 144)
    chan_means_list = [chan_sums[i] / chan_counts[i] for i in range(len(chan_sums))]
    if chan_means_list:
        max_len = max([t.shape[0] for t in chan_means_list])
        chan_means_list = [m for m in chan_means_list if m.shape[0] == max_len]
        chan_means = torch.stack(chan_means_list).cpu().numpy()
    else:
        chan_means = np.array([])

    return temp_means, chan_means


def plot_attention(mat, path, title, xlabel):
    plt.figure(figsize=(8, 5))
    plt.imshow(mat, cmap="magma", aspect="auto")
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel("Depth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    default_ckpt = "/home/wmh/lob/TLOB/data/checkpoints/TLOB/HuggingFace/FI-2010_horizon_1_TLOB_seed_42.ckpt"
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--data-dir", default="/home/wmh/lob/TLOB/data/FI_2010")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seq-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--all-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default="/home/wmh/lob/TLOB/visualize")
    args = parser.parse_args()

    set_deterministic(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect checkpoints
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = sorted([os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir) if f.endswith(".ckpt")])

    if not checkpoints:
        print(f"No checkpoints found. Checked: {args.checkpoint or args.checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s).")

    for ckpt_path in checkpoints:
        ensure_checkpoint(ckpt_path)
        engine = load_engine(ckpt_path, device)
        # Mapping for horizon: checkpoints store horizon index (1,2,3,5,10), but dataset loader needs raw horizon (10,20,30,50,100)
        # Assuming standard mapping: 1->10, 2->20, 3->30, 5->50, 10->100
        if engine.horizon < 10:
            data_horizon = engine.horizon * 10
        else:
            data_horizon = engine.horizon * 10
            
        ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", "")
        current_out_dir = os.path.join(args.out_dir, ckpt_name)
        os.makedirs(current_out_dir, exist_ok=True)

        print(f"\nProcessing checkpoint: {ckpt_name}")
        print(f"  Model horizon param: {engine.horizon} -> Data horizon: {data_horizon}")
        
        train_dl, val_dl, test_dl = build_loaders(args.data_dir, data_horizon, args.seq_size, args.all_features, args.batch_size)

        train_temp, train_chan = collect_mean_attention(engine, train_dl, device)
        val_temp, val_chan = collect_mean_attention(engine, val_dl, device)
        test_temp, test_chan = collect_mean_attention(engine, test_dl, device)

        plot_attention(train_temp, os.path.join(current_out_dir, "train_temp_att.png"), f"Train temporal attention (H={engine.horizon})", "Sequence position")
        plot_attention(val_temp, os.path.join(current_out_dir, "val_temp_att.png"), f"Val temporal attention (H={engine.horizon})", "Sequence position")
        plot_attention(test_temp, os.path.join(current_out_dir, "test_temp_att.png"), f"Test temporal attention (H={engine.horizon})", "Sequence position")
        plot_attention(train_chan, os.path.join(current_out_dir, "train_chan_att.png"), f"Train channel attention (H={engine.horizon})", "Channel index")
        plot_attention(val_chan, os.path.join(current_out_dir, "val_chan_att.png"), f"Val channel attention (H={engine.horizon})", "Channel index")
        plot_attention(test_chan, os.path.join(current_out_dir, "test_chan_att.png"), f"Test channel attention (H={engine.horizon})", "Channel index")


if __name__ == "__main__":
    main()

