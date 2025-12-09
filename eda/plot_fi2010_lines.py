import argparse
from pathlib import Path
import numpy as np
import matplotlib
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cwt_fallback(data, wavelet, widths, **kwargs):
    out = np.zeros((len(widths), data.size), dtype=complex)
    for i, width in enumerate(widths):
        wv = wavelet(data.size, width, **kwargs)
        out[i] = signal.convolve(data, wv, mode="same")
    return out


def morlet_wavelet(M, s, w=5.0):
    t = np.arange(M) - (M - 1.0) / 2
    return np.exp(1j * w * t / s) * np.exp(-0.5 * (t / s) ** 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/home/wmh/lob/TLOB/data/FI_2010/Test_Dst_NoAuction_ZScore_CF_8.txt")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--max-lines", type=int, default=1, dest="max_lines")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-dir", default="/home/wmh/lob/TLOB/visualize/fi2010_lines", dest="output_dir")
    return parser.parse_args()


def plot_line(values, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fft(values, path):
    v = values - values.mean()
    freq = np.fft.rfftfreq(v.size, d=1.0)
    spec = np.abs(np.fft.rfft(v))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq, spec)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cwt(values, path):
    v = values
    if v.size > 4096:
        v = v[:4096]
    widths = np.arange(1, 128)
    cwt_fn = signal.cwt if hasattr(signal, "cwt") else cwt_fallback
    coef = cwt_fn(v, morlet_wavelet, widths, w=5.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(np.abs(coef), extent=[0, v.size, widths[-1], widths[0]], aspect="auto")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    data_path = Path(args.file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with data_path.open() as f:
        for idx, line in enumerate(f, start=1):
            if idx < args.start:
                continue
            if args.max_lines and (idx - args.start + 1) > args.max_lines:
                break
            values = np.fromstring(line, sep=" ")
            if args.stride > 1:
                values = values[::args.stride]
            base = out_dir / f"line_{idx}"
            plot_line(values, base.with_suffix(".png"))
            plot_fft(values, base.with_name(f"{base.name}_fft.png"))
            plot_cwt(values, base.with_name(f"{base.name}_cwt.png"))


if __name__ == "__main__":
    main()
