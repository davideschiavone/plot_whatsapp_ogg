"""
Visualize a WhatsApp .ogg audio either as:
  - waveform (amplitude vs time)
  - spectrogram (time-frequency)

Requirements:
  pip install numpy matplotlib soundfile

Usage:
  python plot_whatsapp_ogg.py /path/to/audio.ogg waveform
  python plot_whatsapp_ogg.py /path/to/audio.ogg spectrogram

Notes:
- If soundfile/libsndfile can't decode your .ogg (often Opus-in-Ogg), use ffmpeg-based decoding instead.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: soundfile\n"
        "Install with: pip install soundfile\n"
        "If that fails, install libsndfile on your system."
    ) from e


def load_audio(path: Path):
    """Loads audio samples and sample rate. Returns: (data, sr)."""
    data, sr = sf.read(str(path), always_2d=False)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    """If multichannel, average to mono."""
    data = np.asarray(data)
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def downsample_for_plot(y: np.ndarray, max_points: int = 200_000):
    """
    Downsample for faster plotting (keeps overall shape).
    Returns: (y_ds, step)
    """
    n = y.shape[0]
    if n <= max_points:
        return y, 1
    step = int(np.ceil(n / max_points))
    return y[::step], step


def plot_waveform(y: np.ndarray, sr: int, title: str):
    y = np.asarray(y, dtype=np.float32)
    y_plot, step = downsample_for_plot(y)
    sr_plot = sr / step
    t = np.arange(len(y_plot)) / sr_plot

    plt.figure(figsize=(12, 4))
    plt.plot(t, y_plot)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(
    y: np.ndarray,
    sr: int,
    title: str,
    nfft: int = 1024,
    noverlap: Optional[int] = None
):
    """
    Plot a magnitude spectrogram using matplotlib's specgram (FFT-based).
    - nfft: FFT window size
    - noverlap: overlap between windows (defaults to 75% of nfft)
    """
    y = np.asarray(y, dtype=np.float32)
    if noverlap is None:
        noverlap = int(0.75 * nfft)

    plt.figure(figsize=(12, 4))
    Pxx, freqs, bins, im = plt.specgram(
        y,
        NFFT=nfft,
        Fs=sr,
        noverlap=noverlap,
        scale="dB",
        mode="magnitude",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.ylim(0, min(8000, sr / 2))  # nice default for voice; adjust as you like
    plt.colorbar(im).set_label("Magnitude (dB)")
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage:\n"
            "  python plot_whatsapp_ogg.py /path/to/audio.ogg waveform\n"
            "  python plot_whatsapp_ogg.py /path/to/audio.ogg spectrogram"
        )

    path = Path(sys.argv[1]).expanduser().resolve()
    mode = sys.argv[2].strip().lower()

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if mode not in {"waveform", "spectrogram"}:
        raise SystemExit("Mode must be either 'waveform' or 'spectrogram'.")

    data, sr = load_audio(path)
    y = to_mono(data)

    if mode == "waveform":
        plot_waveform(y, sr, title=f"Waveform: {path.name} (sr={sr} Hz)")
    else:
        plot_spectrogram(y, sr, title=f"Spectrogram: {path.name} (sr={sr} Hz)")


if __name__ == "__main__":
    main()
