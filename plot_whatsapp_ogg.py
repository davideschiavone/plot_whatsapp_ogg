"""
Visualize a WhatsApp .ogg audio as:
  - waveform (amplitude vs time)
  - spectrogram (linear-frequency FFT)
  - mel (mel-spectrogram via librosa)

Requirements:
  pip install numpy matplotlib soundfile librosa

Usage:
  python plot_whatsapp_ogg.py /path/to/audio.ogg waveform
  python plot_whatsapp_ogg.py /path/to/audio.ogg spectrogram
  python plot_whatsapp_ogg.py /path/to/audio.ogg mel

Notes:
- If soundfile/libsndfile can't decode your .ogg (often Opus-in-Ogg), use ffmpeg-based decoding instead.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: soundfile\n"
        "Install with: pip install soundfile\n"
        "If that fails, install libsndfile on your system."
    ) from e

# Librosa is optional unless you select "mel"
try:
    import librosa
    import librosa.display
except ImportError:
    librosa = None


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
    Downsample for faster waveform plotting.
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
    noverlap: Optional[int] = None,
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
    plt.ylim(0, min(8000, sr / 2))  # voice-friendly default
    plt.colorbar(im).set_label("Magnitude (dB)")
    plt.tight_layout()
    plt.show()


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    title: str,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
):
    """
    Plot a mel-spectrogram using librosa.
    """
    if librosa is None:
        raise SystemExit("Missing dependency: librosa\nInstall with: pip install librosa")

    y = np.asarray(y, dtype=np.float32)
    if hop_length is None:
        hop_length = n_fft // 4
    if fmax is None:
        fmax = sr / 2

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,  # power spectrogram
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        ax=ax,
    )
    plt.title(title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB").set_label("dB")
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage:\n"
            "  python plot_whatsapp_ogg.py /path/to/audio.ogg waveform\n"
            "  python plot_whatsapp_ogg.py /path/to/audio.ogg spectrogram\n"
            "  python plot_whatsapp_ogg.py /path/to/audio.ogg mel"
        )

    path = Path(sys.argv[1]).expanduser().resolve()
    mode = sys.argv[2].strip().lower()

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if mode not in {"waveform", "spectrogram", "mel"}:
        raise SystemExit("Mode must be one of: 'waveform', 'spectrogram', 'mel'.")

    data, sr = load_audio(path)
    y = to_mono(data)

    # ensure float32 for plotting / librosa
    y = np.asarray(y, dtype=np.float32)

    if mode == "waveform":
        plot_waveform(y, sr, title=f"Waveform: {path.name} (sr={sr} Hz)")
    elif mode == "spectrogram":
        plot_spectrogram(y, sr, title=f"Spectrogram: {path.name} (sr={sr} Hz)")
    else:
        plot_mel_spectrogram(y, sr, title=f"Mel-spectrogram: {path.name} (sr={sr} Hz)")


if __name__ == "__main__":
    main()
