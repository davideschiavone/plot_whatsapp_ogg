"""
Create a VIDEO (MP4) where you SEE the waveform/spectrogram/mel AND HEAR the audio,
synchronized with a moving playhead line.

Dependencies:
  pip install numpy matplotlib soundfile moviepy librosa

System dependency (recommended):
  ffmpeg must be installed and on PATH for mp4 writing.

Usage:
  python whatsapp_audio_video.py input.ogg waveform  output.mp4
  python whatsapp_audio_video.py input.ogg spectrogram output.mp4
  python whatsapp_audio_video.py input.ogg mel        output.mp4

Notes:
- This renders a static visualization + a moving vertical cursor (playhead).
- If soundfile can't decode your .ogg (some WhatsApp Opus OGG), decode with ffmpeg first to wav.
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf

# librosa is only needed for mel
import librosa
import librosa.display

from moviepy.editor import VideoClip, AudioFileClip


# ----------------------------
# Audio utilities
# ----------------------------
def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio with soundfile. Returns (samples, sample_rate)."""
    data, sr = sf.read(str(path), always_2d=False)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def ensure_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float32, copy=False)
    # if int, convert to float in [-1, 1]
    maxv = np.iinfo(y.dtype).max
    return (y.astype(np.float32) / maxv).astype(np.float32)


# ----------------------------
# Visualization builders
# ----------------------------
def build_waveform_figure(y: np.ndarray, sr: int, title: str):
    dur = len(y) / sr
    t = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(t, y, linewidth=0.6)
    ax.set_xlim(0, dur)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(False)

    playhead = ax.axvline(0.0, linewidth=2.0)  # moving cursor
    time_txt = ax.text(
        0.99,
        0.95,
        "0.00s",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.6),
    )
    return fig, ax, playhead, time_txt, dur


def build_linear_spectrogram_figure(
    y: np.ndarray,
    sr: int,
    title: str,
    nfft: int = 1024,
    noverlap: Optional[int] = None,
    fmax: Optional[float] = 8000.0,
):
    if noverlap is None:
        noverlap = int(0.75 * nfft)

    dur = len(y) / sr

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    # specgram returns the image object "im" we can keep
    Pxx, freqs, bins, im = ax.specgram(
        y,
        NFFT=nfft,
        Fs=sr,
        noverlap=noverlap,
        scale="dB",
        mode="magnitude",
    )
    ax.set_xlim(0, dur)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    if fmax is not None:
        ax.set_ylim(0, min(fmax, sr / 2))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)")

    playhead = ax.axvline(0.0, linewidth=2.0)
    time_txt = ax.text(
        0.99,
        0.95,
        "0.00s",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.6),
    )
    return fig, ax, playhead, time_txt, dur


def build_mel_spectrogram_figure(
    y: np.ndarray,
    sr: int,
    title: str,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
):
    if hop_length is None:
        hop_length = n_fft // 4

    dur = len(y) / sr

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax if fmax is not None else sr / 2,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax if fmax is not None else sr / 2,
        ax=ax,
    )
    ax.set_title(title)

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("dB")

    ax.set_xlim(0, dur)
    playhead = ax.axvline(0.0, linewidth=2.0)
    time_txt = ax.text(
        0.99,
        0.95,
        "0.00s",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.6),
    )
    return fig, ax, playhead, time_txt, dur


# ----------------------------
# Video rendering
# ----------------------------
def fig_to_rgb_array(fig) -> np.ndarray:
    """Render a matplotlib figure to an RGB numpy array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def make_video(
    y: np.ndarray,
    sr: int,
    mode: str,
    out_path: Path,
    fps: int = 30,
):
    title = f"{mode.capitalize()} (sr={sr} Hz)"

    # Build the static plot once; we’ll only move the playhead per frame
    if mode == "waveform":
        fig, ax, playhead, time_txt, dur = build_waveform_figure(y, sr, title)
    elif mode == "spectrogram":
        fig, ax, playhead, time_txt, dur = build_linear_spectrogram_figure(y, sr, title)
    elif mode == "mel":
        fig, ax, playhead, time_txt, dur = build_mel_spectrogram_figure(y, sr, title)
    else:
        raise ValueError("mode must be one of: waveform, spectrogram, mel")

    # Reduce margins a bit for nicer video framing
    fig.tight_layout()

    def make_frame(t: float) -> np.ndarray:
        # Clamp time to duration
        tt = 0.0 if t < 0 else (dur if t > dur else t)
        playhead.set_xdata([tt, tt])
        time_txt.set_text(f"{tt:0.2f}s")
        return fig_to_rgb_array(fig)

    # Write audio to a temporary WAV so moviepy can attach it reliably
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        sf.write(str(wav_path), y, sr)

        audio_clip = AudioFileClip(str(wav_path))
        video_clip = VideoClip(make_frame, duration=dur).set_fps(fps).set_audio(audio_clip)

        # mp4 with AAC audio is widely compatible
        video_clip.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            threads=4,
            preset="medium",
        )

    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------
def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage:\n"
            "  python whatsapp_audio_video.py input.ogg waveform  output.mp4\n"
            "  python whatsapp_audio_video.py input.ogg spectrogram output.mp4\n"
            "  python whatsapp_audio_video.py input.ogg mel        output.mp4\n"
        )

    in_path = Path(sys.argv[1]).expanduser().resolve()
    mode = sys.argv[2].strip().lower()
    out_path = Path(sys.argv[3]).expanduser().resolve()

    if mode not in {"waveform", "spectrogram", "mel"}:
        raise SystemExit("Mode must be one of: waveform, spectrogram, mel")

    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    data, sr = load_audio(in_path)
    y = ensure_float32(to_mono(data))

    make_video(y, sr, mode, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
