"""
WhatsApp Audio Visualization Tool

Modes:
  - waveform
  - spectrogram
  - mel

Outputs:
  - PNG image (default)
  - MP4 video with synchronized audio (--video)

Requirements:
  pip install numpy matplotlib soundfile librosa moviepy

System dependency (for video):
  ffmpeg installed

Usage:
  Save image:
    python whatsapp_audio_visual.py input.ogg mel output.png

  Save video:
    python whatsapp_audio_visual.py input.ogg mel output.mp4 --video
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import librosa
import librosa.display


# ----------------------------
# Audio utilities
# ----------------------------
def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def ensure_float32(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float32)


# ----------------------------
# Plot builders
# ----------------------------
def build_waveform(y, sr, title):
    dur = len(y) / sr
    t = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(t, y, linewidth=0.6)
    ax.set_xlim(0, dur)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    playhead = ax.axvline(0.0, linewidth=2)
    return fig, playhead, dur


def build_spectrogram(y, sr, title):
    dur = len(y) / sr

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    Pxx, freqs, bins, im = ax.specgram(
        y,
        NFFT=1024,
        Fs=sr,
        noverlap=768,
        scale="dB",
        mode="magnitude",
    )

    ax.set_xlim(0, dur)
    ax.set_ylim(0, min(8000, sr / 2))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    fig.colorbar(im, ax=ax).set_label("Magnitude (dB)")
    playhead = ax.axvline(0.0, linewidth=2)

    return fig, playhead, dur


def build_mel(y, sr, title):
    dur = len(y) / sr

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmax=8000,
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=8000,
        ax=ax,
    )

    ax.set_title(title)
    fig.colorbar(img, ax=ax).set_label("dB")

    playhead = ax.axvline(0.0, linewidth=2)
    ax.set_xlim(0, dur)

    return fig, playhead, dur


# ----------------------------
# Video generation
# ----------------------------
def make_video(fig, playhead, dur, y, sr, out_path):
    from moviepy.editor import VideoClip, AudioFileClip

    def fig_to_frame():
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)

    def make_frame(t):
        playhead.set_xdata([t, t])
        return fig_to_frame()

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        sf.write(str(wav_path), y, sr)

        audio = AudioFileClip(str(wav_path))
        clip = VideoClip(make_frame, duration=dur).set_audio(audio)

        clip.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            fps=30,
        )


# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage:\n"
            "  python whatsapp_audio_visual.py input.ogg mel output.png\n"
            "  python whatsapp_audio_visual.py input.ogg mel output.mp4 --video\n"
        )

    in_path = Path(sys.argv[1])
    mode = sys.argv[2].lower()
    out_path = Path(sys.argv[3])

    make_video_flag = "--video" in sys.argv

    if mode not in {"waveform", "spectrogram", "mel"}:
        raise SystemExit("Mode must be: waveform, spectrogram, mel")

    y, sr = load_audio(in_path)
    y = ensure_float32(to_mono(y))

    title = f"{mode.upper()} - {in_path.name}"

    # Build plot
    if mode == "waveform":
        fig, playhead, dur = build_waveform(y, sr, title)
    elif mode == "spectrogram":
        fig, playhead, dur = build_spectrogram(y, sr, title)
    else:
        fig, playhead, dur = build_mel(y, sr, title)

    fig.tight_layout()

    # Output choice
    if make_video_flag:
        make_video(fig, playhead, dur, y, sr, out_path)
        print("Saved video:", out_path)
    else:
        fig.savefig(out_path)
        print("Saved image:", out_path)

    plt.close(fig)


if __name__ == "__main__":
    main()
