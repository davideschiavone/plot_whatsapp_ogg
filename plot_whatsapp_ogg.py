"""
Generic audio visualizer (waveform / spectrogram / mel)
with OPTIONAL audio processing and trimming.

NEW FEATURE:
  --start_sec X
  --end_sec Y

This cuts the audio before filtering/plotting/video export.

Outputs:
- PNG image (default)
- MP4 video with embedded processed audio (--video)
- Optional processed audio export (--out_audio)

Dependencies:
  pip install numpy matplotlib soundfile scipy librosa moviepy
System:
  ffmpeg required for MP4

Example heartbeat cut:
  python whatsapp_audio_visual.py audio.ogg mel out.mp4 --video \
    --start_sec 5 --end_sec 15 \
    --band_pass 30 800 --normalize
"""

import argparse
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.signal import butter, filtfilt

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


def peak_normalize(y: np.ndarray, target: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m <= 0:
        return y
    return (y / m) * target


# ----------------------------
# Trim audio
# ----------------------------
def trim_audio(
    y: np.ndarray,
    sr: int,
    start_sec: Optional[float],
    end_sec: Optional[float],
) -> np.ndarray:
    """
    Cut audio between start_sec and end_sec.

    Rules:
    - If start_sec is None → start at 0
    - If end_sec is None → go until end of file
    - If both None → return full audio
    """

    n = len(y)

    # Defaults
    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = n / sr  # full duration

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    # Clamp safely
    start_sample = max(0, start_sample)
    end_sample = min(n, end_sample)

    if start_sample >= end_sample:
        raise ValueError(
            f"Invalid trimming range: start={start_sec}s end={end_sec}s"
        )

    return y[start_sample:end_sample]



# ----------------------------
# Filters
# ----------------------------
def apply_highpass_filter(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    nyq = sr / 2
    wn = cutoff_hz / nyq
    b, a = butter(4, wn, btype="high")
    return filtfilt(b, a, y)


def apply_lowpass_filter(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    nyq = sr / 2
    wn = cutoff_hz / nyq
    b, a = butter(4, wn, btype="low")
    return filtfilt(b, a, y)


def apply_hpss_percussive(y: np.ndarray, mix: float = 1.0) -> np.ndarray:
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0:
        return y

    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D)

    y_perc = librosa.istft(P, length=len(y))

    return (1 - mix) * y + mix * y_perc


# ----------------------------
# Audio processing pipeline
# ----------------------------
def process_audio(y, sr, hp, lp, hpss, hpss_mix, normalize):
    if hp is not None:
        print(f"Applying high-pass @ {hp} Hz")
        y = apply_highpass_filter(y, sr, hp)

    if lp is not None:
        print(f"Applying low-pass @ {lp} Hz")
        y = apply_lowpass_filter(y, sr, lp)

    if hpss:
        print(f"Applying HPSS percussive (mix={hpss_mix})")
        y = apply_hpss_percussive(y, mix=hpss_mix)

    if normalize:
        y = peak_normalize(y)

    return y.astype(np.float32)


# ----------------------------
# Plot builders
# ----------------------------
def build_waveform(y, sr, title):
    dur = len(y) / sr
    t = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(t, y, linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    playhead = ax.axvline(0.0, linewidth=2)
    return fig, playhead, dur


def build_spectrogram(y, sr, title, fmax):
    dur = len(y) / sr

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    Pxx, freqs, bins, im = ax.specgram(
        y, Fs=sr, NFFT=1024, noverlap=768, scale="dB"
    )

    ax.set_ylim(0, min(fmax, sr / 2))
    ax.set_xlim(0, dur)
    ax.set_title(title)

    fig.colorbar(im, ax=ax)
    playhead = ax.axvline(0.0, linewidth=2)
    return fig, playhead, dur


def build_mel(y, sr, title, fmax, n_mels):
    dur = len(y) / sr

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512,
        n_mels=n_mels, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=512,
        x_axis="time", y_axis="mel", fmax=fmax, ax=ax
    )

    ax.set_title(title)
    fig.colorbar(img, ax=ax)

    playhead = ax.axvline(0.0, linewidth=2)
    return fig, playhead, dur


# ----------------------------
# Video generation
# ----------------------------
def make_video(fig, playhead, dur, y_audio, sr, out_path):
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
        sf.write(str(wav_path), y_audio, sr)

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
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path)
    parser.add_argument("mode", choices=["waveform", "spectrogram", "mel"])
    parser.add_argument("output", type=Path)

    parser.add_argument("--video", action="store_true")

    parser.add_argument("--start_sec", type=float, default=None)
    parser.add_argument("--end_sec", type=float, default=None)

    parser.add_argument("--high_pass_filter", type=float, default=None)
    parser.add_argument("--low_pass_filter", type=float, default=None)
    parser.add_argument("--band_pass", nargs=2, type=float)

    parser.add_argument("--hpss_percussive", action="store_true")
    parser.add_argument("--hpss_mix", type=float, default=1.0)

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--out_audio", type=Path)

    parser.add_argument("--fmax", type=float, default=8000)
    parser.add_argument("--n_mels", type=int, default=64)

    args = parser.parse_args()

    # Load
    y, sr = load_audio(args.input)
    y = ensure_float32(to_mono(y))

    # Trim if either start OR end is specified
    if args.start_sec is not None or args.end_sec is not None:
        real_start = args.start_sec if args.start_sec is not None else 0
        real_end = args.end_sec if args.end_sec is not None else len(y) / sr

        print(f"Trimming audio: {real_start}s → {real_end}s")

        y = trim_audio(y, sr, args.start_sec, args.end_sec)

    # Band-pass shortcut
    hp = args.high_pass_filter
    lp = args.low_pass_filter
    if args.band_pass:
        hp, lp = args.band_pass

    # Process
    y_proc = process_audio(
        y, sr,
        hp, lp,
        args.hpss_percussive,
        args.hpss_mix,
        args.normalize
    )

    # Save processed audio
    if args.out_audio:
        sf.write(str(args.out_audio), y_proc, sr)
        print("Saved processed audio:", args.out_audio)

    # Plot
    title = f"{args.mode.upper()} - {args.input.name}"

    if args.mode == "waveform":
        fig, playhead, dur = build_waveform(y_proc, sr, title)

    elif args.mode == "spectrogram":
        fig, playhead, dur = build_spectrogram(y_proc, sr, title, args.fmax)

    else:
        fig, playhead, dur = build_mel(y_proc, sr, title, args.fmax, args.n_mels)

    fig.tight_layout()

    # Output
    if args.video:
        make_video(fig, playhead, dur, y_proc, sr, args.output)
        print("Saved video:", args.output)
    else:
        fig.savefig(args.output)
        print("Saved image:", args.output)

    plt.close(fig)


if __name__ == "__main__":
    main()
