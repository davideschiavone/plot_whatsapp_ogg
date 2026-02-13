# 🎧 WhatsApp Audio Visualizer

A simple Python script to visualize WhatsApp (or any) audio files as:

- Waveform
- Spectrogram
- Mel-Spectrogram

It also supports:

- trimming audio in time
- filtering (high-pass / low-pass / band-pass)
- speech reduction using HPSS percussive extraction
- exporting plots as PNG
- exporting videos with embedded audio as MP4
- saving the processed audio separately

---

## ✅ Features

| Feature | Supported |
|--------|----------|
| Waveform plot | ✅ |
| Linear Spectrogram | ✅ |
| Mel-Spectrogram | ✅ |
| Export PNG image | ✅ |
| Export MP4 video with audio | ✅ |
| Trim audio with timestamps | ✅ |
| High-pass / Low-pass filtering | ✅ |
| Band-pass shortcut | ✅ |
| Reduce speech with HPSS | ✅ |
| Save cleaned audio file | ✅ |

---

## 📦 Installation

Requires `python` with a version >= `3.8`.

```bash
pip install numpy matplotlib soundfile scipy librosa moviepy
```

---

## 🚀 Usage

The script is called like this:

```bash
python plot_whatsapp_ogg.py INPUT_AUDIO MODE OUTPUT_FILE
```

Where:

- INPUT_AUDIO = .ogg, .wav, .mp3, etc.
- MODE = waveform, spectrogram, or mel
- OUTPUT_FILE = .png or .mp4

---

# 🎛️ Positional Arguments

## 1. INPUT_AUDIO

The audio file to analyze.

Example:

```bash
audio.ogg
heartbeat.wav
recording.mp3
```

---

## 2. MODE

Choose one of:

| Mode | Meaning |
|------|---------|
| `waveform` | Plot amplitude vs time |
| `spectrogram` | Plot frequency vs time (linear frequency scale) |
| `mel` | Plot mel-spectrogram (perceptual scale, good for speech/heartbeat analysis) |


## 3. OUTPUT_FILE

The output file to generate.

- If you want an image → use `.png`
- If you want a video → use `.mp4` + `--video`

Example:

```bash
waveform.png
heartbeat.mp4
```

## ✅ Basic Examples

### Waveform Plot (PNG)

```bash
python plot_whatsapp_ogg.py audio.ogg waveform waveform.png
```

### Spectrogram Plot (PNG)

```bash
python plot_whatsapp_ogg.py audio.ogg spectrogram spec.png
```

### Mel-Spectrogram Plot (PNG)

```bash
python plot_whatsapp_ogg.py audio.ogg mel mel.png
```

---

## 🎥 Export Video with Audio

Add the --video flag to generate an MP4 with plot + sound + moving playhead.

### Waveform Video

```bash
python plot_whatsapp_ogg.py audio.ogg waveform waveform.mp4 --video
```

### Mel-Spectrogram Video

```bash
python plot_whatsapp_ogg.py audio.ogg mel mel.mp4 --video
```

---

## ✂️ Trimming Options

These options cut the audio BEFORE filtering/plotting.

---

### --start_sec X

Start the audio at second **X**.

Example (skip first 5 seconds):

```bash
--start_sec 5
```

---

### --end_sec Y

Stop the audio at second **Y**.

Example (keep only first 12 seconds):

```bash
--end_sec 12
```

---

### Both together

Keep only a segment:

```bash
--start_sec 5 --end_sec 15
```

---

### Only start OR only end

- `--start_sec 10` → from 10s to end  
- `--end_sec 8` → from start to 8s  

---

### ✅ Basic Examples with Trimming

#### Keep only from 5s to 15s

```bash
python plot_whatsapp_ogg.py audio.ogg mel cut.png --start_sec 5 --end_sec 15
```

#### Keep from 10s until the end

```bash
python plot_whatsapp_ogg.py audio.ogg mel cut.png --start_sec 10
```

#### Keep from start until 12s

```bash
python plot_whatsapp_ogg.py audio.ogg mel cut.png --end_sec 12
```

---


## 🎚️ Filtering Options

Filters modify the AUDIO itself (not only the plot).

### --high_pass_filter Hz

Removes low-frequency rumble below cutoff.

Example:

```bash
--high_pass_filter 50
```

---

### --low_pass_filter Hz

Removes high-frequency noise above cutoff.

Example:

```bash
--low_pass_filter 500
```

---

### --band_pass LOW HIGH

Convenience shortcut that applies BOTH:

- high-pass at LOW Hz
- low-pass at HIGH Hz

Example (heartbeat band):

```bash
--band_pass 30 800
```
---


## 🥁 Speech Reduction (HPSS)

---

### --hpss_percussive

Apply Harmonic–Percussive Source Separation.

This often reduces speech and keeps transient “beat-like” components.

Example:

```bash
--hpss_percussive
```

---

### --hpss_mix VALUE

Controls how strong the percussive extraction is.

| Value | Effect |
|------|--------|
| 1.0 | Only percussive component (strong speech removal) |
| 0.5 | Mix of original + percussive |
| 0.0 | No effect |

Example:

```bash
--hpss_percussive --hpss_mix 0.4
```

---

## 🔊 --normalize

Peak-normalize the processed audio.

Useful if filtering makes the signal very quiet.

Example:

```bash
--normalize
```

---

## 🎵 --out_audio FILE.wav

Save the processed audio to disk.

Example:

```bash
--out_audio cleaned.wav
```

Output:

- `cleaned.wav` contains the filtered/trimmed audio

---

## 📊 Plot Control Options

### --fmax Hz

Sets the maximum frequency shown in spectrogram/mel plots.

Default: 8000 Hz

Heartbeat-friendly:

```bash
--fmax 1200
```

---

### --n_mels N

Number of mel frequency bands (mel mode only).

Default: 64

Low-frequency heartbeat plots:

```bash
--n_mels 32
```

---

# Examples

```bash
python plot_whatsapp_ogg.py audio.ogg mel heartbeat.mp4 --video \
  --start_sec 5 --end_sec 15 \
  --band_pass 30 800 \
  --normalize \
  --fmax 1200 \
  --n_mels 32
```

This will:

- cut only the specified region in time region
- suppress most speech frequencies
- zoom the plot to the correct band
