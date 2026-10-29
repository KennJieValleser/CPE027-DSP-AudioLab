# AudioLab

A beginner-friendly Streamlit application for audio signal-processing demonstrations.

## Features

- RMS loudness normalization with adjustable target level and maximum gain
- Basic spectral-gating noise reduction with adjustable strength and noise-profile duration
- Dynamic range compressor with threshold, ratio, attack, release, and makeup gain
- Peak limiter
- Three-band bass/mid/treble EQ
- Original vs processed A/B playback
- Waveform and frequency spectrum comparison
- RMS, peak, clipping-count, duration, sample-rate, and channel metrics
- Presets for speech/podcast, noisy recording, music, and gentle processing
- WAV export as 16-bit PCM, 24-bit PCM, or 32-bit float

## Requirements

Python 3.10 or newer is recommended.

## Install and run

### Windows (CMD)

```Cmd

python -m venv .venv
.\.venv\Scripts\Activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Linux / Raspberry Pi

```bash
cd audiolab_dsp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit usually opens the app at `http://localhost:8501`.

## How to test

1. Upload a WAV or other supported audio file.
2. Select a preset, then adjust target RMS, noise reduction, compressor, limiter, and EQ.
3. Listen to A (original) and B (processed).
4. Check waveform, spectrum, RMS and peak measurements.
5. Download the processed WAV.

For a good noise-reduction demonstration, use a recording with a short noise-only section at the beginning. The spectral gate estimates its noise profile from that section.

## Important limitations

- This is a DSP teaching/demo app, not a professional mastering suite.
- Spectral gating can produce watery or musical-noise artifacts, especially at high strengths.
- The noise profile is estimated from the beginning of the clip; if speech or music starts immediately, the gate may misclassify it as noise.
- RMS normalization targets the overall RMS level, not a broadcast loudness standard such as LUFS.
- The compressor and limiter are intentionally simple so the processing chain is easier to explain.
- The app reports clipping counts near full scale. It cannot repair clipping that was already present in the source.
- The spectrum plot is a relative magnitude comparison, not a calibrated acoustic measurement.
