
import io
import math

import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, stft, istft


st.set_page_config(
    page_title="AudioLab",
    page_icon="🎚️",
    layout="wide"
)

st.title("🎚️ AudioLab")
st.caption(
    "Automatic Audio Volume Normalizer • Noise Reduction • "
    "Compressor • Limiter • EQ"
)

st.markdown("""
Upload an audio file, adjust the DSP settings, and compare the original
with the processed result. Processing happens in memory, and the output
can be exported as a WAV file.
""")


# --------------------------------------------------
# DSP PRESETS
# --------------------------------------------------

PRESETS = {
    "Custom": {},

    "Speech / Podcast": {
        "target_rms": -18.0,
        "noise_reduction": True,
        "noise_strength": 1.3,
        "compressor": True,
        "threshold": -24.0,
        "ratio": 3.0,
        "limiter": True,
        "peak_limit": -1.0,
        "low_gain": -1.0,
        "mid_gain": 1.0,
        "high_gain": 1.5,
    },

    "Noisy Recording": {
        "target_rms": -20.0,
        "noise_reduction": True,
        "noise_strength": 2.0,
        "compressor": True,
        "threshold": -26.0,
        "ratio": 3.5,
        "limiter": True,
        "peak_limit": -1.0,
        "low_gain": 0.0,
        "mid_gain": 0.0,
        "high_gain": -1.0,
    },

    "Music": {
        "target_rms": -16.0,
        "noise_reduction": False,
        "noise_strength": 1.0,
        "compressor": True,
        "threshold": -18.0,
        "ratio": 2.0,
        "limiter": True,
        "peak_limit": -1.0,
        "low_gain": 1.0,
        "mid_gain": 0.0,
        "high_gain": 1.0,
    },

    "Gentle / Minimal": {
        "target_rms": -20.0,
        "noise_reduction": False,
        "noise_strength": 1.0,
        "compressor": False,
        "threshold": -24.0,
        "ratio": 2.0,
        "limiter": True,
        "peak_limit": -1.0,
        "low_gain": 0.0,
        "mid_gain": 0.0,
        "high_gain": 0.0,
    },
}


# --------------------------------------------------
# AUDIO CONVERSION AND METRICS
# --------------------------------------------------

def db_to_amp(db):
    """Convert decibels to linear amplitude."""
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def amp_to_db(value, floor_db=-120.0):
    """
    Convert amplitude to dBFS.

    Supports both scalar values and NumPy arrays.
    Returns a float for scalar input and an array for array input.
    """
    value = np.asarray(value, dtype=np.float64)

    result = 20.0 * np.log10(
        np.maximum(value, 1e-12)
    )

    result = np.maximum(result, floor_db)

    if result.ndim == 0:
        return float(result)

    return result


def amp_to_db_vector(values):
    """Convert an array of amplitudes to decibels."""
    values = np.asarray(values, dtype=np.float64)

    return 20.0 * np.log10(
        np.maximum(values, 1e-6)
    )


def to_stereo_shape(audio):
    """Ensure audio has shape (samples, channels)."""
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 1:
        audio = audio[:, None]

    if audio.ndim != 2:
        raise ValueError(
            "Audio must have one or two dimensions."
        )

    return audio


def rms_db(audio):
    """Calculate RMS level in dBFS."""
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        return -120.0

    rms_value = np.sqrt(
        np.mean(np.square(audio))
    )

    return amp_to_db(rms_value)


def peak_db(audio):
    """Calculate peak amplitude in dBFS."""
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        return -120.0

    return amp_to_db(
        np.max(np.abs(audio))
    )


def count_clipped_samples(audio, threshold=0.999):
    """Count samples close to or beyond full scale."""
    return int(
        np.count_nonzero(
            np.abs(audio) >= threshold
        )
    )


# --------------------------------------------------
# DIGITAL FILTERS AND EQUALIZER
# --------------------------------------------------

def highpass_lowpass_filter(
    audio,
    sr,
    kind,
    cutoff,
    order=2
):
    """Apply a Butterworth low-pass or high-pass filter."""
    nyquist = sr / 2.0

    cutoff = min(
        max(float(cutoff), 10.0),
        nyquist * 0.95
    )

    sos = butter(
        order,
        cutoff,
        btype=kind,
        fs=sr,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        audio,
        axis=0
    )


def equalize(
    audio,
    sr,
    low_db=0.0,
    mid_db=0.0,
    high_db=0.0
):
    """Apply a simple three-band EQ."""
    if (
        low_db == 0
        and mid_db == 0
        and high_db == 0
    ):
        return audio.copy()

    nyquist = sr / 2.0

    low_cut = min(
        250.0,
        nyquist * 0.25
    )

    high_cut = min(
        4000.0,
        nyquist * 0.80
    )

    if high_cut <= low_cut * 1.2:
        high_cut = min(
            low_cut * 1.5,
            nyquist * 0.9
        )

    low_band = highpass_lowpass_filter(
        audio,
        sr,
        "lowpass",
        low_cut
    )

    above_low = highpass_lowpass_filter(
        audio,
        sr,
        "highpass",
        low_cut
    )

    mid_band = highpass_lowpass_filter(
        above_low,
        sr,
        "lowpass",
        high_cut
    )

    high_band = highpass_lowpass_filter(
        audio,
        sr,
        "highpass",
        high_cut
    )

    result = (
        low_band * db_to_amp(low_db)
        + mid_band * db_to_amp(mid_db)
        + high_band * db_to_amp(high_db)
    )

    return result


# --------------------------------------------------
# SPECTRAL NOISE REDUCTION
# --------------------------------------------------

def spectral_gate(
    audio,
    sr,
    strength=1.3,
    noise_seconds=0.5,
    floor=0.04
):
    """
    Estimate the noise profile from the beginning of the audio
    and attenuate frequency bins close to the estimated noise.

    Best results occur when the beginning contains background
    noise without speech or music.
    """
    if audio.shape[0] < 256 or strength <= 0:
        return audio.copy()

    segment_size = min(
        audio.shape[0],
        2048
    )

    nperseg = max(
        256,
        2 ** int(
            np.floor(
                np.log2(segment_size)
            )
        )
    )

    noverlap = nperseg * 3 // 4

    cleaned_channels = []

    for channel in range(audio.shape[1]):
        x = audio[:, channel]

        frequencies, times, z = stft(
            x,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True
        )

        magnitude = np.abs(z)

        noise_frames = max(
            1,
            int(
                np.searchsorted(
                    times,
                    max(0.05, noise_seconds)
                )
            )
        )

        noise_frames = min(
            noise_frames,
            magnitude.shape[1]
        )

        noise_profile = np.mean(
            magnitude[:, :noise_frames],
            axis=1,
            keepdims=True
        )

        threshold = noise_profile * (
            1.0 + 1.8 * strength
        )

        mask = magnitude / (
            magnitude + threshold + 1e-10
        )

        mask = np.clip(
            mask,
            floor,
            1.0
        )

        mask = mask ** min(
            max(strength, 0.1),
            3.0
        )

        _, cleaned = istft(
            z * mask,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=True
        )

        if len(cleaned) < len(x):
            cleaned = np.pad(
                cleaned,
                (0, len(x) - len(cleaned))
            )

        cleaned_channels.append(
            cleaned[:len(x)]
        )

    return np.column_stack(
        cleaned_channels
    )


# --------------------------------------------------
# DYNAMIC RANGE COMPRESSION
# --------------------------------------------------

def compressor(
    audio,
    sr,
    threshold_db=-24.0,
    ratio=3.0,
    attack_ms=10.0,
    release_ms=120.0,
    makeup_db=0.0
):
    """Apply a simple feed-forward dynamic range compressor."""
    if ratio <= 1.0:
        return audio.copy()

    x = audio.copy()

    detector = np.max(
        np.abs(x),
        axis=1
    )

    attack = max(
        0.001,
        attack_ms / 1000.0
    )

    release = max(
        0.001,
        release_ms / 1000.0
    )

    attack_coeff = math.exp(
        -1.0 / (sr * attack)
    )

    release_coeff = math.exp(
        -1.0 / (sr * release)
    )

    envelope = np.zeros_like(
        detector
    )

    previous = 0.0

    for i, value in enumerate(detector):
        if value > previous:
            coeff = attack_coeff
        else:
            coeff = release_coeff

        previous = (
            coeff * previous
            + (1.0 - coeff) * value
        )

        envelope[i] = previous

    # Convert the envelope array to decibels.
    env_db = np.maximum(
        amp_to_db_vector(envelope),
        -120.0
    )

    over_threshold = np.maximum(
        env_db - threshold_db,
        0.0
    )

    compressed_db = (
        env_db
        - over_threshold * (1.0 - 1.0 / ratio)
    )

    gain_db = (
        compressed_db
        - env_db
        + makeup_db
    )

    gain = db_to_amp(gain_db)

    return x * gain[:, None]


# --------------------------------------------------
# RMS LOUDNESS NORMALIZATION
# --------------------------------------------------

def normalize_rms(
    audio,
    target_db=-18.0,
    max_gain_db=18.0
):
    """Adjust overall RMS level toward the selected target."""
    current = rms_db(audio)

    if current <= -119.0:
        return audio.copy(), 0.0

    gain_db = float(
        np.clip(
            target_db - current,
            -24.0,
            max_gain_db
        )
    )

    return (
        audio * db_to_amp(gain_db),
        gain_db
    )


# --------------------------------------------------
# PEAK LIMITER
# --------------------------------------------------

def peak_limiter(
    audio,
    ceiling_db=-1.0
):
    """Apply simple peak protection by reducing overall gain if needed."""
    ceiling = float(db_to_amp(ceiling_db))
    peak = np.max(np.abs(audio)) if audio.size else 0.0

    if peak <= ceiling or peak == 0.0:
        return audio.copy()

    # Reduce the whole signal just enough to keep its peak at the ceiling.
    return audio * (ceiling / peak)


# --------------------------------------------------
# COMPLETE DSP PROCESSING CHAIN
# --------------------------------------------------

def process_audio(
    original,
    sr,
    settings
):
    x = original.copy()
    stages = []

    # Stage 1: Noise reduction
    if settings["noise_reduction"]:
        x = spectral_gate(
            x,
            sr,
            settings["noise_strength"],
            settings["noise_seconds"]
        )

        stages.append(
            "Spectral noise gate"
        )

    # Stage 2: Equalization
    x = equalize(
        x,
        sr,
        settings["low_gain"],
        settings["mid_gain"],
        settings["high_gain"]
    )

    if any(
        abs(settings[key]) > 0.01
        for key in (
            "low_gain",
            "mid_gain",
            "high_gain"
        )
    ):
        stages.append(
            "3-band EQ"
        )

    # Stage 3: Dynamic range compression
    if settings["compressor"]:
        x = compressor(
            x,
            sr,
            settings["threshold"],
            settings["ratio"],
            settings["attack_ms"],
            settings["release_ms"],
            settings["makeup_db"]
        )

        stages.append(
            "Dynamic range compressor"
        )

    # Stage 4: RMS normalization
    x, norm_gain = normalize_rms(
        x,
        settings["target_rms"],
        settings["max_gain_db"]
    )

    stages.append(
        f"RMS normalization ({norm_gain:+.1f} dB gain)"
    )

    # Stage 5: Peak limiting
    if settings["limiter"]:
        x = peak_limiter(
            x,
            settings["peak_limit"]
        )

        stages.append(
            "Peak limiter"
        )

    # Prevent invalid samples from reaching the output.
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    x = np.clip(
        x,
        -1.0,
        1.0
    )

    return x, stages


# --------------------------------------------------
# WAV EXPORT
# --------------------------------------------------

def audio_wav_bytes(
    audio,
    sr,
    subtype="PCM_16"
):
    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        sr,
        format="WAV",
        subtype=subtype
    )

    return buffer.getvalue()


# --------------------------------------------------
# WAVEFORM VISUALIZATION
# --------------------------------------------------

def plot_waveforms(
    original,
    processed,
    sr
):
    step = max(
        1,
        len(original) // 12000
    )

    times = (
        np.arange(
            0,
            len(original),
            step
        ) / sr
    )

    original_mono = np.mean(
        original,
        axis=1
    )

    processed_mono = np.mean(
        processed,
        axis=1
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True
    )

    axes[0].plot(
        times,
        original_mono[::step],
        linewidth=0.7
    )

    axes[0].set_title(
        "Original waveform"
    )

    axes[1].plot(
        times,
        processed_mono[::step],
        linewidth=0.7
    )

    axes[1].set_title(
        "Processed waveform"
    )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    for ax in axes:
        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            True,
            alpha=0.25
        )

    fig.tight_layout()

    return fig


# --------------------------------------------------
# FREQUENCY SPECTRUM VISUALIZATION
# --------------------------------------------------

def plot_spectrum(
    original,
    processed,
    sr
):
    def spectrum(audio):
        mono = np.mean(
            audio,
            axis=1
        )

        n = len(mono)

        if n == 0:
            return (
                np.array([0]),
                np.array([-120])
            )

        window = np.hanning(n)

        spec = np.abs(
            np.fft.rfft(
                mono * window
            )
        )

        frequencies = np.fft.rfftfreq(
            n,
            d=1.0 / sr
        )

        spec_db = 20.0 * np.log10(
            np.maximum(
                spec / max(1, n),
                1e-8
            )
        )

        return frequencies, spec_db

    f1, s1 = spectrum(original)
    f2, s2 = spectrum(processed)

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(
        f1,
        s1,
        label="Original",
        linewidth=0.8
    )

    ax.plot(
        f2,
        s2,
        label="Processed",
        linewidth=0.8
    )

    ax.set_xlim(
        20,
        min(sr / 2, 20000)
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Frequency (Hz, logarithmic scale)"
    )

    ax.set_ylabel(
        "Magnitude (relative dB)"
    )

    ax.set_title(
        "Frequency spectrum comparison"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    return fig


# --------------------------------------------------
# METRICS DASHBOARD
# --------------------------------------------------

def metric_table(
    original,
    processed,
    sr
):
    duration = len(original) / sr

    orig_rms = rms_db(original)
    proc_rms = rms_db(processed)

    orig_peak = peak_db(original)
    proc_peak = peak_db(processed)

    orig_clip = count_clipped_samples(
        original
    )

    proc_clip = count_clipped_samples(
        processed
    )

    return {
        "Duration (seconds)": f"{duration:.2f}",
        "Sample rate": f"{sr:,} Hz",
        "Channels": str(original.shape[1]),
        "Original RMS": f"{orig_rms:.2f} dBFS",
        "Processed RMS": f"{proc_rms:.2f} dBFS",
        "RMS change": f"{proc_rms - orig_rms:+.2f} dB",
        "Original peak": f"{orig_peak:.2f} dBFS",
        "Processed peak": f"{proc_peak:.2f} dBFS",
        "Original clipped samples": f"{orig_clip:,}",
        "Processed clipped samples": f"{proc_clip:,}",
    }


# --------------------------------------------------
# AUDIO UPLOAD
# --------------------------------------------------

uploaded = st.file_uploader(
    "Choose an audio file",
    type=[
        "wav",
        "flac",
        "ogg",
        "aiff",
        "aif"
    ]
)

if uploaded is None:
    st.info(
        "Start by uploading a WAV, FLAC, OGG, or AIFF file."
    )


    st.stop()


try:
    data, sample_rate = sf.read(
        uploaded,
        dtype="float32",
        always_2d=True
    )

    original = to_stereo_shape(data)

    if len(original) < 2:
        st.error(
            "The audio file is too short to process."
        )

        st.stop()

except Exception as exc:
    st.error(
        f"Could not read this audio file: {exc}"
    )

    st.stop()


if len(original) > sample_rate * 60 * 10:
    st.warning(
        "This file is longer than 10 minutes. "
        "Processing may take a while."
    )


# --------------------------------------------------
# PRESET SELECTOR AND SIDEBAR DSP CONTROLS
# --------------------------------------------------

# Widget keys let the preset callback update controls before they are rendered.
PRESET_WIDGET_KEYS = {
    "target_rms": "target_rms",
    "noise_reduction": "noise_reduction",
    "noise_strength": "noise_strength",
    "compressor": "compressor_enabled",
    "threshold": "threshold",
    "ratio": "ratio",
    "limiter": "limiter_enabled",
    "peak_limit": "peak_limit",
    "low_gain": "low_gain",
    "mid_gain": "mid_gain",
    "high_gain": "high_gain",
}

DEFAULT_WIDGET_VALUES = {
    "target_rms": -18.0,
    "noise_reduction": True,
    "noise_strength": 1.3,
    "compressor_enabled": True,
    "threshold": -24.0,
    "ratio": 3.0,
    "limiter_enabled": True,
    "peak_limit": -1.0,
    "low_gain": 0.0,
    "mid_gain": 0.0,
    "high_gain": 0.0,
}


def apply_selected_preset():
    selected = st.session_state.get("preset_name", "Speech / Podcast")
    preset_settings = PRESETS.get(selected, {})
    for preset_key, widget_key in PRESET_WIDGET_KEYS.items():
        if preset_key in preset_settings:
            st.session_state[widget_key] = preset_settings[preset_key]


# Initialize the controls from the default preset on the first app run.
initial_preset = PRESETS["Speech / Podcast"]
for preset_key, widget_key in PRESET_WIDGET_KEYS.items():
    if widget_key not in st.session_state:
        st.session_state[widget_key] = initial_preset.get(
            preset_key, DEFAULT_WIDGET_VALUES[widget_key]
        )

preset_col, info_col = st.columns([1, 2])
with preset_col:
    preset_name = st.selectbox(
        "Preset",
        list(PRESETS.keys()),
        index=1,
        key="preset_name",
        on_change=apply_selected_preset,
    )
with info_col:
    st.caption(
        "Selecting a preset updates the DSP controls. You can then adjust any setting manually."
    )

with st.sidebar:
    st.header("DSP controls")

    st.subheader("1. Loudness normalization")
    target_rms = st.slider(
        "Target RMS level (dBFS)", -30.0, -10.0,
        step=1.0, key="target_rms"
    )
    max_gain_db = st.slider(
        "Maximum gain boost (dB)", 0.0, 30.0, 18.0, 1.0,
        key="max_gain_db"
    )

    st.subheader("2. Noise reduction")
    noise_reduction = st.checkbox(
        "Enable spectral noise gate", key="noise_reduction"
    )
    noise_strength = st.slider(
        "Noise reduction strength", 0.2, 3.0,
        step=0.1, key="noise_strength"
    )
    noise_seconds = st.slider(
        "Noise profile duration (seconds)", 0.1, 2.0, 0.5, 0.1,
        key="noise_seconds"
    )

    st.subheader("3. Dynamic range compressor")
    compressor_enabled = st.checkbox(
        "Enable compressor", key="compressor_enabled"
    )
    threshold = st.slider(
        "Compressor threshold (dBFS)", -40.0, -6.0,
        step=1.0, key="threshold"
    )
    ratio = st.slider(
        "Compression ratio", 1.0, 8.0,
        step=0.5, key="ratio"
    )
    attack_ms = st.slider(
        "Attack (ms)", 1.0, 100.0, 10.0, 1.0,
        key="attack_ms"
    )
    release_ms = st.slider(
        "Release (ms)", 20.0, 500.0, 120.0, 10.0,
        key="release_ms"
    )
    makeup_db = st.slider(
        "Compressor makeup gain (dB)", 0.0, 12.0, 0.0, 0.5,
        key="makeup_db"
    )

    st.subheader("4. Peak limiter")
    limiter_enabled = st.checkbox(
        "Enable peak limiter", key="limiter_enabled"
    )
    peak_limit = st.slider(
        "Peak ceiling (dBFS)", -6.0, -0.1,
        step=0.1, key="peak_limit"
    )

    st.subheader("5. Three-band EQ")
    low_gain = st.slider(
        "Bass / low gain (dB)", -12.0, 12.0,
        step=0.5, key="low_gain"
    )
    mid_gain = st.slider(
        "Midrange gain (dB)", -12.0, 12.0,
        step=0.5, key="mid_gain"
    )
    high_gain = st.slider(
        "Treble / high gain (dB)", -12.0, 12.0,
        step=0.5, key="high_gain"
    )


# --------------------------------------------------
# COLLECT SETTINGS
# --------------------------------------------------

settings = {
    "target_rms": target_rms,
    "max_gain_db": max_gain_db,
    "noise_reduction": noise_reduction,
    "noise_strength": noise_strength,
    "noise_seconds": noise_seconds,
    "compressor": compressor_enabled,
    "threshold": threshold,
    "ratio": ratio,
    "attack_ms": attack_ms,
    "release_ms": release_ms,
    "makeup_db": makeup_db,
    "limiter": limiter_enabled,
    "peak_limit": peak_limit,
    "low_gain": low_gain,
    "mid_gain": mid_gain,
    "high_gain": high_gain,
}


# --------------------------------------------------
# CACHED AUDIO PROCESSING
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_process(
    file_bytes,
    sr,
    settings_tuple
):
    decoded, _ = sf.read(
        io.BytesIO(file_bytes),
        dtype="float32",
        always_2d=True
    )

    decoded = to_stereo_shape(
        decoded
    )

    setting_dict = dict(
        settings_tuple
    )

    return process_audio(
        decoded,
        sr,
        setting_dict
    )


settings_tuple = tuple(
    sorted(settings.items())
)


try:
    with st.spinner(
        "Processing audio with the selected DSP stages..."
    ):
        processed, stages = cached_process(
            uploaded.getvalue(),
            sample_rate,
            settings_tuple
        )

except Exception as exc:
    st.error(
        f"Processing failed: {exc}"
    )

    with st.expander(
        "Show technical error details"
    ):
        st.exception(exc)

    st.stop()


# --------------------------------------------------
# PLAYBACK COMPARISON
# --------------------------------------------------

st.success(
    "Processing complete"
)

st.write(
    "**Processing chain:** "
    + " → ".join(stages)
)
st.caption(
    "The RMS target is approximate. If peak protection reduces the signal, "
    "the final RMS level may be lower than the selected target."
)

left, right = st.columns(2)

with left:
    st.subheader(
        "A · Original"
    )

    st.audio(
        audio_wav_bytes(
            original,
            sample_rate
        ),
        format="audio/wav"
    )

    st.caption(
        f"RMS: {rms_db(original):.2f} dBFS "
        f"• Peak: {peak_db(original):.2f} dBFS"
    )


with right:
    st.subheader(
        "B · Processed"
    )

    st.audio(
        audio_wav_bytes(
            processed,
            sample_rate
        ),
        format="audio/wav"
    )

    st.caption(
        f"RMS: {rms_db(processed):.2f} dBFS "
        f"• Peak: {peak_db(processed):.2f} dBFS"
    )


# --------------------------------------------------
# ANALYSIS DASHBOARD
# --------------------------------------------------

st.subheader(
    "Analysis dashboard"
)

metrics = metric_table(
    original,
    processed,
    sample_rate
)

metric_cols = st.columns(5)

metric_keys = [
    ("Original RMS", "Original RMS"),
    ("Processed RMS", "Processed RMS"),
    ("Original peak", "Original peak"),
    ("Processed peak", "Processed peak"),
    (
        "Processed clipped samples",
        "Processed clipped samples"
    ),
]

for col, (key, label) in zip(
    metric_cols,
    metric_keys
):
    with col:
        st.metric(
            label,
            metrics[key]
        )


with st.expander(
    "View all metrics"
):
    st.table({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values())
    })


# --------------------------------------------------
# VISUALIZATION TABS
# --------------------------------------------------

tab_wave, tab_spectrum, tab_info = st.tabs([
    "Waveforms",
    "Frequency spectrum",
    "DSP notes"
])


with tab_wave:
    waveform_fig = plot_waveforms(
        original,
        processed,
        sample_rate
    )

    st.pyplot(
        waveform_fig,
        clear_figure=True
    )

    plt.close(waveform_fig)


with tab_spectrum:
    spectrum_fig = plot_spectrum(
        original,
        processed,
        sample_rate
    )

    st.pyplot(
        spectrum_fig,
        clear_figure=True
    )

    plt.close(spectrum_fig)


with tab_info:
    st.markdown("""
    - **RMS normalization:** adjusts the overall average signal
      level toward the selected target. It does not make every
      moment equally loud.

    - **Spectral noise gate:** estimates the noise profile from
      the beginning of the recording. It works best when that
      section contains background noise without speech or music.

    - **Compressor:** reduces gain when the signal envelope
      exceeds the threshold. A 3:1 ratio means that a 3 dB rise
      above the threshold becomes approximately a 1 dB rise
      at the output.

    - **Peak limiter:** if a peak exceeds the selected ceiling, this
      simple peak-protection stage reduces the overall signal gain to
      keep the peak at the ceiling. It is not a full dynamic limiter
      and cannot restore audio that was already clipped.

    - **Three-band EQ:** adjusts the low, mid, and high
      frequency bands. EQ changes can also affect RMS and
      peak levels.

    - **Noise metric limitation:** this app does not claim an
      objective noise-floor improvement score. A reliable
      measurement requires a known noise-only segment.
    """)


# --------------------------------------------------
# WAV EXPORT
# --------------------------------------------------

st.subheader(
    "Export"
)

export_col1, export_col2 = st.columns(
    [1, 2]
)

with export_col1:
    bit_depth = st.selectbox(
        "WAV bit depth",
        [
            "16-bit PCM",
            "24-bit PCM",
            "32-bit float"
        ],
        index=0
    )


subtype_map = {
    "16-bit PCM": "PCM_16",
    "24-bit PCM": "PCM_24",
    "32-bit float": "FLOAT"
}


try:
    wav_data = audio_wav_bytes(
        processed,
        sample_rate,
        subtype_map[bit_depth]
    )

    st.download_button(
        "⬇️ Download processed WAV",
        data=wav_data,
        file_name="audiolab_processed.wav",
        mime="audio/wav",
        use_container_width=True
    )

except Exception as exc:
    st.error(
        f"Could not create the WAV export: {exc}"
    )


st.caption(
    "DSP educational demo. Always listen to the result because "
    "aggressive noise reduction or compression can introduce artifacts."
)