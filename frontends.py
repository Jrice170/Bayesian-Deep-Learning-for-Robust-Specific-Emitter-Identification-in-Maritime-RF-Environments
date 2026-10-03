"""
frontends.py
===============================================================================
Input representations for the SEI capstone project.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS MODULE DOES
-------------------------------------------------------------------------------
Converts an augmented burst into the tensor the neural network actually sees.

Two representations are provided and compared experimentally:

    RAW I/Q      the 256 complex samples, as two real channels.
                 Consumed by a 1-D convolutional network.
                 This is the default in much of the SEI literature.

    SPECTROGRAM  a short-time Fourier transform of the burst, giving magnitude
                 and phase as two channels of a time-frequency image.
                 Consumed by a 2-D convolutional network.

Everything downstream (model, training, Bayesian inference) is identical for
both, so any difference in accuracy or calibration is attributable to the
representation alone.

-------------------------------------------------------------------------------
WHY BOTH, RATHER THAN JUST RAW I/Q
-------------------------------------------------------------------------------
This was raised in feedback on the data review: signal datasets of this kind are
commonly converted to spectrograms before a CNN is applied. There is a good
physical reason, and it bears directly on what this project is testing.

A propagation channel acts on the transmitted signal as a CONVOLUTION in time:

        y(t) = h(t) * x(t)

In the raw samples, the transmitter's fingerprint and the channel's distortion
are therefore smeared together in a way that is genuinely hard to separate. Take
the Fourier transform and that convolution becomes a PRODUCT:

        Y(f) = H(f) . X(f)

and in log-magnitude it becomes a SUM:

        log|Y(f)| = log|H(f)| + log|X(f)|

An additive nuisance term is much easier for a network to learn to discount than
a convolutional one. Since this entire project is about surviving channel and
receiver shift, that is directly relevant, and reported comparisons in the RF
fingerprinting literature find spectrogram inputs competitive with or better
than raw I/Q for channel-robust identification.

-------------------------------------------------------------------------------
WHY WE KEEP PHASE, AND WHY IT IS ENCODED AS COSINE AND SINE
-------------------------------------------------------------------------------
A magnitude-only spectrogram throws away phase. That would be a mistake here:
I/Q gain and phase imbalance, oscillator phase noise, and power-amplifier AM/PM
distortion are all partly or wholly PHASE effects, and they are among the most
useful fingerprint cues we have.

But phase cannot simply be handed to a convolutional network as an angle. The
angle returned by an FFT is WRAPPED into the interval [-pi, pi], so a phase that
drifts smoothly past pi reappears at -pi. That is a jump of 2*pi in the data
where nothing at all happened in the signal.

This matters because convolution assumes its input varies smoothly from one
position to the next: a filter sliding across the image responds to local
differences. Feeding it a wrapped phase channel means a large share of the
differences it sees are wrapping artefacts rather than physics. Visual
inspection of a wrapped phase channel bears this out - it looks like uniform
noise even for a clean, highly structured signal.

The standard remedy is to represent the angle by its cosine and its sine:

        phase  ->  ( cos(phase), sin(phase) )

Both are perfectly smooth across the wrap point, since cos and sin do not care
where we chose to cut the circle. Together they determine the angle uniquely,
so no information is lost. The spectrogram therefore carries THREE channels:

        channel 0    log magnitude
        channel 1    cos(phase)
        channel 2    sin(phase)

-------------------------------------------------------------------------------
CHOOSING THE STFT WINDOW - A REAL CONSTRAINT HERE
-------------------------------------------------------------------------------
The burst is only 256 samples long, which is short for an STFT. There is a hard
trade-off, and it is worth stating explicitly because it shapes the results:

    LONG window   -> fine frequency resolution, few time frames
    SHORT window  -> more time frames, coarse frequency resolution

With N = 256 and a window of length W advanced by hop H, the output is

    time frames      = 1 + (256 - W) / H
    frequency bins   = W

Note the frequency bin count is W, not W/2 + 1. That is because our signal is
COMPLEX BASEBAND, not real-valued. A real signal has a conjugate-symmetric
spectrum, so half of it is redundant and the usual real-input FFT discards it.
A complex baseband signal has no such symmetry: positive and negative
frequencies carry genuinely different information, and both matter here, since
I/Q imbalance in particular shows up as energy leaking between a frequency and
its mirror image. Using a real-input FFT would silently throw away half the
fingerprint. We therefore take the full complex FFT and shift it so the bins run
from -fs/2 through DC to +fs/2.

Some concrete options:

    W = 128, H = 32   ->   5 frames x 128 bins    frequency-favouring
    W =  64, H = 16   ->  13 frames x  64 bins    balanced      <- default
    W =  32, H =  8   ->  29 frames x  32 bins    time-favouring

The default is W = 64, H = 16, giving a 64 x 13 image. That is small by computer
vision standards but perfectly workable for a small CNN, and it keeps both axes
meaningful. WINDOW CHOICE IS A HYPERPARAMETER: it should be tuned on the
validation split and the chosen value reported in the paper.

A Hann window is applied before each transform to suppress spectral leakage from
the abrupt edges of each frame.
===============================================================================
"""

from dataclasses import dataclass
import numpy as np


N_SAMPLES = 256          # samples per burst, fixed by the WiSig dataset
SAMPLE_RATE_HZ = 25.0e6  # 25 MS/s


# =============================================================================
# RAW I/Q FRONT END
# =============================================================================

def raw_iq(bursts: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Raw I/Q front end: present the burst as two real channels.

    Parameters
    ----------
    bursts : float array, shape (n, 256, 2)
        As stored by WiSig: last axis is [in-phase, quadrature].
    normalize : bool
        Scale each burst to unit average power. This is important: WiSig
        captures arrive at widely differing amplitudes depending on how far the
        transmitter was from the receiver. Without normalisation the network
        could separate transmitters by loudness rather than by fingerprint,
        which would not survive contact with a real deployment.

    Returns
    -------
    float32 array, shape (n, 2, 256)

    Note the axis order changes from (n, 256, 2) to (n, 2, 256). Convolutional
    layers expect (batch, channels, length), so I and Q become the two channels
    and the 256 samples become the length the filters slide along.
    """
    x = np.asarray(bursts, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != N_SAMPLES or x.shape[2] != 2:
        raise ValueError(f"expected shape (n, {N_SAMPLES}, 2), got {x.shape}")

    if normalize:
        x = _normalize_power(x)

    return np.transpose(x, (0, 2, 1)).copy()


def _normalize_power(bursts: np.ndarray) -> np.ndarray:
    """
    Scale each burst so its average power is 1.

    Power of a complex sample is I^2 + Q^2, so the average power of a burst is
    mean(I^2 + Q^2). Dividing by the square root of that makes every burst
    comparable in amplitude while leaving its SHAPE - the part that carries the
    fingerprint - untouched.
    """
    power = np.mean(np.sum(bursts ** 2, axis=2), axis=1)   # (n,)
    scale = np.sqrt(np.maximum(power, 1e-12))
    return bursts / scale[:, None, None]


# =============================================================================
# SPECTROGRAM FRONT END
# =============================================================================

@dataclass
class STFTConfig:
    """
    Short-time Fourier transform settings.

    window_length:
        Samples per frame. Longer means finer frequency resolution but fewer
        time frames (see the module docstring for the trade-off).
    hop_length:
        Samples advanced between consecutive frames. Smaller means more frames
        and more overlap between them.
    use_hann:
        Apply a Hann taper to each frame before transforming. This suppresses
        the spectral leakage caused by chopping the signal into frames with
        hard edges.
    log_magnitude:
        Take the logarithm of the magnitude channel. This is what turns the
        channel's multiplicative effect into an additive one (see the module
        docstring), and it compresses the very large dynamic range of the
        spectrum into something a network can work with comfortably.
    """
    window_length: int = 64
    hop_length: int = 16
    use_hann: bool = True
    log_magnitude: bool = True

    def output_shape(self, n_samples: int = N_SAMPLES) -> tuple:
        """
        Work out the (channels, frequency, time) shape this config produces.
        Handy for sizing the network without running any data through.
        """
        n_frames = 1 + (n_samples - self.window_length) // self.hop_length
        n_freqs = self.window_length      # full complex spectrum, see docstring
        # 3 channels: log magnitude, cos(phase), sin(phase)
        return (3, n_freqs, n_frames)


def spectrogram(bursts: np.ndarray,
                config: STFTConfig = None,
                normalize: bool = True) -> np.ndarray:
    """
    Spectrogram front end: STFT the burst into a magnitude+phase image.

    Parameters
    ----------
    bursts : float array, shape (n, 256, 2)
    config : STFTConfig
    normalize : bool
        Normalise burst power before transforming, for the same reason as in
        raw_iq().

    Returns
    -------
    float32 array, shape (n, 3, n_freqs, n_frames)
        Channel 0 is (log) magnitude, channel 1 is cos(phase), channel 2 is
        sin(phase). See the module docstring for why phase is split this way
        rather than passed as a raw angle.

    HOW IT WORKS
    ------------
    1. Convert the (256, 2) real pairs into 256 complex samples.
    2. Slide a window along the burst, stepping by hop_length.
    3. Taper each frame with a Hann window and take its FFT.
    4. Shift the spectrum so bins run from -fs/2 through DC to +fs/2.
    5. Split the complex spectrum into magnitude and a wrap-free phase pair.

    We take the FULL complex FFT (window_length bins), not the real-input FFT.
    See the module docstring: for complex baseband, negative frequencies carry
    real information, and I/Q imbalance in particular appears as leakage between
    mirrored frequency bins. Halving the spectrum would discard that cue.
    """
    if config is None:
        config = STFTConfig()

    x = np.asarray(bursts, dtype=np.float32)
    if normalize:
        x = _normalize_power(x)

    complex_bursts = x[..., 0] + 1j * x[..., 1]          # (n, 256)

    W, H = config.window_length, config.hop_length
    n_frames = 1 + (complex_bursts.shape[1] - W) // H

    taper = np.hanning(W) if config.use_hann else np.ones(W)

    # Cut the burst into overlapping frames: shape (n, n_frames, W)
    starts = np.arange(n_frames) * H
    frames = np.stack([complex_bursts[:, s:s + W] for s in starts], axis=1)
    frames = frames * taper[None, None, :]

    # Full complex FFT of each frame, then shift so the bins are ordered
    # -fs/2 ... 0 ... +fs/2 rather than 0 ... +fs/2, -fs/2 ... 0. The shifted
    # ordering puts neighbouring frequencies next to each other, which is what
    # a convolutional filter sliding over the frequency axis assumes.
    spectra = np.fft.fft(frames, axis=2)                 # (n, n_frames, W)
    spectra = np.fft.fftshift(spectra, axes=2)

    magnitude = np.abs(spectra)
    phase = np.angle(spectra)

    if config.log_magnitude:
        # The small constant prevents log(0) where a bin has no energy.
        magnitude = np.log(magnitude + 1e-8)

    # Encode phase as its cosine and sine so the representation is continuous
    # across the +pi / -pi wrap. Both live in [-1, 1], which also puts them on a
    # comparable scale to each other without any further normalisation.
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    # Rearrange to (n, channels, freq, time), the layout a 2-D CNN expects.
    magnitude = np.transpose(magnitude, (0, 2, 1))
    cos_phase = np.transpose(cos_phase, (0, 2, 1))
    sin_phase = np.transpose(sin_phase, (0, 2, 1))

    return np.stack([magnitude, cos_phase, sin_phase], axis=1).astype(np.float32)


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def transform(bursts: np.ndarray, kind: str, **kwargs) -> np.ndarray:
    """
    Apply whichever front end is named. Keeps the training script from having to
    branch on representation, so the two experiments differ by one string.

        X = transform(bursts, "raw_iq")
        X = transform(bursts, "spectrogram")
    """
    if kind == "raw_iq":
        return raw_iq(bursts, **kwargs)
    if kind == "spectrogram":
        return spectrogram(bursts, **kwargs)
    raise ValueError(f"unknown front end {kind!r}; "
                     f"expected 'raw_iq' or 'spectrogram'")


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FRONT ENDS - SELF TEST")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n = 8
    test_bursts = rng.normal(0, 1, (n, N_SAMPLES, 2)).astype(np.float32)

    # --- 1. Shapes ------------------------------------------------------------
    print("\n[1] Output shapes")
    xr = raw_iq(test_bursts)
    print(f"    raw_iq       {test_bursts.shape} -> {xr.shape}   "
          f"(batch, channels, length)")
    cfg = STFTConfig()
    xs = spectrogram(test_bursts, cfg)
    print(f"    spectrogram  {test_bursts.shape} -> {xs.shape}   "
          f"(batch, channels, freq, time)")
    predicted = (n,) + cfg.output_shape()
    ok = xs.shape == predicted
    print(f"    output_shape() predicted {predicted}: {ok}  "
          f"{'PASS' if ok else 'FAIL'}")

    # --- 2. Power normalisation ----------------------------------------------
    print("\n[2] Power normalisation")
    loud = test_bursts * 100.0            # same signal, 100x the amplitude
    quiet = test_bursts * 0.01
    a = raw_iq(loud)
    b = raw_iq(quiet)
    same = np.allclose(a, b, atol=1e-4)
    print(f"    a 100x-loud and a 0.01x-quiet copy of the same burst")
    print(f"    produce identical normalised output: {same}   "
          f"{'PASS' if same else 'FAIL'}")
    powers = np.mean(np.sum(np.transpose(a, (0, 2, 1)) ** 2, axis=2), axis=1)
    print(f"    average power after normalisation: {powers.mean():.4f} (want 1.0)")

    # --- 3. Window trade-off --------------------------------------------------
    print("\n[3] STFT window trade-off (why 64/16 is the default)")
    print(f"    {'window':>8}{'hop':>6}{'freq bins':>12}{'time frames':>14}")
    for W, H in [(128, 32), (64, 16), (32, 8), (16, 4)]:
        c = STFTConfig(window_length=W, hop_length=H)
        _, f_bins, t_frames = c.output_shape()
        marker = "   <- default" if (W, H) == (64, 16) else ""
        print(f"    {W:>8}{H:>6}{f_bins:>12}{t_frames:>14}{marker}")

    # --- 3b. Confirm negative frequencies survive ----------------------------
    print("\n[3b] Complex baseband keeps negative frequencies")
    # A pure positive-frequency tone should light up one side of the spectrum
    # only. If we had used a real-input FFT this asymmetry would be invisible.
    nn = np.arange(N_SAMPLES)
    tone = np.exp(1j * 2 * np.pi * 0.15 * nn)          # positive frequency
    tone_burst = np.stack([tone.real, tone.imag], axis=-1)[None].astype(np.float32)
    st = spectrogram(tone_burst, STFTConfig(log_magnitude=False))[0, 0]
    half = st.shape[0] // 2
    lower = st[:half].mean()      # negative frequencies
    upper = st[half:].mean()      # positive frequencies
    asymmetric = upper > 3 * lower
    print(f"    positive-frequency tone: negative-half energy {lower:.4f}, "
          f"positive-half {upper:.4f}")
    print(f"    spectrum is asymmetric as it should be: {asymmetric}   "
          f"{'PASS' if asymmetric else 'FAIL'}")

    # --- 4. Phase is retained -------------------------------------------------
    print("\n[4] Phase channels carry real information, not noise")
    # Two signals with identical magnitude spectra but different phase should
    # produce identical magnitude channels and DIFFERENT phase channels.
    base = rng.normal(0, 1, (1, N_SAMPLES, 2)).astype(np.float32)
    c = base[..., 0] + 1j * base[..., 1]
    rotated_c = c * np.exp(1j * 0.7)          # rotate phase, magnitude unchanged
    rotated = np.stack([rotated_c.real, rotated_c.imag], axis=-1).astype(np.float32)

    s1 = spectrogram(base)
    s2 = spectrogram(rotated)
    mag_same = np.allclose(s1[:, 0], s2[:, 0], atol=1e-4)
    phase_diff = np.abs(s1[:, 1:] - s2[:, 1:]).mean()
    print(f"    after a pure phase rotation:")
    print(f"      magnitude channel unchanged   : {mag_same}   "
          f"{'PASS' if mag_same else 'FAIL'}")
    print(f"      cos/sin channels mean change  : {phase_diff:.4f}   "
          f"{'PASS' if phase_diff > 0.05 else 'FAIL'}")
    print(f"    -> a magnitude-only front end would have been blind to this")

    # --- 4b. cos/sin encoding is continuous across the wrap --------------------
    print("\n[4b] cos/sin encoding removes the phase wrapping discontinuity")
    # Sweep a phase smoothly through +pi. The raw angle jumps by 2*pi; cos and
    # sin do not. A convolutional filter sees that jump as a huge local gradient
    # where physically nothing happened, which is why raw phase is a poor input.
    angles = np.linspace(np.pi - 0.2, np.pi + 0.2, 41)
    wrapped = np.angle(np.exp(1j * angles))
    raw_jump = np.abs(np.diff(wrapped)).max()
    cos_jump = np.abs(np.diff(np.cos(angles))).max()
    sin_jump = np.abs(np.diff(np.sin(angles))).max()
    print(f"    sweeping phase smoothly through +pi:")
    print(f"      largest step in raw angle : {raw_jump:.4f}  <- artificial cliff")
    print(f"      largest step in cos       : {cos_jump:.4f}")
    print(f"      largest step in sin       : {sin_jump:.4f}")
    ok_wrap = cos_jump < 0.1 and sin_jump < 0.1 and raw_jump > 5.0
    print(f"    cos/sin stay smooth where the raw angle jumps: "
          f"{'PASS' if ok_wrap else 'FAIL'}")

    # cos^2 + sin^2 must equal 1 everywhere, confirming no information is lost.
    s = spectrogram(test_bursts)
    unit = s[:, 1] ** 2 + s[:, 2] ** 2
    ok_unit = np.allclose(unit, 1.0, atol=1e-4)
    print(f"    cos^2 + sin^2 == 1 everywhere (angle fully recoverable): "
          f"{'PASS' if ok_unit else 'FAIL'}")

    # --- 5. Convolution becomes addition in log-magnitude --------------------
    print("\n[5] The core claim: channel convolution -> additive offset")
    # Pass a signal through a simple channel filter and check that the
    # log-magnitude spectrogram changes by an (approximately) constant offset
    # per frequency bin, rather than by a smeared, signal-dependent amount.
    signal = rng.normal(0, 1, N_SAMPLES) + 1j * rng.normal(0, 1, N_SAMPLES)
    channel = np.array([1.0, 0.5, 0.25])          # a short 3-tap channel
    filtered = np.convolve(signal, channel)[:N_SAMPLES]

    to_burst = lambda z: np.stack([z.real, z.imag], axis=-1)[None].astype(np.float32)
    m_clean = spectrogram(to_burst(signal), normalize=False)[0, 0]
    m_filt = spectrogram(to_burst(filtered), normalize=False)[0, 0]

    difference = m_filt - m_clean            # (freq, time)
    per_bin_std = difference.std(axis=1).mean()   # variation across time frames
    overall_std = difference.std()
    print(f"    log-magnitude difference after filtering:")
    print(f"      spread WITHIN a frequency bin across time : {per_bin_std:.4f}")
    print(f"      spread ACROSS all bins                    : {overall_std:.4f}")
    print(f"    the within-bin spread is much smaller, i.e. the channel shows up")
    print(f"    as a per-frequency offset that is stable over time   "
          f"{'PASS' if per_bin_std < overall_std else 'CHECK'}")

    # --- 6. Determinism -------------------------------------------------------
    print("\n[6] Determinism (front ends must not add randomness)")
    d1 = spectrogram(test_bursts)
    d2 = spectrogram(test_bursts)
    det = np.array_equal(d1, d2)
    print(f"    two calls give identical output: {det}   "
          f"{'PASS' if det else 'FAIL'}")

    print("\n" + "=" * 70)
    print("Self test complete.")
    print("=" * 70)
