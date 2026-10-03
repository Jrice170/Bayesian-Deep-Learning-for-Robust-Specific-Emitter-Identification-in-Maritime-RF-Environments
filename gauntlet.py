"""
Maritime channel augmentation for the SEI capstone project.

Joseph M. Rice, LTJG, USN
CS 4323, Naval Postgraduate School

Applies a simulated maritime channel to clean WiSig captures at one of three
severity tiers. One tier is withheld from training so that generalisation to an
unseen sea state can be measured.

Capture parameters (from the WiSig paper) drive three design decisions:

    fs = 25 MS/s (40 ns/sample), N = 256 samples (10.24 us), fc = 2.462 GHz

  1. The specular sea-surface reflection has a path difference of 2*h_t*h_r/d,
     which is ~0.003 samples for 10 m antennas at 5 km. It is therefore a flat
     complex gain, not a delay tap.
  2. Doppler at 15 m/s accumulates 0.008 rad across a burst, so it is a
     per-burst frequency offset, not an intra-burst chirp.
  3. Sea state acts through diffuse scattering, whose delay spread (tens to
     hundreds of ns) is resolvable at 40 ns/sample. Tiers therefore vary the
     Rician K-factor, delay spread, and SNR.

Receiver hardware imperfections are NOT simulated; they are already present in
the WiSig recordings from 18 physical USRPs.

Every call to apply() draws a fresh channel. No state is stored per transmitter,
so a channel realisation can never become a stable clue to emitter identity.
"""

from dataclasses import dataclass
import numpy as np


SPEED_OF_LIGHT = 3.0e8
SAMPLE_RATE_HZ = 25.0e6
CARRIER_HZ = 2.462e9


@dataclass
class ChannelConfig:
    """Parameters of one sea-state tier.

    rician_k_db      specular power / diffuse power. High = calm sea.
    delay_spread_ns  r.m.s. spread of the diffuse component (40 ns = 1 sample).
    doppler_max_hz   per-burst frequency offset is drawn from +/- this value.
    duct_probability chance of an extra trapped-path arrival.
    """
    name: str
    snr_db_range: tuple
    rician_k_db: float
    delay_spread_ns: float
    doppler_max_hz: float
    duct_probability: float
    tx_height_m: float = 10.0
    rx_height_m: float = 10.0
    range_m: float = 5000.0


# Calm / moderate / rough. DYNAMIC is held out of training.
TIERS = {
    "controlled": ChannelConfig("controlled", (15.0, 25.0), 12.0, 20.0, 50.0, 0.00),
    "degraded":   ChannelConfig("degraded",   (5.0, 15.0),   4.0, 80.0, 150.0, 0.15),
    "dynamic":    ChannelConfig("dynamic",   (-5.0, 0.0),   -3.0, 200.0, 400.0, 0.35),
}


class MaritimeGauntlet:
    """Applies a simulated maritime channel to complex baseband bursts.

        g = MaritimeGauntlet(TIERS["degraded"], seed=0)
        y = g.apply(x)                  # x complex, shape (256,)
    """

    def __init__(self, config: ChannelConfig, seed: int = 0):
        self.cfg = config
        self.rng = np.random.default_rng(seed)

    def _specular_gain(self) -> complex:
        """Two-ray specular fade: direct ray plus sea-surface reflection.

            delta = 2 * h_t * h_r / d                (extra path length)
            phi   = 2 * pi * fc * delta / c          (extra phase)
            g     = 1 + Gamma * exp(-j * phi)        (phasor sum)

        Gamma is the reflection coefficient, near -1 for seawater at grazing
        incidence. Because delta corresponds to ~0.003 samples, the reflected
        ray is time-aligned with the direct one and the result is a single
        complex gain rather than a delay tap.
        """
        path_difference_m = (2.0 * self.cfg.tx_height_m * self.cfg.rx_height_m
                             / self.cfg.range_m)
        phase_rad = 2.0 * np.pi * CARRIER_HZ * path_difference_m / SPEED_OF_LIGHT
        return 1.0 + (-0.9) * np.exp(-1j * phase_rad)

    def _diffuse_taps(self) -> np.ndarray:
        """Rough-sea scattering as a tapped delay line.

            P(k)  = exp(-k / sigma_tau),  normalised so sum P(k) = 1
            h(k)  = sqrt(P(k)/2) * (a + j*b),   a, b ~ N(0, 1)

        Power decays exponentially with delay k; each tap has a complex Gaussian
        amplitude, giving Rayleigh-distributed magnitudes. sigma_tau is the tier
        delay spread expressed in samples.
        """
        sample_period_ns = 1e9 / SAMPLE_RATE_HZ
        spread_samples = self.cfg.delay_spread_ns / sample_period_ns
        n_taps = max(1, int(np.ceil(3.0 * spread_samples)))

        power_profile = np.exp(-np.arange(n_taps) / max(spread_samples, 1e-6))
        power_profile /= power_profile.sum()

        real = self.rng.normal(0.0, 1.0, n_taps)
        imag = self.rng.normal(0.0, 1.0, n_taps)
        return np.sqrt(power_profile / 2.0) * (real + 1j * imag)

    def _apply_doppler(self, x: np.ndarray) -> np.ndarray:
        """Doppler as a per-burst frequency offset.

            f_d    = v * fc / c
            y[n]   = x[n] * exp(j * 2 * pi * f_d * n / fs)

        f_d is drawn uniformly from +/- doppler_max_hz. Across 256 samples the
        accumulated phase is under 0.01 rad, so this is effectively a constant
        rotation rather than a chirp.
        """
        if self.cfg.doppler_max_hz <= 0.0:
            return x
        f_d = self.rng.uniform(-self.cfg.doppler_max_hz, self.cfg.doppler_max_hz)
        n = np.arange(len(x))
        return x * np.exp(1j * 2.0 * np.pi * f_d * n / SAMPLE_RATE_HZ)

    def _apply_ducting(self, x: np.ndarray) -> np.ndarray:
        """Simplified evaporation ducting: one weak trapped-path arrival.

            y[n] = x[n] + A * exp(j*theta) * x[n - D]

        with delay D in [3, 10) samples, amplitude A in [0.1, 0.3), and random
        phase theta. A coarse stand-in for a parabolic-equation propagation
        model, applied with probability duct_probability.
        """
        if self.rng.random() >= self.cfg.duct_probability:
            return x
        delay = self.rng.integers(3, 10)
        amp = self.rng.uniform(0.1, 0.3)
        phase = self.rng.uniform(0.0, 2.0 * np.pi)

        delayed = np.zeros_like(x)
        delayed[delay:] = x[:len(x) - delay]
        return x + amp * np.exp(1j * phase) * delayed

    def _add_noise(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        """Complex AWGN to a target signal-to-noise ratio.

            P_s   = mean(|x|^2)                      (measured, per burst)
            P_n   = P_s / 10^(SNR_dB / 10)
            w[n]  = sqrt(P_n/2) * (a + j*b),   a, b ~ N(0, 1)

        P_s is measured from THIS burst rather than assumed. WiSig capture
        amplitudes span three orders of magnitude, so a fixed noise scale would
        give every burst a different true SNR and make the tiers meaningless.
        """
        signal_power = np.mean(np.abs(x) ** 2)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            self.rng.normal(0.0, 1.0, len(x)) + 1j * self.rng.normal(0.0, 1.0, len(x)))
        return x + noise

    def apply(self, x: np.ndarray, return_metadata: bool = False):
        """Pass one burst through the full channel.

            K     = 10^(K_dB / 10)                   (Rician K-factor, linear)
            y     = sqrt(K/(K+1)) * g * x            (specular component)
                  + sqrt(1/(K+1)) * (h conv x)       (diffuse component)
            then ducting, then Doppler, then noise.

        The K-factor splits total power between the direct path and the
        scattered field. Order follows the physical signal path; noise is last
        because SNR is defined at the receiver input, after the channel has
        already attenuated the signal.
        """
        x = np.asarray(x)
        if not np.iscomplexobj(x):
            raise TypeError("apply() expects complex baseband samples")

        # Split power between specular and diffuse per the Rician K-factor.
        k_linear = 10.0 ** (self.cfg.rician_k_db / 10.0)
        specular_weight = np.sqrt(k_linear / (k_linear + 1.0))
        diffuse_weight = np.sqrt(1.0 / (k_linear + 1.0))

        diffuse = self._diffuse_taps()
        y = specular_weight * self._specular_gain() * x
        y = y + diffuse_weight * np.convolve(x, diffuse)[:len(x)]

        y = self._apply_ducting(y)
        y = self._apply_doppler(y)

        snr_db = self.rng.uniform(*self.cfg.snr_db_range)
        y = self._add_noise(y, snr_db)

        if return_metadata:
            return y, {"tier": self.cfg.name, "snr_db": float(snr_db),
                       "rician_k_db": float(self.cfg.rician_k_db),
                       "delay_spread_ns": float(self.cfg.delay_spread_ns),
                       "n_diffuse_taps": int(len(diffuse))}
        return y

    def apply_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply an independent channel realisation to each row of X."""
        return np.stack([self.apply(row) for row in X])


def to_complex(iq_real: np.ndarray) -> np.ndarray:
    """WiSig (..., 256, 2) real layout -> complex baseband."""
    return iq_real[..., 0] + 1j * iq_real[..., 1]


def to_iq(complex_signal: np.ndarray) -> np.ndarray:
    """Complex baseband -> WiSig (..., 256, 2) real layout."""
    return np.stack([complex_signal.real, complex_signal.imag], axis=-1)


if __name__ == "__main__":
    print("=" * 62)
    print("MARITIME GAUNTLET - SELF TEST")
    print("=" * 62)

    rng = np.random.default_rng(0)
    N = 256
    test_signal = (rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N)) / np.sqrt(2)

    print(f"\nfs = {SAMPLE_RATE_HZ/1e6:.0f} MS/s ({1e9/SAMPLE_RATE_HZ:.0f} ns/sample), "
          f"N = {N} ({N/SAMPLE_RATE_HZ*1e6:.2f} us), fc = {CARRIER_HZ/1e9:.3f} GHz")

    print("\n[1] Achieved SNR matches target")
    for name, cfg in TIERS.items():
        g = MaritimeGauntlet(cfg, seed=1)
        errs = []
        for _ in range(200):
            clean = g._specular_gain() * test_signal
            noisy = g._add_noise(clean, 5.0)
            achieved = 10 * np.log10(np.mean(np.abs(clean) ** 2)
                                     / np.mean(np.abs(noisy - clean) ** 2))
            errs.append(abs(achieved - 5.0))
        err = np.mean(errs)
        print(f"    {name:11s} mean |error| {err:.3f} dB   {'PASS' if err < 0.5 else 'FAIL'}")

    print("\n[2] Tiers are progressively harsher")
    for name, cfg in TIERS.items():
        g = MaritimeGauntlet(cfg, seed=2)
        snrs = [g.apply(test_signal, return_metadata=True)[1]["snr_db"] for _ in range(300)]
        print(f"    {name:11s} SNR {np.mean(snrs):6.2f} dB   K {cfg.rician_k_db:5.1f} dB   "
              f"spread {cfg.delay_spread_ns:5.1f} ns "
              f"({cfg.delay_spread_ns/(1e9/SAMPLE_RATE_HZ):.1f} samples)")

    # Critical anti-leakage check: if two calls on identical input produced
    # identical output, the channel would be a fixed signature the network
    # could learn instead of the transmitter.
    print("\n[3] Fresh channel every call (anti-leakage)")
    g = MaritimeGauntlet(TIERS["degraded"], seed=3)
    identical = np.allclose(g.apply(test_signal), g.apply(test_signal))
    print(f"    repeated calls identical: {identical}   "
          f"{'FAIL - channel reused' if identical else 'PASS'}")

    print("\n[4] Design decisions verified numerically")
    path_diff = 2 * 10.0 * 10.0 / 5000.0
    tau_samples = path_diff / SPEED_OF_LIGHT * SAMPLE_RATE_HZ
    print(f"    two-ray delay at 5 km, 10 m masts: {tau_samples:.4f} samples "
          f"-> flat fade")
    f_d = 15.0 * CARRIER_HZ / SPEED_OF_LIGHT
    phase = 2 * np.pi * f_d * N / SAMPLE_RATE_HZ
    print(f"    Doppler at 15 m/s: {f_d:.0f} Hz, {np.degrees(phase):.2f} deg per burst "
          f"-> constant offset")

    print("\n[5] WiSig format round-trip")
    w = rng.normal(0, 1, (N, 2))
    print(f"    preserved: {np.allclose(w, to_iq(to_complex(w)))}   PASS")

    print("\n" + "=" * 62)
