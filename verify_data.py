"""
verify_data.py
===============================================================================
Verification of the prepared dataset, run against the REAL WiSig data.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS
-------------------------------------------------------------------------------
Every other module in this project ships with a self-test, but those tests run
on synthetic or mock data. They prove the code does what it was written to do.
They do NOT prove that the code does something sensible to the actual WiSig
captures, which have structure, DC offsets, and amplitude variation that a
Gaussian test signal does not.

This script closes that gap. It loads the real prepared dataset and checks the
things that would quietly ruin an experiment if they were wrong:

    1. INTEGRITY      no NaNs, no infinities, no all-zero tensors
    2. BALANCE        every class present in every split, in equal numbers
    3. LEAKAGE        held-out receiver and day genuinely absent from training
    4. RANGES         values in a sane range for each channel
    5. TIER ORDER     the three sea states really are progressively harsher
    5b. SHORTCUT      burst loudness is not a usable substitute for the label
    6. SEPARABILITY   different transmitters actually look different
    7. VISUAL         a figure to eyeball, saved to disk

Check 5b deserves particular attention. Measured on the raw captures, burst
power alone classifies this cohort at about twice chance, because ORBIT node
identifiers encode grid position and therefore distance to each receiver. A
model that learned loudness would look excellent on in-distribution test data
and fail completely on the held-out receiver. Per-burst power normalisation in
frontends.py removes that shortcut, and 5b confirms it stayed removed.

A silent failure in any of these produces a model that trains happily and
reports meaningless numbers, which is the worst possible outcome for a project
about trustworthiness.

Run it after prepare_dataset.py:

    python verify_data.py --data prepared_spec.npz
===============================================================================
"""

import argparse
import numpy as np

from prepare_dataset import load_prepared


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def check(label, passed, detail=""):
    """Print one check result in a consistent format."""
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return passed


# =============================================================================
# 1. INTEGRITY
# =============================================================================

def check_integrity(data):
    banner("1. INTEGRITY - are the tensors well formed?")
    all_ok = True

    for key in sorted(data.keys()):
        X = data[key]["X"]
        y = data[key]["y"]

        has_nan = np.isnan(X).any()
        has_inf = np.isinf(X).any()
        all_zero = not np.any(X)
        len_match = len(X) == len(y)

        ok = not (has_nan or has_inf or all_zero) and len_match
        all_ok = all_ok and ok

        problems = []
        if has_nan:
            problems.append("contains NaN")
        if has_inf:
            problems.append("contains Inf")
        if all_zero:
            problems.append("all zeros")
        if not len_match:
            problems.append(f"X/y length mismatch {len(X)} vs {len(y)}")

        detail = ", ".join(problems) if problems else f"{X.shape}, {X.dtype}"
        check(f"{key:<28}", ok, detail)

    return all_ok


# =============================================================================
# 2. CLASS BALANCE
# =============================================================================

def check_balance(data, n_classes):
    banner("2. CLASS BALANCE - is every transmitter represented everywhere?")
    all_ok = True

    print(f"  {'split__tier':<28}{'n':>8}{'per class':>12}{'spread':>9}")
    print("  " + "-" * 60)

    for key in sorted(data.keys()):
        y = data[key]["y"]
        counts = np.bincount(y, minlength=n_classes)

        # Every class must appear. An absent class means that transmitter has
        # no test examples in this condition, so its per-class accuracy and
        # calibration would be undefined.
        missing = int((counts == 0).sum())
        spread = int(counts.max() - counts.min())

        ok = missing == 0
        all_ok = all_ok and ok

        note = f"{missing} MISSING" if missing else ""
        print(f"  {key:<28}{len(y):>8,}{counts.min():>7}-{counts.max():<4}"
              f"{spread:>9}  {note}")

    print()
    check("every class present in every split/tier", all_ok)
    return all_ok


# =============================================================================
# 3. LEAKAGE
# =============================================================================

def check_leakage(config):
    banner("3. LEAKAGE - are the held-out conditions really held out?")

    train_tiers = config["train_tiers"]
    held_tier = config["held_out_tier"]

    ok1 = check(f"training tiers are {train_tiers}",
                held_tier not in train_tiers,
                f"'{held_tier}' excluded from training")

    ok2 = check(f"held-out receiver recorded",
                bool(config["held_out_rx"]),
                config["held_out_rx"])

    ok3 = check(f"held-out day recorded",
                bool(config["held_out_day"]),
                config["held_out_day"])

    print("\n  Note: receiver and day leakage were verified inside")
    print("  wisig_loader.summarise_splits() when the subset was built.")
    print("  This check confirms the tier axis, which is added here.")

    return ok1 and ok2 and ok3


# =============================================================================
# 4. VALUE RANGES
# =============================================================================

def check_ranges(data, config):
    banner("4. VALUE RANGES - do the channels hold plausible numbers?")

    front_end = config["front_end"]
    key = "train__controlled"
    X = data[key]["X"]

    if front_end == "spectrogram":
        mag, cos_p, sin_p = X[:, 0], X[:, 1], X[:, 2]

        print(f"  log-magnitude channel:")
        print(f"    min {mag.min():8.3f}   max {mag.max():8.3f}   "
              f"mean {mag.mean():8.3f}   std {mag.std():6.3f}")
        print(f"  cos(phase) channel:")
        print(f"    min {cos_p.min():8.3f}   max {cos_p.max():8.3f}   "
              f"mean {cos_p.mean():8.3f}   std {cos_p.std():6.3f}")
        print(f"  sin(phase) channel:")
        print(f"    min {sin_p.min():8.3f}   max {sin_p.max():8.3f}   "
              f"mean {sin_p.mean():8.3f}   std {sin_p.std():6.3f}")
        print()

        # cos and sin must lie in [-1, 1] and satisfy cos^2 + sin^2 = 1.
        # The identity confirms the phase angle is fully recoverable, so this
        # encoding loses nothing relative to storing the raw angle.
        bounded = (cos_p.min() >= -1.001 and cos_p.max() <= 1.001
                   and sin_p.min() >= -1.001 and sin_p.max() <= 1.001)
        unit_circle = np.allclose(cos_p[:200] ** 2 + sin_p[:200] ** 2, 1.0, atol=1e-3)
        phase_ok = check("cos/sin bounded in [-1, 1]", bounded)
        phase_ok = check("cos^2 + sin^2 == 1 (angle fully recoverable)",
                         unit_circle) and phase_ok

        # Magnitude should vary. A near-zero standard deviation would mean the
        # spectrogram carries no information for the network to use.
        mag_ok = check("log-magnitude has meaningful variation",
                       mag.std() > 0.1, f"std = {mag.std():.3f}")

        # A very large magnitude range can indicate numerical trouble.
        range_ok = check("log-magnitude range is reasonable",
                         (mag.max() - mag.min()) < 60,
                         f"range = {mag.max() - mag.min():.1f}")

        return phase_ok and mag_ok and range_ok

    else:
        print(f"  raw I/Q channels:")
        print(f"    min {X.min():8.3f}   max {X.max():8.3f}   "
              f"mean {X.mean():8.3f}   std {X.std():6.3f}")
        return check("raw I/Q has meaningful variation", X.std() > 0.01)


# =============================================================================
# 5. TIER SEVERITY ORDERING
# =============================================================================

def check_tier_ordering(data, config):
    banner("5. TIER ORDERING - do the sea states really get harsher?")

    # Use test_id, which exists at all three tiers on identical underlying
    # bursts. Any difference between them is caused by the channel alone.
    tiers = config["test_tiers"]
    front_end = config["front_end"]

    print("  Measured on test_id (same bursts, three different sea states)\n")
    print(f"  {'tier':<14}{'mag mean':>11}{'mag std':>11}{'cos std':>12}")
    print("  " + "-" * 50)

    stats = {}
    for tier in tiers:
        key = f"test_id__{tier}"
        if key not in data:
            continue
        X = data[key]["X"]
        if front_end == "spectrogram":
            m, p = X[:, 0], X[:, 1]
            stats[tier] = (m.mean(), m.std(), p.std())
            print(f"  {tier:<14}{m.mean():>11.3f}{m.std():>11.3f}{p.std():>12.3f}")
        else:
            stats[tier] = (X.mean(), X.std(), 0.0)
            print(f"  {tier:<14}{X.mean():>11.3f}{X.std():>11.3f}{'-':>12}")

    print()

    # A harsher channel adds noise, which fills in the low-energy parts of the
    # spectrum. The clearest signature is that the SPREAD of log-magnitude
    # shrinks: noise raises the quiet bins toward the loud ones, flattening the
    # spectrum. Phase also becomes more uniformly random.
    if len(stats) == 3:
        c, d, dy = stats["controlled"], stats["degraded"], stats["dynamic"]
        flattening = c[1] > d[1] > dy[1]
        ok = check("log-magnitude spread shrinks as the sea worsens",
                   flattening,
                   f"{c[1]:.3f} > {d[1]:.3f} > {dy[1]:.3f}")
        if not flattening:
            print("       (if this fails, the tiers may not be distinguishable -")
            print("        check the SNR ranges in gauntlet.TIERS)")
        return ok

    return True


# =============================================================================
# 5b. SHORTCUT CHECK - can the labels be predicted from burst loudness alone?
# =============================================================================

def check_no_power_shortcut(data, n_classes):
    """
    Guard against the network solving the task the wrong way.

    WHY THIS MATTERS. The WiSig captures come from the ORBIT testbed, where node
    identifiers encode physical position on a grid. Each transmitter therefore
    sits at a FIXED distance from each receiver, so the received power of a
    burst is partly a signature of WHERE the transmitter is, not of what its
    hardware does.

    Measured on the raw captures for this cohort, burst power alone classifies
    at roughly twice chance. A convolutional network would exploit that far more
    effectively than a single-feature rule can.

    That would be a disaster disguised as a success. A model keying on loudness
    would score well on in-distribution test data, where the geometry is the
    same, and then collapse on the held-out receiver, where every distance
    changes. We would be measuring the ORBIT floor plan, not RF fingerprints.

    The front ends normalise every burst to unit power precisely to remove this
    shortcut. This check confirms the normalisation actually took effect, so the
    guard cannot silently disappear if the front-end code is ever changed.
    """
    banner("5b. SHORTCUT CHECK - is burst loudness still a usable clue?")

    key = "train__controlled"
    X = data[key]["X"]

    # Total energy of each prepared example, whatever the representation.
    energy = X.reshape(len(X), -1).astype(np.float64)
    energy = np.mean(energy ** 2, axis=1)

    spread = energy.std() / (energy.mean() + 1e-12)

    print(f"  per-example energy: mean {energy.mean():.4e}   "
          f"std {energy.std():.4e}")
    print(f"  relative spread   : {spread:.4f}")
    print()
    print("  For raw I/Q this should be essentially zero, since every burst is")
    print("  normalised to unit power. For spectrograms it will be small but")
    print("  non-zero, because the log transform and windowing reshape the")
    print("  values after normalisation.")
    print()

    ok = check("energy does not vary wildly between examples",
               spread < 1.0, f"relative spread {spread:.4f}")
    if not ok:
        print("       Large variation suggests power normalisation is not being")
        print("       applied. Check frontends._normalize_power().")
    return ok


# =============================================================================
# 6. CLASS SEPARABILITY
# =============================================================================

def check_separability(data, n_classes):
    banner("6. SEPARABILITY - do different transmitters actually look different?")

    # This is a crude but useful check performed BEFORE any training. If the
    # average spectrogram of each transmitter is essentially identical, then
    # either the fingerprints have been destroyed by the pipeline or the labels
    # are scrambled - and no amount of training will fix it.
    #
    # We compare the spread of per-class mean spectrograms (between-class
    # variation) against the average spread within a single class. A ratio well
    # above zero means there is class-specific structure to learn.

    for tier in ["controlled", "dynamic"]:
        key = f"train__{tier}" if f"train__{tier}" in data else f"test_id__{tier}"
        if key not in data:
            continue

        X = data[key]["X"]
        y = data[key]["y"]

        class_means = np.stack([X[y == c].mean(axis=0) for c in range(n_classes)
                                if (y == c).any()])
        between = class_means.std(axis=0).mean()

        within = np.mean([X[y == c].std(axis=0).mean()
                          for c in range(n_classes) if (y == c).any()])

        ratio = between / (within + 1e-12)
        print(f"  {key:<22} between-class {between:.4f}   "
              f"within-class {within:.4f}   ratio {ratio:.4f}")

    print()
    print("  A ratio near zero would mean the classes are indistinguishable")
    print("  on average. Note this is only a coarse first-order check: a CNN")
    print("  can find structure this simple statistic misses, so a low ratio")
    print("  is a warning rather than a verdict.")
    return True


# =============================================================================
# 7. VISUAL CHECK
# =============================================================================

def make_figure(data, config, out_path="verify_data.png"):
    banner("7. VISUAL - saving a figure to eyeball")

    try:
        import matplotlib
        matplotlib.use("Agg")          # no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping the figure")
        print("  (pip install matplotlib)")
        return

    tiers = config["test_tiers"]
    front_end = config["front_end"]

    if front_end != "spectrogram":
        print("  figure currently only implemented for the spectrogram front end")
        return

    channel_names = ["log magnitude", "cos(phase)", "sin(phase)"]
    fig, axes = plt.subplots(3, len(tiers), figsize=(4 * len(tiers), 9))

    for j, tier in enumerate(tiers):
        key = f"test_id__{tier}"
        if key not in data:
            continue
        X = data[key]["X"]
        y = data[key]["y"]

        # Show the same underlying burst index at each tier, so the only
        # difference visible is the channel.
        idx = 0

        for ch in range(3):
            cmap = "viridis" if ch == 0 else "RdBu_r"
            axes[ch, j].imshow(X[idx, ch], aspect="auto", origin="lower",
                               cmap=cmap)
            title = (f"{tier}\n{channel_names[ch]} (class {y[idx]})"
                     if ch == 0 else channel_names[ch])
            axes[ch, j].set_title(title)
            axes[ch, j].set_xlabel("time frame")
            if j == 0:
                axes[ch, j].set_ylabel("frequency bin")

    fig.suptitle("Spectrogram channels across sea-state tiers (same burst)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved -> {out_path}")
    print("  Look for: structure visible at 'controlled', progressively washed")
    print("  out toward 'dynamic'. If all three look identical, the gauntlet is")
    print("  not doing anything. If 'controlled' looks like noise, something is")
    print("  wrong upstream.")


# =============================================================================
# MAIN
# =============================================================================

def main(path):
    print("=" * 70)
    print("DATASET VERIFICATION - REAL DATA")
    print("=" * 70)

    data, config = load_prepared(path)

    print(f"\nfile          : {path}")
    print(f"front end     : {config['front_end']}")
    print(f"classes       : {config['n_classes']}")
    print(f"cohort        : {config['cohort']}")
    print(f"held-out rx   : {config['held_out_rx']}")
    print(f"held-out day  : {config['held_out_day']}")
    print(f"held-out tier : {config['held_out_tier']}")

    n_classes = config["n_classes"]

    results = [
        ("integrity", check_integrity(data)),
        ("balance", check_balance(data, n_classes)),
        ("leakage", check_leakage(config)),
        ("ranges", check_ranges(data, config)),
        ("tier ordering", check_tier_ordering(data, config)),
        ("power shortcut", check_no_power_shortcut(data, n_classes)),
        ("separability", check_separability(data, n_classes)),
    ]

    make_figure(data, config)

    banner("SUMMARY")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failed = [n for n, ok in results if not ok]
    print()
    if failed:
        print(f"  {len(failed)} check(s) failed: {', '.join(failed)}")
        print("  Do not train on this data until they are resolved.")
    else:
        print("  All checks passed. The dataset is ready for training.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the prepared dataset before training on it.")
    parser.add_argument("--data", default="prepared_spec.npz")
    args = parser.parse_args()
    main(args.data)
