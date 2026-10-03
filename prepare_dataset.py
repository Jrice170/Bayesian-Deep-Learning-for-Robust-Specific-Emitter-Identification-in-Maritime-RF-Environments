"""
prepare_dataset.py
===============================================================================
Builds the experiment-ready dataset for the SEI capstone project.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
-------------------------------------------------------------------------------
This is the bridge between the three building blocks and the training script.
It runs the whole data path exactly once and saves the result, so that every
training run afterwards sees IDENTICAL data.

    wisig_loader.py   real captures, split by receiver and capture day
            |
            v
    gauntlet.py       maritime channel applied at a chosen severity tier
            |
            v
    frontends.py      short-time Fourier transform -> magnitude/phase image
            |
            v
    prepared_*.npz    ready to load and train on

-------------------------------------------------------------------------------
WHY FREEZE THE AUGMENTED DATA INSTEAD OF AUGMENTING ON THE FLY
-------------------------------------------------------------------------------
It would be possible to apply the channel fresh during every training epoch,
which would expose the model to more channel variety. We deliberately do not,
for three reasons:

    1. REPRODUCIBILITY. The paper must describe a fixed dataset. Freezing it
       means the MLE run and the MAP run see exactly the same examples, so any
       difference between them is caused by the prior and nothing else.

    2. FAIR COMPARISON ACROSS DELIVERABLES. Deliverable III trains Laplace and
       Monte Carlo dropout on this same frozen data. If the data changed between
       deliverables, method comparisons would be meaningless.

    3. CALIBRATION MEASUREMENT. Reliability diagrams compare predicted
       confidence against observed accuracy on a fixed test set. A test set that
       changed every epoch would make that measurement moving target.

The channel configuration that produced the file is stored alongside it, so a
run can be reproduced from the saved settings. Note that this records the tier
PARAMETERS, not the per-burst values drawn from them: the SNR, Doppler offset
and fading realisation sampled for each individual burst are applied and then
discarded. That is a real limitation. It means accuracy can be reported against
a tier, whose SNR range is known, but not against the exact SNR of a given
burst. Recording the per-burst draws would be a small change here and would
allow a continuous accuracy-versus-SNR curve.

-------------------------------------------------------------------------------
HOW THE SEA-STATE TIERS ARE ASSIGNED
-------------------------------------------------------------------------------
This is the third axis of distribution shift, alongside held-out receiver and
held-out capture day.

    TRAINING   sees only the CONTROLLED and DEGRADED tiers.
    TESTING    is evaluated separately on all three tiers, including DYNAMIC,
               which the model has never encountered.

So the Dynamic tier plays the same role for sea state that receiver "8-8" plays
for hardware: a condition present at test time and absent from training. When we
report accuracy and calibration on Dynamic, we are asking the central question
of this project directly.

Each split is therefore materialised once per tier, giving evaluation cells like
"test_rx at Dynamic" (unfamiliar receiver AND unfamiliar sea state).
===============================================================================
"""

import argparse
import json
import numpy as np

from wisig_loader import load_subset
from gauntlet import MaritimeGauntlet, TIERS, to_complex, to_iq
from frontends import spectrogram, raw_iq, STFTConfig


# Tiers the model is allowed to train on. DYNAMIC is deliberately excluded.
TRAIN_TIERS = ["controlled", "degraded"]

# Tiers each test split is evaluated on, including the held-out one.
TEST_TIERS = ["controlled", "degraded", "dynamic"]

# Splits that come from wisig_loader.
TRAIN_SPLITS = ["train", "val"]
TEST_SPLITS = ["test_id", "test_rx", "test_day", "test_both"]


def augment(bursts: np.ndarray, tier: str, seed: int) -> np.ndarray:
    """
    Pass a stack of bursts through the maritime channel at one severity tier.

    Parameters
    ----------
    bursts : float array, shape (n, 256, 2)
        Raw captures in WiSig's real-valued layout.
    tier : str
        Key into gauntlet.TIERS.
    seed : int
        Seeds this tier's channel realisations. Different tiers and different
        splits get different seeds, so no channel realisation is ever shared
        between training and test data.

    Returns
    -------
    float array, shape (n, 256, 2)

    Note that a FRESH channel is drawn for every burst (see gauntlet.py). If the
    same channel were reused across a transmitter's bursts, the network could
    learn the channel as a stand-in for identity, which is the exact failure this
    project exists to detect.
    """
    g = MaritimeGauntlet(TIERS[tier], seed=seed)
    complex_bursts = to_complex(bursts)
    augmented = np.stack([g.apply(row) for row in complex_bursts])
    return to_iq(augmented).astype(np.float32)


def prepare(subset_path: str,
            out_path: str,
            front_end: str = "spectrogram",
            stft_window: int = 64,
            stft_hop: int = 16,
            seed: int = 0) -> None:
    """
    Run the full preparation pipeline and save the result.

    The output .npz contains one array per (split, tier) combination, named
    like "train__controlled__X" and "train__controlled__y".
    """
    print("=" * 70)
    print("PREPARING DATASET")
    print("=" * 70)

    splits = load_subset(subset_path)
    meta = splits["_meta"]
    n_classes = int(meta["n_classes"])

    print(f"\ncohort        : {n_classes} transmitters")
    print(f"held-out rx   : {meta['held_out_rx']}")
    print(f"held-out day  : {meta['held_out_day']}")
    print(f"held-out tier : dynamic  (training sees {TRAIN_TIERS})")
    print(f"front end     : {front_end}")

    stft_cfg = STFTConfig(window_length=stft_window, hop_length=stft_hop)
    if front_end == "spectrogram":
        shape = stft_cfg.output_shape()
        print(f"STFT          : window {stft_window}, hop {stft_hop} "
              f"-> {shape[1]} freq x {shape[2]} time")

    def to_tensor(bursts):
        """Apply the chosen front end."""
        if front_end == "spectrogram":
            return spectrogram(bursts, stft_cfg)
        return raw_iq(bursts)

    output = {}
    manifest = []

    # Use a distinct seed per (split, tier) so no channel realisation is ever
    # shared between two different parts of the experiment.
    seed_counter = seed

    print(f"\n{'split':<12}{'tier':<12}{'bursts':>9}{'tensor shape':>22}")
    print("-" * 70)

    for split_name in TRAIN_SPLITS + TEST_SPLITS:
        bursts = splits[split_name]["X"]
        labels = splits[split_name]["y"]

        if len(labels) == 0:
            continue

        tiers = TRAIN_TIERS if split_name in TRAIN_SPLITS else TEST_TIERS

        for tier in tiers:
            seed_counter += 1
            augmented = augment(bursts, tier, seed=seed_counter)
            tensors = to_tensor(augmented)

            key = f"{split_name}__{tier}"
            output[f"{key}__X"] = tensors
            output[f"{key}__y"] = labels.astype(np.int64)

            manifest.append({
                "split": split_name,
                "tier": tier,
                "n": int(len(labels)),
                "shape": list(tensors.shape[1:]),
                "seed": seed_counter,
            })
            print(f"{split_name:<12}{tier:<12}{len(labels):>9,}"
                  f"{str(tuple(tensors.shape)):>22}")

    # Record everything needed to reproduce or interpret this dataset.
    config = {
        "front_end": front_end,
        "stft_window": stft_window,
        "stft_hop": stft_hop,
        "n_classes": n_classes,
        "cohort": list(meta["cohort"]),
        "held_out_rx": str(meta["held_out_rx"]),
        "held_out_day": str(meta["held_out_day"]),
        "train_tiers": TRAIN_TIERS,
        "test_tiers": TEST_TIERS,
        "held_out_tier": "dynamic",
        "base_seed": seed,
        "manifest": manifest,
    }
    output["_config"] = np.array([json.dumps(config)], dtype=object)

    np.savez_compressed(out_path, **output)
    print(f"\nsaved -> {out_path}")

    total = sum(m["n"] for m in manifest)
    print(f"total tensors written: {total:,}")


def load_prepared(path: str):
    """
    Load a prepared dataset.

    Returns
    -------
    data : dict keyed "split__tier" -> {"X": array, "y": array}
    config : dict
    """
    raw = np.load(path, allow_pickle=True)
    config = json.loads(str(raw["_config"][0]))

    data = {}
    for key in raw.files:
        if key == "_config":
            continue
        split_tier, field = key.rsplit("__", 1)
        data.setdefault(split_tier, {})[field] = raw[key]
    return data, config


# =============================================================================
# COMMAND LINE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the augmented, transformed dataset for training.")
    parser.add_argument("--subset", default="cohort10.npz",
                        help="output of wisig_loader.build_subset()")
    parser.add_argument("--out", default="prepared_spec.npz",
                        help="where to write the prepared dataset")
    parser.add_argument("--front-end", default="spectrogram",
                        choices=["spectrogram", "raw_iq"])
    parser.add_argument("--stft-window", type=int, default=64)
    parser.add_argument("--stft-hop", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepare(subset_path=args.subset,
            out_path=args.out,
            front_end=args.front_end,
            stft_window=args.stft_window,
            stft_hop=args.stft_hop,
            seed=args.seed)

    print("""
-------------------------------------------------------------------------------
NEXT STEPS
-------------------------------------------------------------------------------
If you have not built the cohort subset yet, run this first:

    from wisig_loader import SplitSpec, build_subset
    spec = SplitSpec(held_out_rx="8-8", held_out_day="2021_03_23", seed=0)
    build_subset(r"..\\Capstone data\\ManyTx.pkl", "cohort10.npz", k=10, spec=spec)

Then prepare the tensors:

    python prepare_dataset.py --subset cohort10.npz --out prepared_spec.npz

Then train:

    python train.py --data prepared_spec.npz
-------------------------------------------------------------------------------
""")
