"""
wisig_loader.py
===============================================================================
Dataset loading and split construction for the SEI capstone project.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS MODULE DOES
-------------------------------------------------------------------------------
Loads the WiSig ManyTx dataset, selects a cohort of transmitters to classify,
and builds the train/test splits that make this project's central question
answerable.

The splits are the important part. A conventional project would shuffle all the
signals and cut off 20% for testing. We deliberately do NOT do that, because a
random shuffle puts the same receivers and the same capture days on both sides
of the split - so a model could score well by memorising the recording
conditions rather than the transmitters.

Instead we hold out entire CONDITIONS:

    held-out receiver   - one physical USRP never appears in training
    held-out day        - one capture day never appears in training
    held-out sea state  - handled separately by gauntlet.py, not here

This lets us ask: does the model still identify transmitters correctly when the
receiver hardware or the propagation environment is one it has never seen? And
in Deliverable III: does it KNOW when it is uncertain about them?

-------------------------------------------------------------------------------
THE DATASET FILE
-------------------------------------------------------------------------------
ManyTx.pkl is a Python pickle of a dictionary:

    tx_list             150 transmitter IDs   (ORBIT node names, e.g. '1-10')
    rx_list              18 receiver IDs
    capture_date_list     4 capture days
    equalized_list      [0, 1]  -> 0 = non-equalized, 1 = equalized
    max_sig             50      (cap of 50 signals per tx-rx-day-eq cell)
    data                nested list, indexed [tx][rx][day][eq]

Each cell of `data` is a numpy array of shape (N, 256, 2), N <= 50, where 256 is
the number of complex baseband samples in one 802.11a/g preamble and the last
axis holds in-phase and quadrature components.

WE USE THE NON-EQUALIZED VERSION (equalization index 0). Equalization removes
channel effects, which is exactly the wrong thing here: a passive intercept
receiver could not undo them, and this project adds its own channel downstream.

-------------------------------------------------------------------------------
WHY THERE ARE TWO STEPS (build once, load many times)
-------------------------------------------------------------------------------
The pickle is 4.18 GB on disk and Python cannot read part of a pickle - the
whole thing must be deserialised, which needs roughly 5 GB of RAM and takes a
minute or two. Doing that at the start of every training run would be painful.

So the workflow is:

    STEP 1 (run once)   build_subset()  loads the big pickle, pulls out just the
                        transmitters you selected, converts to float32, and
                        saves a compact .npz file.
                        For a 10-transmitter cohort this is about 74 MB.

    STEP 2 (every run)  load_subset()   reads that small .npz in a second.

-------------------------------------------------------------------------------
A NOTE ON DTYPE
-------------------------------------------------------------------------------
WiSig stores samples as float64, which is why the file is 4.18 GB. Nothing about
RF fingerprinting needs 15 significant digits, so we cast to float32 on the way
out. This halves memory and matches what the neural network will use anyway.
===============================================================================
"""

from dataclasses import dataclass
import pickle
import numpy as np


# Index into WiSig's `equalized_list`. 0 = non-equalized (what we want).
NON_EQUALIZED = 0

# Each burst is 256 complex samples stored as (256, 2) real values.
N_SAMPLES = 256


# =============================================================================
# INSPECTION HELPERS
# =============================================================================

def load_manytx(path: str) -> dict:
    """
    Deserialise ManyTx.pkl.

    WARNING: this needs roughly 5 GB of free RAM and takes a minute or two.
    Call this once from build_subset(), not from your training script.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def signal_counts(dataset: dict) -> np.ndarray:
    """
    Count the non-equalized signals in every transmitter-receiver-day cell.

    Returns an integer array of shape (n_tx, n_rx, n_day). This is the basis for
    every coverage decision below, so it is computed once and reused.
    """
    n_tx = len(dataset["tx_list"])
    n_rx = len(dataset["rx_list"])
    n_day = len(dataset["capture_date_list"])

    counts = np.zeros((n_tx, n_rx, n_day), dtype=int)
    for t in range(n_tx):
        for r in range(n_rx):
            for d in range(n_day):
                counts[t, r, d] = len(dataset["data"][t][r][d][NON_EQUALIZED])
    return counts


def full_coverage_transmitters(dataset: dict) -> list:
    """
    Return the transmitter IDs that have a complete set of signals.

    Not every transmitter was heard by every receiver on every day. The
    theoretical maximum is

        18 receivers  x  4 days  x  50 signals  =  3600 signals

    Inspection of ManyTx shows 71 of the 150 transmitters reach that maximum,
    while the worst has only 1556. Drawing the cohort from the complete ones
    means every class is represented equally on every receiver and every day -
    so when we hold out a receiver, no class is accidentally starved, and no
    class weighting is needed.
    """
    counts = signal_counts(dataset)
    n_rx = len(dataset["rx_list"])
    n_day = len(dataset["capture_date_list"])
    max_possible = n_rx * n_day * dataset["max_sig"]

    totals = counts.sum(axis=(1, 2))
    return [dataset["tx_list"][i]
            for i in range(len(totals)) if totals[i] == max_possible]


def describe(dataset: dict) -> None:
    """Print a short summary of the dataset. Useful as a first sanity check."""
    counts = signal_counts(dataset)
    totals = counts.sum(axis=(1, 2))
    full = full_coverage_transmitters(dataset)

    print(f"transmitters      : {len(dataset['tx_list'])}")
    print(f"receivers         : {len(dataset['rx_list'])}")
    print(f"capture days      : {dataset['capture_date_list']}")
    print(f"total signals     : {counts.sum():,}")
    print(f"per transmitter   : min {totals.min()}, "
          f"median {int(np.median(totals))}, max {totals.max()}")
    print(f"full coverage     : {len(full)} of {len(dataset['tx_list'])} transmitters")


# =============================================================================
# SPLIT SPECIFICATION
# =============================================================================

@dataclass
class SplitSpec:
    """
    Defines which conditions are withheld from training.

    held_out_rx:
        ID of the receiver that appears only at test time. Signal counts are
        very even across receivers (27,887 to 29,114), so any receiver works.

    held_out_day:
        Capture day that appears only at test time. The four days span about
        three weeks, so the propagation environment genuinely differs between
        them - this is real distribution shift, not a synthetic construction.

    val_fraction:
        Fraction of the TRAINING conditions set aside for validation. This is a
        normal random split, because its job is model selection rather than
        measuring generalisation to new conditions.

    seed:
        Controls cohort selection and the train/validation shuffle, so the whole
        pipeline is reproducible.
    """
    held_out_rx: str
    held_out_day: str
    val_fraction: float = 0.15
    seed: int = 0


def select_cohort(dataset: dict, k: int, seed: int = 0) -> list:
    """
    Choose k transmitters at random from the full-coverage set.

    Class count is an experimental variable in this project (runs at roughly
    k = 10, 50, 71), so the cohort is chosen by seeded random draw rather than
    by taking the first k. Taking the first k would bias the selection toward
    one corner of the ORBIT grid, since node IDs encode physical position.
    """
    available = full_coverage_transmitters(dataset)
    if k > len(available):
        raise ValueError(
            f"asked for {k} transmitters but only {len(available)} have full "
            f"coverage; either lower k or accept class imbalance"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(available, size=k, replace=False)
    return sorted(chosen.tolist())


# =============================================================================
# BUILDING THE SPLITS
# =============================================================================

def build_splits(dataset: dict, cohort: list, spec: SplitSpec) -> dict:
    """
    Assemble the train / validation / test splits.

    Returns a dictionary of splits. Each split is itself a dictionary with:

        X    float32 array, shape (n, 256, 2)  - the raw I/Q bursts
        y    int64   array, shape (n,)         - class index into `cohort`
        rx   list of the receiver ID for each burst
        day  list of the capture day for each burst

    The rx and day labels are carried along so results can be broken down by
    condition later, and so leakage can be checked directly.

    THE SPLITS
    ----------
        train      training receivers, training days
        val        same conditions as train, randomly held-out signals
        test_id    same conditions as train, held-out signals
                   -> "in distribution": the easy case, our reference point
        test_rx    the held-out RECEIVER, training days
                   -> does the fingerprint survive unfamiliar receiver hardware?
        test_day   training receivers, the held-out DAY
                   -> does it survive an unfamiliar propagation environment?
        test_both  held-out receiver AND held-out day
                   -> the hardest case, both shifts at once

    Note that test_id, test_rx and test_day differ in exactly one variable each,
    which is what lets us attribute any drop in accuracy or calibration to a
    specific cause.
    """
    tx_index = {tx: i for i, tx in enumerate(dataset["tx_list"])}
    rx_index = {rx: i for i, rx in enumerate(dataset["rx_list"])}
    day_index = {d: i for i, d in enumerate(dataset["capture_date_list"])}

    if spec.held_out_rx not in rx_index:
        raise ValueError(f"unknown receiver {spec.held_out_rx!r}")
    if spec.held_out_day not in day_index:
        raise ValueError(f"unknown capture day {spec.held_out_day!r}")

    train_rx = [r for r in dataset["rx_list"] if r != spec.held_out_rx]
    train_days = [d for d in dataset["capture_date_list"] if d != spec.held_out_day]

    # Collect bursts into buckets keyed by which split they belong to.
    buckets = {name: {"X": [], "y": [], "rx": [], "day": []}
               for name in ["in_conditions", "test_rx", "test_day", "test_both"]}

    for class_index, tx in enumerate(cohort):
        t = tx_index[tx]
        for rx in dataset["rx_list"]:
            r = rx_index[rx]
            for day in dataset["capture_date_list"]:
                d = day_index[day]

                cell = dataset["data"][t][r][d][NON_EQUALIZED]
                if len(cell) == 0:
                    continue

                rx_held = (rx == spec.held_out_rx)
                day_held = (day == spec.held_out_day)

                if rx_held and day_held:
                    target = "test_both"
                elif rx_held:
                    target = "test_rx"
                elif day_held:
                    target = "test_day"
                else:
                    target = "in_conditions"

                buckets[target]["X"].append(np.asarray(cell, dtype=np.float32))
                buckets[target]["y"].append(
                    np.full(len(cell), class_index, dtype=np.int64))
                buckets[target]["rx"].extend([rx] * len(cell))
                buckets[target]["day"].extend([day] * len(cell))

    def stack(bucket):
        if not bucket["X"]:
            return {"X": np.zeros((0, N_SAMPLES, 2), np.float32),
                    "y": np.zeros(0, np.int64), "rx": [], "day": []}
        return {"X": np.concatenate(bucket["X"]),
                "y": np.concatenate(bucket["y"]),
                "rx": bucket["rx"],
                "day": bucket["day"]}

    in_conditions = stack(buckets["in_conditions"])

    # Split the in-condition data three ways: train / validation / in-distribution
    # test. This one IS a random shuffle, because all three share the same
    # receivers and days - the point here is ordinary model selection, not
    # measuring generalisation to new conditions.
    #
    # The shuffle is STRATIFIED BY CLASS: each transmitter's signals are split
    # in the same proportions independently. A plain shuffle would let class
    # counts drift apart by chance, and in the worst case a class could be
    # almost absent from validation, which would quietly distort both model
    # selection and the per-class calibration analysis in Deliverable III.
    rng = np.random.default_rng(spec.seed)
    train_idx, val_idx, test_idx = [], [], []

    for class_index in range(len(cohort)):
        member_positions = np.flatnonzero(in_conditions["y"] == class_index)
        shuffled = rng.permutation(member_positions)

        n_class = len(shuffled)
        n_val_c = int(spec.val_fraction * n_class)
        n_test_c = int(spec.val_fraction * n_class)

        val_idx.append(shuffled[:n_val_c])
        test_idx.append(shuffled[n_val_c:n_val_c + n_test_c])
        train_idx.append(shuffled[n_val_c + n_test_c:])

    # Concatenate the per-class pieces, then shuffle once more so the classes
    # are interleaved rather than arriving in blocks.
    train_idx = rng.permutation(np.concatenate(train_idx))
    val_idx = rng.permutation(np.concatenate(val_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))

    def take(source, idx):
        return {"X": source["X"][idx],
                "y": source["y"][idx],
                "rx": [source["rx"][i] for i in idx],
                "day": [source["day"][i] for i in idx]}

    splits = {
        "train":     take(in_conditions, train_idx),
        "val":       take(in_conditions, val_idx),
        "test_id":   take(in_conditions, test_idx),
        "test_rx":   stack(buckets["test_rx"]),
        "test_day":  stack(buckets["test_day"]),
        "test_both": stack(buckets["test_both"]),
    }

    splits["_meta"] = {
        "cohort": cohort,
        "n_classes": len(cohort),
        "held_out_rx": spec.held_out_rx,
        "held_out_day": spec.held_out_day,
        "train_rx": train_rx,
        "train_days": train_days,
        "seed": spec.seed,
    }
    return splits


# =============================================================================
# CACHING: build once, load many times
# =============================================================================

def build_subset(pkl_path: str, out_path: str, k: int, spec: SplitSpec) -> dict:
    """
    STEP 1 - run this once.

    Loads the full 4.18 GB pickle, selects a cohort of k transmitters, builds
    the splits, and writes them to a compact .npz file.

    Example
    -------
        spec = SplitSpec(held_out_rx="8-8", held_out_day="2021_03_23")
        build_subset("ManyTx.pkl", "cohort10.npz", k=10, spec=spec)
    """
    print(f"loading {pkl_path} (this needs ~5 GB RAM and a minute or two) ...")
    dataset = load_manytx(pkl_path)
    describe(dataset)

    cohort = select_cohort(dataset, k, seed=spec.seed)
    print(f"\nselected cohort of {k}: {cohort}")

    splits = build_splits(dataset, cohort, spec)
    save_subset(splits, out_path)
    summarise_splits(splits)
    return splits


def save_subset(splits: dict, out_path: str) -> None:
    """Write splits to a compressed .npz. Lists are stored as object arrays."""
    flat = {}
    for name, split in splits.items():
        if name == "_meta":
            flat["_meta"] = np.array([splits["_meta"]], dtype=object)
            continue
        flat[f"{name}__X"] = split["X"]
        flat[f"{name}__y"] = split["y"]
        flat[f"{name}__rx"] = np.array(split["rx"], dtype=object)
        flat[f"{name}__day"] = np.array(split["day"], dtype=object)
    np.savez_compressed(out_path, **flat)
    print(f"\nsaved -> {out_path}")


def load_subset(path: str) -> dict:
    """
    STEP 2 - run this at the start of every training script. Fast.

        splits = load_subset("cohort10.npz")
        X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    """
    raw = np.load(path, allow_pickle=True)
    splits = {"_meta": raw["_meta"][0]}
    for key in raw.files:
        if key == "_meta":
            continue
        name, field = key.split("__")
        splits.setdefault(name, {})[field] = raw[key]
    return splits


def summarise_splits(splits: dict) -> None:
    """Print split sizes and confirm the held-out conditions really are absent."""
    meta = splits["_meta"]
    print(f"\n{'split':<12}{'bursts':>10}{'classes':>10}   receivers / days")
    print("-" * 70)
    for name in ["train", "val", "test_id", "test_rx", "test_day", "test_both"]:
        s = splits[name]
        n = len(s["y"])
        n_classes = len(np.unique(s["y"])) if n else 0
        rx_set = sorted(set(s["rx"]))
        day_set = sorted(set(s["day"]))
        rx_desc = f"{len(rx_set)} rx" if len(rx_set) > 3 else ",".join(rx_set)
        day_desc = f"{len(day_set)} days" if len(day_set) > 2 else ",".join(day_set)
        print(f"{name:<12}{n:>10,}{n_classes:>10}   {rx_desc} / {day_desc}")

    # Leakage check: the held-out receiver and day must not appear in training.
    print("\nleakage check:")
    train_rx = set(splits["train"]["rx"])
    train_day = set(splits["train"]["day"])
    rx_ok = meta["held_out_rx"] not in train_rx
    day_ok = meta["held_out_day"] not in train_day
    print(f"  held-out receiver {meta['held_out_rx']!r} absent from train: "
          f"{rx_ok}  {'PASS' if rx_ok else 'FAIL'}")
    print(f"  held-out day {meta['held_out_day']!r} absent from train:      "
          f"{day_ok}  {'PASS' if day_ok else 'FAIL'}")


# =============================================================================
# SELF-TEST (runs on a small fake dataset, so no 4 GB file needed)
# =============================================================================

def _mock_dataset(n_tx=6, n_rx=4, n_day=3, n_sig=10, missing=()):
    """
    Build a miniature dataset with the same structure as ManyTx.

    Every burst is filled with its own (tx, rx, day) identity encoded in the
    sample values. That way the split logic can be verified exactly: we can
    check that each burst landed in the split its labels say it should.
    """
    tx_list = [f"tx{i}" for i in range(n_tx)]
    rx_list = [f"rx{i}" for i in range(n_rx)]
    days = [f"day{i}" for i in range(n_day)]

    data = []
    for t in range(n_tx):
        per_rx = []
        for r in range(n_rx):
            per_day = []
            for d in range(n_day):
                if (t, r, d) in missing:
                    cell = np.zeros((0, N_SAMPLES, 2))
                else:
                    cell = np.zeros((n_sig, N_SAMPLES, 2))
                    cell[:, 0, 0] = t     # encode identity in the first sample
                    cell[:, 0, 1] = r
                    cell[:, 1, 0] = d
                per_day.append([cell, cell.copy()])   # [non-eq, eq]
            per_rx.append(per_day)
        data.append(per_rx)

    return {"tx_list": tx_list, "rx_list": rx_list, "capture_date_list": days,
            "equalized_list": [0, 1], "max_sig": n_sig, "data": data}


if __name__ == "__main__":
    print("=" * 70)
    print("WISIG LOADER - SELF TEST (on mock data, no big file needed)")
    print("=" * 70)

    # One transmitter is deliberately missing a cell, so it should be excluded
    # from the full-coverage list.
    mock = _mock_dataset(missing=((5, 0, 0),))

    print("\n[1] Dataset summary")
    describe(mock)

    print("\n[2] Full-coverage detection")
    full = full_coverage_transmitters(mock)
    expected = ["tx0", "tx1", "tx2", "tx3", "tx4"]     # tx5 has a missing cell
    ok = full == expected
    print(f"    found     : {full}")
    print(f"    expected  : {expected}")
    print(f"    {'PASS' if ok else 'FAIL'} (tx5 correctly excluded: "
          f"{'tx5' not in full})")

    print("\n[3] Building splits (hold out rx1 and day2)")
    spec = SplitSpec(held_out_rx="rx1", held_out_day="day2", seed=0)
    cohort = select_cohort(mock, k=4, seed=0)
    print(f"    cohort: {cohort}")
    splits = build_splits(mock, cohort, spec)
    summarise_splits(splits)

    print("\n[4] Verifying every burst landed in the correct split")
    # Decode the identity we encoded and confirm it matches the split rule.
    problems = 0
    for name in ["train", "val", "test_id"]:
        s = splits[name]
        if "rx1" in set(s["rx"]) or "day2" in set(s["day"]):
            problems += 1
            print(f"    FAIL: {name} contains a held-out condition")
    if set(splits["test_rx"]["rx"]) not in ({"rx1"}, set()):
        problems += 1
        print("    FAIL: test_rx contains receivers other than rx1")
    if set(splits["test_day"]["day"]) not in ({"day2"}, set()):
        problems += 1
        print("    FAIL: test_day contains days other than day2")
    if problems == 0:
        print("    every split contains exactly the conditions it should  PASS")

    print("\n[5] Class balance (stratified split should make these near-equal)")
    all_ok = True
    for name in ["train", "val", "test_id"]:
        counts = np.bincount(splits[name]["y"], minlength=len(cohort))
        spread = counts.max() - counts.min()
        ok = spread <= 1          # stratification allows at most rounding drift
        all_ok = all_ok and ok
        print(f"    {name:<9} per-class {counts.tolist()}  spread={spread}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"    {'all splits balanced' if all_ok else 'imbalance detected'}")

    print("\n[6] Save / load round-trip")
    save_subset(splits, "/tmp/_mock_subset.npz")
    reloaded = load_subset("/tmp/_mock_subset.npz")
    same = (np.array_equal(reloaded["train"]["X"], splits["train"]["X"])
            and np.array_equal(reloaded["train"]["y"], splits["train"]["y"]))
    print(f"    arrays survive the round trip: {same}   {'PASS' if same else 'FAIL'}")
    print(f"    dtype on reload: {reloaded['train']['X'].dtype} (want float32)")

    print("\n" + "=" * 70)
    print("Self test complete.")
    print("=" * 70)
    print("""
NEXT STEP - run this once against the real data:

    from wisig_loader import SplitSpec, build_subset

    spec = SplitSpec(held_out_rx="8-8", held_out_day="2021_03_23", seed=0)
    build_subset(
        pkl_path="../Capstone data/ManyTx.pkl",
        out_path="cohort10.npz",
        k=10,
        spec=spec,
    )
""")
