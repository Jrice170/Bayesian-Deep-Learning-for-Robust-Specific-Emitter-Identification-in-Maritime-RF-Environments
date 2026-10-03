"""
train.py
===============================================================================
MLE and MAP point-estimation training for the SEI capstone project.
This script covers Intermediate Deliverable II.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
-------------------------------------------------------------------------------
Trains the same convolutional network twice, differing only in the objective:

    MLE   maximum likelihood estimation
          loss = cross_entropy(logits, y)

    MAP   maximum a posteriori estimation
          loss = cross_entropy(logits, y) + l2_coeff * model.L2reg()

and then evaluates both across every combination of test split and sea-state
tier, recording accuracy, loss, per-class confusion, and the predicted
probabilities needed for the calibration analysis.

-------------------------------------------------------------------------------
THE MATHEMATICS BEING IMPLEMENTED
-------------------------------------------------------------------------------
The model outputs K logits, z_k. A softmax turns them into a categorical
distribution over the K transmitters:

    p(y = k | x, theta) = exp(z_k) / sum_j exp(z_j)

MLE looks for the weights that make the observed labels most probable:

    theta_MLE = argmax_theta  prod_i p(y_i | x_i, theta)
              = argmin_theta  -sum_i log p(y_i | x_i, theta)

That second line is exactly the categorical cross-entropy loss. Maximising a
product of probabilities and minimising a sum of negative log-probabilities are
the same operation, which is why the "argmax" of the theory appears as an
"argmin" in the code.

MAP adds a prior over the weights. With a zero-mean Gaussian prior,
p(theta) = N(0, sigma^2 I), the negative log prior is proportional to the sum of
squared weights, so:

    theta_MAP = argmin_theta [ -sum_i log p(y_i | x_i, theta) + lambda * ||theta||^2 ]

The only difference between the two runs is that single extra term. Everything
else - architecture, initialisation, data, batch order, optimiser, epochs - is
held identical, so any difference in the results is attributable to the prior
and nothing else.

-------------------------------------------------------------------------------
WHY NO EARLY STOPPING
-------------------------------------------------------------------------------
It would be conventional to halt training when validation loss stops improving.
We deliberately do not, because the whole point of comparing MLE with MAP is to
observe the prior CONTROLLING OVERFITTING. If both runs were halted the moment
overfitting began, the effect being studied would be cut off before it could be
measured.

Instead both runs use a fixed epoch budget, the full loss curves are recorded
for the paper, and the best-validation checkpoint is kept separately for
reporting test numbers. The gap between the training and validation curves is
itself a result: it should be visibly wider for MLE than for MAP.

-------------------------------------------------------------------------------
WHY CALIBRATION IS MEASURED HERE, IN DELIVERABLE II
-------------------------------------------------------------------------------
Expected calibration error and the reliability diagram belong to Deliverable
III's Bayesian analysis, but they are computed here too, for the point-estimate
models. That establishes the BASELINE: how badly a conventional classifier
misjudges its own confidence, especially on the held-out conditions. Without
that baseline there is nothing for the Bayesian methods to be compared against.

Predicted probabilities for every test example are saved to disk so Deliverable
III can produce its figures without retraining anything.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
    # single run
    python train.py --data prepared_spec.npz

    # five seeds, so differences can be separated from run-to-run noise
    python train.py --data prepared_spec.npz --seeds 0,1,2,3,4 --tag spec5

    # magnitude-only ablation, to test whether the phase channels help
    python train.py --data prepared_spec.npz --seeds 0,1,2,3,4 --channels 0 --tag mag5

    # quick smoke test before committing to a full run
    python train.py --data prepared_spec.npz --epochs 3 --tag smoke

-------------------------------------------------------------------------------
ON REPORTING SEEDS
-------------------------------------------------------------------------------
A single training run produces one number per evaluation cell, and that number
carries two separate uncertainties: the finite size of the test split, and the
randomness of training itself. Running several seeds exposes the second.

This matters here more than in most projects. The whole subject of this work is
whether a model's stated confidence can be trusted, so reporting bare point
estimates with no indication of their variability would undercut the argument
being made. Where a difference between two methods is smaller than the
seed-to-seed variation, it is reported as inconclusive rather than as a result.
===============================================================================
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from prepare_dataset import load_prepared
from models import build_model, count_parameters, set_seed


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def expected_calibration_error(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE).

    Calibration asks whether a model's confidence is honest: of all the times it
    said "90% sure", was it right about 90% of the time?

    The measurement sorts predictions into bins by confidence, and in each bin
    compares the average confidence against the actual accuracy. ECE is the
    average of those gaps, weighted by how many predictions fall in each bin:

        ECE = sum_b  (n_b / N) * | accuracy(b) - confidence(b) |

    A perfectly calibrated model scores 0. A confidently wrong model scores
    high. This is the number the Bayesian methods in Deliverable III are meant
    to improve, and the reliability diagram is its visual form.
    """
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Upper edge included in the final bin so confidence == 1.0 is counted.
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else \
                 (confidence >= lo) & (confidence <= hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            bin_stats.append({"lo": lo, "hi": hi, "n": 0,
                              "accuracy": None, "confidence": None})
            continue

        acc = float(correct[in_bin].mean())
        conf = float(confidence[in_bin].mean())
        ece += (n_in / len(labels)) * abs(acc - conf)
        bin_stats.append({"lo": float(lo), "hi": float(hi), "n": n_in,
                          "accuracy": acc, "confidence": conf})

    return float(ece), bin_stats


def confusion_matrix(predictions, labels, n_classes):
    """Plain confusion matrix. Rows are true classes, columns predicted."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, predictions):
        cm[t, p] += 1
    return cm


# =============================================================================
# DATA HANDLING
# =============================================================================

def select_channels(X, channels):
    """
    Keep only the requested input channels.

    Used for the ablation that tests whether the cos/sin phase channels
    contribute anything. `--channels 0` keeps log magnitude alone;
    the default keeps all three.
    """
    if channels is None:
        return X
    return X[:, channels, ...]


def make_loader(X, y, batch_size, shuffle, device):
    """Wrap numpy arrays in a DataLoader."""
    tensors = TensorDataset(torch.from_numpy(X).float(),
                            torch.from_numpy(y).long())
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def combine_training_tiers(data, split, tiers, channels):
    """
    Stack the training tiers into one training set.

    The model trains on CONTROLLED and DEGRADED together, so it sees a range of
    channel conditions but never the DYNAMIC tier that is held out for testing.
    """
    Xs, ys = [], []
    for tier in tiers:
        key = f"{split}__{tier}"
        if key in data:
            Xs.append(select_channels(data[key]["X"], channels))
            ys.append(data[key]["y"])
    return np.concatenate(Xs), np.concatenate(ys)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(objective, train_loader, val_loader, n_classes, in_shape,
                args, device, seed_label=""):
    """
    Train one model under one objective.

    Parameters
    ----------
    objective : "mle" or "map"
    seed_label : str, printed in the header when running multiple seeds

    Returns
    -------
    model    the model with the best validation loss seen during training
    history  per-epoch losses and accuracies, for the paper's curves
    """
    # The MLE and MAP runs for a given seed start from IDENTICAL initial
    # weights, so the comparison isolates the effect of the prior rather than
    # the luck of the draw. Across seeds the initialisation varies, which is
    # what lets us measure run-to-run variation.
    set_seed(args.seed)

    model = build_model("spectrogram", n_classes,
                        dropout_p=args.dropout,
                        in_channels=in_shape[0]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    l2_coeff = args.l2_coeff if objective == "map" else 0.0

    print(f"\n{'=' * 70}")
    print(f"TRAINING: {objective.upper()}"
          f"{'  (l2_coeff = %g)' % l2_coeff if l2_coeff else '  (no prior)'}"
          f"{'   ' + seed_label if seed_label else ''}")
    print(f"{'=' * 70}")
    print(f"parameters : {count_parameters(model):,}")
    print(f"device     : {device}")
    print(f"epochs     : {args.epochs}   batch {args.batch_size}   lr {args.lr}")
    print()

    history = {"train_loss": [], "train_nll": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    best_val = float("inf")
    best_state = None
    start = time.time()

    for epoch in range(args.epochs):
        # ---- training pass -------------------------------------------------
        model.train()
        running_loss = running_nll = running_correct = running_n = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            # Negative log-likelihood of the categorical model. This IS the
            # multiclass cross-entropy loss.
            nll = F.cross_entropy(logits, yb)

            # MAP adds the Gaussian prior term. Written explicitly in the loss
            # rather than passed to the optimiser as weight_decay - see the
            # note in models.py for why that distinction matters with Adam.
            loss = nll + l2_coeff * model.L2reg() if l2_coeff else nll

            loss.backward()
            optimizer.step()

            batch_n = len(yb)
            running_loss += loss.item() * batch_n
            running_nll += nll.item() * batch_n
            running_correct += (logits.argmax(1) == yb).sum().item()
            running_n += batch_n

        train_loss = running_loss / running_n
        train_nll = running_nll / running_n
        train_acc = running_correct / running_n

        # ---- validation pass -----------------------------------------------
        # Dropout is disabled here (model.eval()), so validation measures the
        # deterministic network. Deliverable III will deliberately re-enable it.
        model.eval()
        val_loss = val_correct = val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += F.cross_entropy(logits, yb, reduction="sum").item()
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_n += len(yb)

        val_loss /= val_n
        val_acc = val_correct / val_n

        history["train_loss"].append(train_loss)
        history["train_nll"].append(train_nll)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Keep the best-validation weights for reporting, but keep training so
        # the overfitting behaviour is fully visible in the curves.
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:>3}/{args.epochs}   "
                  f"train nll {train_nll:.4f}  acc {train_acc:.4f}   |   "
                  f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
                  f"{'  *' if val_loss == best_val else ''}")

    elapsed = time.time() - start
    print(f"\n  finished in {elapsed/60:.1f} min   best val loss {best_val:.4f}")

    # A useful diagnostic for the paper: how far did validation loss drift above
    # its best value? A large gap is the signature of overfitting.
    final_gap = history["val_loss"][-1] - best_val
    print(f"  val loss drift from best to final epoch: {final_gap:+.4f}"
          f"   ({'overfitting visible' if final_gap > 0.05 else 'little overfitting'})")

    model.load_state_dict(best_state)
    return model, history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, data, config, channels, n_classes, device, batch_size):
    """
    Evaluate across every test split and sea-state tier.

    Returns a dictionary keyed "split__tier" holding accuracy, loss, ECE, the
    confusion matrix, and the raw predicted probabilities. The probabilities are
    kept so Deliverable III can build reliability diagrams without retraining.
    """
    model.eval()
    results = {}
    probabilities = {}

    test_keys = [k for k in data.keys() if k.startswith("test_")]

    for key in sorted(test_keys):
        X = select_channels(data[key]["X"], channels)
        y = data[key]["y"]

        loader = make_loader(X, y, batch_size, shuffle=False, device=device)

        all_probs = []
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                total_loss += F.cross_entropy(logits, yb,
                                              reduction="sum").item()
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())

        probs = np.concatenate(all_probs)
        preds = probs.argmax(axis=1)

        accuracy = float((preds == y).mean())
        loss = total_loss / len(y)
        ece, bins = expected_calibration_error(probs, y)
        cm = confusion_matrix(preds, y, n_classes)

        results[key] = {
            "accuracy": accuracy,
            "loss": loss,
            "ece": ece,
            "mean_confidence": float(probs.max(axis=1).mean()),
            "n": int(len(y)),
            "confusion_matrix": cm.tolist(),
            "reliability_bins": bins,
        }
        probabilities[key] = probs

    return results, probabilities


def print_results_table(results, config):
    """Print evaluation results grouped so the shift axes are easy to compare."""
    print(f"\n{'split':<11}{'tier':<12}{'n':>7}{'accuracy':>10}"
          f"{'mean conf':>11}{'ECE':>8}")
    print("-" * 70)

    held_tier = config["held_out_tier"]
    for split in ["test_id", "test_rx", "test_day", "test_both"]:
        for tier in config["test_tiers"]:
            key = f"{split}__{tier}"
            if key not in results:
                continue
            r = results[key]
            marker = " <- held out" if tier == held_tier else ""
            print(f"{split:<11}{tier:<12}{r['n']:>7,}{r['accuracy']:>10.4f}"
                  f"{r['mean_confidence']:>11.4f}{r['ece']:>8.4f}{marker}")
        print()


# =============================================================================
# AGGREGATION ACROSS SEEDS
# =============================================================================

def aggregate_seeds(per_seed_results):
    """
    Collapse a list of per-seed result dictionaries into mean and standard
    deviation for every metric in every evaluation cell.

    WHY THIS MATTERS. A single training run gives one number per cell, and that
    number carries two kinds of uncertainty: the finite size of the test split,
    and the randomness of training itself (weight initialisation and batch
    order). Repeating the run with different seeds exposes the second kind.

    Without it there is no way to tell a real effect from a lucky initialisation
    - which would be a particularly awkward omission in a project about
    trustworthy uncertainty.
    """
    aggregated = {}
    keys = per_seed_results[0].keys()

    for key in keys:
        entry = {"n": per_seed_results[0][key]["n"]}
        for metric in ["accuracy", "loss", "ece", "mean_confidence"]:
            values = np.array([r[key][metric] for r in per_seed_results])
            entry[metric] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            entry[f"{metric}_values"] = values.tolist()
        # Confusion matrices are summed rather than averaged, so the total still
        # reads as a count of classifications.
        cms = np.array([r[key]["confusion_matrix"] for r in per_seed_results])
        entry["confusion_matrix"] = cms.sum(axis=0).tolist()
        aggregated[key] = entry

    return aggregated


def print_aggregated_table(agg, config, n_seeds):
    """Print mean +/- standard deviation across seeds."""
    print(f"\n{'split':<11}{'tier':<12}{'n':>7}"
          f"{'accuracy':>18}{'ECE':>18}")
    print("-" * 70)

    held_tier = config["held_out_tier"]
    for split in ["test_id", "test_rx", "test_day", "test_both"]:
        for tier in config["test_tiers"]:
            key = f"{split}__{tier}"
            if key not in agg:
                continue
            r = agg[key]
            marker = " <-" if tier == held_tier else ""
            acc = f"{r['accuracy']:.4f} +/- {r['accuracy_std']:.4f}"
            ece = f"{r['ece']:.4f} +/- {r['ece_std']:.4f}"
            print(f"{split:<11}{tier:<12}{r['n']:>7,}{acc:>18}{ece:>18}{marker}")
        print()
    print(f"  (mean and sample standard deviation over {n_seeds} seeds)")


def compare_with_significance(agg_a, agg_b, label_a, label_b, config):
    """
    Compare two aggregated result sets and flag which differences are larger
    than the run-to-run variation.

    The test used is deliberately simple and conservative: a difference is
    called MEANINGFUL only if it exceeds twice the pooled standard deviation
    across seeds. That is roughly a two-sigma criterion. It is not a formal
    hypothesis test, and it is described as such in the paper - with a handful
    of seeds a t-test would imply more precision than the evidence supports.
    """
    print(f"\n{'=' * 78}")
    print(f"{label_a.upper()} vs {label_b.upper()}   (difference vs seed variation)")
    print(f"{'=' * 78}")
    print(f"{'split__tier':<24}{'d accuracy':>22}{'d ECE':>22}")
    print("-" * 78)

    def fmt(mean_a, std_a, mean_b, std_b):
        delta = mean_b - mean_a
        pooled = np.sqrt(std_a ** 2 + std_b ** 2)
        if pooled < 1e-9:
            flag = " "
        elif abs(delta) > 2 * pooled:
            flag = "*"          # exceeds twice the seed-to-seed variation
        else:
            flag = " "
        return f"{delta:+.4f} (+/-{pooled:.4f}){flag}"

    for key in sorted(agg_a.keys()):
        a, b = agg_a[key], agg_b[key]
        acc = fmt(a["accuracy"], a["accuracy_std"], b["accuracy"], b["accuracy_std"])
        ece = fmt(a["ece"], a["ece_std"], b["ece"], b["ece_std"])
        print(f"{key:<24}{acc:>22}{ece:>22}")

    print("\n  * marks a difference larger than twice the combined seed variation.")
    print("  Unmarked rows are within run-to-run noise and should not be claimed")
    print("  as effects in the paper.")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")

    data, config = load_prepared(args.data)
    n_classes = config["n_classes"]

    channels = None
    if args.channels is not None:
        channels = [int(c) for c in args.channels.split(",")]

    # Assemble training and validation sets from the permitted tiers.
    X_train, y_train = combine_training_tiers(
        data, "train", config["train_tiers"], channels)
    X_val, y_val = combine_training_tiers(
        data, "val", config["train_tiers"], channels)

    in_shape = X_train.shape[1:]

    print("=" * 70)
    print("DELIVERABLE II - MLE AND MAP POINT ESTIMATION")
    print("=" * 70)
    print(f"data          : {args.data}")
    print(f"classes       : {n_classes}")
    print(f"input shape   : {in_shape}"
          f"{'  (channels ' + args.channels + ')' if channels else ''}")
    print(f"train         : {len(y_train):,} examples "
          f"({config['train_tiers']})")
    print(f"val           : {len(y_val):,} examples")
    print(f"held-out rx   : {config['held_out_rx']}")
    print(f"held-out day  : {config['held_out_day']}")
    print(f"held-out tier : {config['held_out_tier']}")

    train_loader = make_loader(X_train, y_train, args.batch_size, True, device)
    val_loader = make_loader(X_val, y_val, args.batch_size, False, device)

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"seeds         : {seeds}")

    per_seed = {"mle": [], "map": []}
    all_histories = {"mle": [], "map": []}
    all_probs = {}          # probabilities from the FIRST seed only

    for objective in ["mle", "map"]:
        for run_index, seed in enumerate(seeds):
            args.seed = seed
            model, history = train_model(objective, train_loader, val_loader,
                                         n_classes, in_shape, args, device,
                                         seed_label=f"seed {seed}")

            results, probs = evaluate(model, data, config, channels, n_classes,
                                      device, args.batch_size)

            per_seed[objective].append(results)
            all_histories[objective].append(history)

            if run_index == 0:
                # Save probabilities and the checkpoint from the first seed, so
                # Deliverable III has a concrete model to work from. The other
                # seeds exist to quantify run-to-run variation.
                all_probs[objective] = probs
                torch.save(model.state_dict(),
                           os.path.join("results", f"model_{objective}_{args.tag}.pt"))

            if len(seeds) == 1:
                print(f"\n{objective.upper()} RESULTS")
                print_results_table(results, config)

    # ---- aggregate ---------------------------------------------------------
    agg = {obj: aggregate_seeds(per_seed[obj]) for obj in ["mle", "map"]}

    if len(seeds) > 1:
        for objective in ["mle", "map"]:
            print(f"\n{'=' * 70}")
            print(f"{objective.upper()} RESULTS  (averaged over {len(seeds)} seeds)")
            print(f"{'=' * 70}")
            print_aggregated_table(agg[objective], config, len(seeds))

    compare_with_significance(agg["mle"], agg["map"], "mle", "map", config)

    # ---- save everything ---------------------------------------------------
    summary = {
        "config": config,
        "args": vars(args),
        "seeds": seeds,
        "aggregated": agg,
        "per_seed": per_seed,
        "histories": all_histories,
    }
    with open(os.path.join("results", f"results_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        os.path.join("results", f"probabilities_{args.tag}.npz"),
        **{f"{obj}__{key}": p
           for obj, d in all_probs.items() for key, p in d.items()},
        **{f"labels__{key}": data[key]["y"]
           for key in all_probs["mle"].keys()})

    print(f"\nsaved results_{args.tag}.json   "
          f"(aggregated + per-seed + curves)")
    print(f"saved probabilities_{args.tag}.npz  "
          f"(first seed, for Deliverable III reliability diagrams)")
    print(f"saved model_mle_{args.tag}.pt, model_map_{args.tag}.pt  (first seed)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train MLE and MAP point estimates for Deliverable II.")
    p.add_argument("--data", default="prepared_spec.npz")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2-coeff", type=float, default=1e-4,
                   help="prior strength for MAP; tune on the validation split")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seeds", default="0",
                   help="comma-separated seeds, e.g. '0,1,2,3,4'. Multiple "
                        "seeds give mean +/- std per cell, so real effects can "
                        "be told apart from run-to-run noise.")
    p.add_argument("--seed", type=int, default=0,
                   help=argparse.SUPPRESS)   # set internally from --seeds
    p.add_argument("--channels", default=None,
                   help="comma-separated channel indices, e.g. '0' for "
                        "magnitude only. Default uses all channels.")
    p.add_argument("--tag", default="run1", help="suffix for output files")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    main(p.parse_args())
