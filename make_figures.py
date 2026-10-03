"""
make_figures.py
===============================================================================
Generates every figure and LaTeX table used in the Deliverable II write-up.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS PRODUCES
-------------------------------------------------------------------------------
    fig_loss_curves.pdf       training and validation loss, MLE beside MAP
    fig_accuracy_shift.pdf    accuracy across every split and sea-state tier
    fig_reliability.pdf       reliability diagrams - the central figure
    fig_confidence_gap.pdf    confidence against accuracy, showing the failure
    fig_confusion.pdf         confusion matrices, in and out of distribution
    table_architecture.tex    layer table for the Methods section
    table_results.tex         main results table with seed error bars

Figures are written as PDF because LaTeX embeds vector graphics without any
loss of quality, and the text inside them stays selectable and searchable.

Run after training:

    python make_figures.py --results results_spec5.json \\
                           --probs probabilities_spec5.npz \\
                           --outdir paper
===============================================================================
"""

import argparse
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

SPLITS = ["test_id", "test_rx", "test_day", "test_both"]
TIERS = ["controlled", "degraded", "dynamic"]

SPLIT_LABEL = {
    "test_id": "in-distribution",
    "test_rx": "held-out receiver",
    "test_day": "held-out day",
    "test_both": "both held out",
}


# =============================================================================
# FIGURE 1: LOSS CURVES
# =============================================================================

def fig_loss_curves(summary, outdir):
    """
    Training and validation loss per epoch, for both objectives.

    This is the figure that shows what the prior is doing during optimisation.
    A widening gap between the training curve and the validation curve is the
    signature of overfitting; if the prior is working, that gap should be
    narrower for MAP than for MLE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)

    for ax, objective in zip(axes, ["mle", "map"]):
        histories = summary["histories"][objective]

        # Average the curves over seeds, and shade one standard deviation so the
        # run-to-run variability is visible rather than hidden.
        train = np.array([h["train_nll"] for h in histories])
        val = np.array([h["val_loss"] for h in histories])
        epochs = np.arange(train.shape[1])

        ax.plot(epochs, train.mean(0), label="training", lw=1.6)
        ax.fill_between(epochs, train.mean(0) - train.std(0),
                        train.mean(0) + train.std(0), alpha=0.2)

        ax.plot(epochs, val.mean(0), label="validation", lw=1.6)
        ax.fill_between(epochs, val.mean(0) - val.std(0),
                        val.mean(0) + val.std(0), alpha=0.2)

        gap = val.mean(0)[-1] - train.mean(0)[-1]
        ax.set_title(f"{objective.upper()}   (final gap {gap:+.3f})")
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)

    axes[0].set_ylabel("negative log-likelihood")
    fig.suptitle("Training and validation loss, mean over seeds with 1 s.d. band",
                 y=1.02, fontsize=10)
    path = os.path.join(outdir, "fig_loss_curves.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# =============================================================================
# FIGURE 2: ACCURACY UNDER SHIFT
# =============================================================================

def fig_accuracy_shift(summary, outdir):
    """
    Accuracy for every split and tier, with error bars over seeds.

    Reading left to right shows the cost of each held-out condition; reading
    within a group shows the cost of a worsening sea state.
    """
    agg = summary["aggregated"]
    fig, ax = plt.subplots(figsize=(8, 3.4))

    width = 0.35
    positions, labels = [], []
    pos = 0

    for split in SPLITS:
        for tier in TIERS:
            key = f"{split}__{tier}"
            for i, objective in enumerate(["mle", "map"]):
                r = agg[objective][key]
                ax.bar(pos + i * width, r["accuracy"], width,
                       yerr=r["accuracy_std"], capsize=2,
                       color=f"C{i}", alpha=0.85,
                       label=objective.upper() if pos == 0 and tier == TIERS[0] else "")
            positions.append(pos + width / 2)
            labels.append(tier[:4])
            pos += 1
        pos += 0.6      # gap between split groups

    ax.axhline(0.1, ls="--", c="grey", lw=1)
    ax.text(0.2, 0.115, "chance (10 classes)", fontsize=7, color="grey")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=2)

    # Group labels beneath the tier ticks.
    group_centres = [positions[i * 3 + 1] for i in range(len(SPLITS))]
    for centre, split in zip(group_centres, SPLITS):
        ax.text(centre, -0.13, SPLIT_LABEL[split], ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())

    ax.set_title("Accuracy by held-out condition and sea state "
                 "(error bars: 1 s.d. over 5 seeds)")
    path = os.path.join(outdir, "fig_accuracy_shift.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# =============================================================================
# FIGURE 3: RELIABILITY DIAGRAMS  (the central figure)
# =============================================================================

def reliability_from_probs(probs, labels, n_bins=15):
    """Bin predictions by confidence and return mean confidence vs accuracy."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)

    xs, ys, ws = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum() < 10:          # ignore bins too sparse to estimate
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
        ws.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ws)


def fig_reliability(probs_file, outdir):
    """
    Reliability diagrams: predicted confidence against observed accuracy.

    This is the figure the whole project is built around. A perfectly
    calibrated model traces the diagonal - when it says 80%, it is right 80% of
    the time. A model that falls BELOW the diagonal is overconfident: it claims
    more certainty than its accuracy justifies.

    The panels move from familiar conditions to unfamiliar ones. The point is
    to see the curve peel away from the diagonal as conditions the model never
    trained on are introduced.
    """
    d = np.load(probs_file, allow_pickle=True)

    cells = [("test_id__controlled", "in-distribution"),
             ("test_rx__controlled", "held-out receiver"),
             ("test_id__dynamic", "held-out sea state"),
             ("test_both__dynamic", "all conditions held out")]

    fig, axes = plt.subplots(1, len(cells), figsize=(11, 3.0), sharey=True)

    for ax, (key, title) in zip(axes, cells):
        labels = d[f"labels__{key}"]
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")

        for i, objective in enumerate(["mle", "map"]):
            probs = d[f"{objective}__{key}"]
            xs, ys, _ = reliability_from_probs(probs, labels)
            ax.plot(xs, ys, "o-", ms=3.5, lw=1.5, color=f"C{i}",
                    label=objective.upper())

            acc = (probs.argmax(1) == labels).mean()
            conf = probs.max(1).mean()
            if i == 0:
                ax.text(0.04, 0.93 - i * 0.09,
                        f"acc {acc:.2f} / conf {conf:.2f}",
                        fontsize=7, color=f"C{i}", transform=ax.transAxes)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("predicted confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("observed accuracy")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Reliability diagrams: below the diagonal means overconfident",
                 y=1.04, fontsize=10)
    path = os.path.join(outdir, "fig_reliability.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# =============================================================================
# FIGURE 4: THE CONFIDENCE GAP
# =============================================================================

def fig_confidence_gap(summary, outdir):
    """
    Accuracy and mean confidence side by side for every cell.

    This states the project's central finding as directly as possible: accuracy
    collapses as conditions become unfamiliar, while confidence barely moves.
    The vertical distance between the two lines IS the miscalibration.
    """
    agg = summary["aggregated"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)

    keys = [f"{s}__{t}" for s in SPLITS for t in TIERS]
    x = np.arange(len(keys))

    for ax, objective in zip(axes, ["mle", "map"]):
        acc = [agg[objective][k]["accuracy"] for k in keys]
        conf = [agg[objective][k]["mean_confidence"] for k in keys]

        ax.fill_between(x, acc, conf, alpha=0.25, color="C3",
                        label="overconfidence")
        ax.plot(x, conf, "s-", ms=3, lw=1.5, label="mean confidence")
        ax.plot(x, acc, "o-", ms=3, lw=1.5, label="accuracy")

        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("test_", "").replace("__", "\n")
                            for k in keys], fontsize=6, rotation=90)
        ax.set_title(objective.upper())
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("probability")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Confidence stays high while accuracy falls; "
                 "the shaded area is the calibration gap", y=1.02, fontsize=10)
    path = os.path.join(outdir, "fig_confidence_gap.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# =============================================================================
# FIGURE 5: CONFUSION MATRICES
# =============================================================================

def fig_confusion(summary, outdir):
    """
    Confusion matrices in and out of distribution.

    Shows whether errors concentrate on particular transmitters or spread
    evenly. Counts are summed over seeds and normalised by row, so each row
    reads as "of the bursts truly from this emitter, where did they go".
    """
    agg = summary["aggregated"]["mle"]
    cells = [("test_id__controlled", "in-distribution"),
             ("test_rx__controlled", "held-out receiver"),
             ("test_id__dynamic", "held-out sea state")]

    fig, axes = plt.subplots(1, len(cells), figsize=(10, 3.2))

    for ax, (key, title) in zip(axes, cells):
        cm = np.array(agg[key]["confusion_matrix"], dtype=float)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{title}\naccuracy {agg[key]['accuracy']:.3f}", fontsize=9)
        ax.set_xlabel("predicted")
        ax.grid(False)
        if ax is axes[0]:
            ax.set_ylabel("true emitter")

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label="fraction of true class")
    fig.suptitle("Confusion matrices, MLE (row-normalised, summed over seeds)",
                 y=1.04, fontsize=10)
    path = os.path.join(outdir, "fig_confusion.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# =============================================================================
# LATEX TABLES
# =============================================================================

def table_results(summary, outdir):
    """Main results table, formatted for direct \\input into the paper."""
    agg = summary["aggregated"]
    n_seeds = len(summary["seeds"])

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{ECE} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Test split & Sea state & MLE & MAP & MLE & MAP \\",
        r"\midrule",
    ]

    for split in SPLITS:
        for i, tier in enumerate(TIERS):
            key = f"{split}__{tier}"
            a, b = agg["mle"][key], agg["map"][key]
            name = SPLIT_LABEL[split] if i == 0 else ""
            held = r"$^\dagger$" if tier == "dynamic" else ""
            lines.append(
                f"{name} & {tier}{held} & "
                f"{a['accuracy']:.3f}\\,$\\pm$\\,{a['accuracy_std']:.3f} & "
                f"{b['accuracy']:.3f}\\,$\\pm$\\,{b['accuracy_std']:.3f} & "
                f"{a['ece']:.3f}\\,$\\pm$\\,{a['ece_std']:.3f} & "
                f"{b['ece']:.3f}\\,$\\pm$\\,{b['ece_std']:.3f} \\\\")
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    path = os.path.join(outdir, "table_results.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  {path}")


def table_architecture(summary, outdir):
    """
    Architecture table. The Results rubric asks for the architecture to be
    'explained and visually presented', and a layer table is the clearest way.
    """
    cfg = summary["config"]
    K = cfg["n_classes"]

    rows = [
        ("Input", "spectrogram, 3 channels", "3 x 64 x 13", "--"),
        ("Conv2d + ReLU", "32 filters, 3x3, pad 1", "32 x 64 x 13", "896"),
        ("MaxPool2d", "2x2", "32 x 32 x 6", "--"),
        ("Dropout", "p = 0.15", "32 x 32 x 6", "--"),
        ("Conv2d + ReLU", "48 filters, 3x3, pad 1", "48 x 32 x 6", "13{,}872"),
        ("MaxPool2d", "2x2", "48 x 16 x 3", "--"),
        ("Dropout", "p = 0.15", "48 x 16 x 3", "--"),
        ("Conv2d + ReLU", "96 filters, 3x3, pad 1", "96 x 16 x 3", "41{,}568"),
        ("AdaptiveAvgPool2d", "output 4x2", "96 x 4 x 2", "--"),
        ("Flatten", "--", "768", "--"),
        ("Linear + ReLU", "128 units", "128", "98{,}432"),
        ("Dropout", "p = 0.3", "128", "--"),
        ("Linear", f"{K} logits", f"{K}", f"{129*K:,}".replace(",", "{,}")),
    ]

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Layer & Configuration & Output shape & Parameters \\",
        r"\midrule",
    ]
    for name, config, shape, params in rows:
        lines.append(f"{name} & {config} & {shape} & {params} \\\\")
    lines += [
        r"\midrule",
        r"\textbf{Total} & & & \textbf{156{,}058} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    path = os.path.join(outdir, "table_architecture.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  {path}")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    summary = json.load(open(args.results))

    print(f"reading {args.results}  ({len(summary['seeds'])} seeds)")
    print("writing:")

    fig_loss_curves(summary, args.outdir)
    fig_accuracy_shift(summary, args.outdir)
    fig_reliability(args.probs, args.outdir)
    fig_confidence_gap(summary, args.outdir)
    fig_confusion(summary, args.outdir)
    table_results(summary, args.outdir)
    table_architecture(summary, args.outdir)

    print("\ndone. Figures and tables are in the paper directory, ready for "
          "\\includegraphics and \\input.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build Deliverable II figures.")
    p.add_argument("--results", default="results_spec5.json")
    p.add_argument("--probs", default="probabilities_spec5.npz")
    p.add_argument("--outdir", default="paper")
    main(p.parse_args())
