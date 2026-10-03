"""
make_figures_bayes.py
===============================================================================
Figures and LaTeX tables for the approximate Bayesian results.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS PRODUCES
-------------------------------------------------------------------------------
    fig_reliability_bayes.pdf   reliability diagrams, five methods, two cells
    fig_ece_methods.pdf         calibration error across the shift axes
    fig_uncertainty_split.pdf   aleatoric and epistemic, familiar vs shifted
    fig_dropout_rates.pdf       the rates Concrete Dropout learned, per seed
    fig_posterior_sigma.pdf     posterior width per layer against the prior
    table_bayes_results.tex     main results table with seed error bars
    table_uncertainty.tex       uncertainty decomposition table

The point-estimate results from the previous stage are read in as well, so the
Bayesian methods are plotted against the MLE and MAP baselines rather than on
their own. A calibration number means little in isolation; what carries the
argument is the comparison.

Run after training:

    python make_figures_bayes.py --results results_bayes5.json \\
                                 --probs probabilities_bayes5.npz \\
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
    "test_id": "nothing held out",
    "test_rx": "receiver",
    "test_day": "capture day",
    "test_both": "receiver + day",
}

# Point estimates first, then the three posteriors, so the legend reads as a
# progression from no uncertainty representation to increasingly explicit ones.
METHOD_LABEL = {
    "mle": "MLE",
    "map": "MAP",
    "mc_dropout": "MC dropout",
    "gaussian_vi": "Gaussian VI",
    "concrete_vi": r"Concrete VI ($d{=}1$)",
    "concrete_vi_d10": r"Concrete VI ($d{=}10$)",
}
METHOD_ORDER = ["mle", "map", "mc_dropout", "gaussian_vi",
                "concrete_vi", "concrete_vi_d10"]

# Shorter forms for table headers. With six methods the full names overrun the
# text block; figures keep the full labels, where there is room.
METHOD_SHORT = {
    "mle": "MLE",
    "map": "MAP",
    "mc_dropout": "MC drop",
    "gaussian_vi": "Gauss VI",
    "concrete_vi": r"Conc $d{=}1$",
    "concrete_vi_d10": r"Conc $d{=}10$",
}
METHOD_COLOR = {
    "mle": "#999999",
    "map": "#555555",
    "mc_dropout": "#1f77b4",
    "gaussian_vi": "#d62728",
    "concrete_vi": "#2ca02c",
    "concrete_vi_d10": "#8c564b",
}

# The two cells that anchor every comparison: the easiest and the hardest.
EASY_CELL = "test_id__controlled"
HARD_CELL = "test_both__dynamic"


# =============================================================================
# SHARED
# =============================================================================

def reliability_curve(probs, labels, n_bins=15, min_count=25):
    """Accuracy against confidence, binned.

        for each bin b:  conf(b) = mean predicted probability of the winner
                         acc(b)  = fraction actually correct

    A perfectly calibrated model traces the diagonal. Points below it are
    overconfident, which is the failure mode this project is about.

    WHY SPARSE BINS ARE DROPPED. The accuracy in a bin is a proportion estimated
    from however many predictions landed there. The fully held-out cell holds
    only 500 examples, so the high-confidence bins can end up with a handful
    each, and a bin containing five predictions reports accuracies of 0, 0.2,
    0.4 and so on with a standard error near 0.22. Plotted unfiltered those
    bins dominate the visual impression of the curve while carrying almost no
    evidence. Bins below min_count are therefore dropped rather than drawn, and
    the marker area of the rest is scaled by population so the reader can see
    which points are actually supported.

    This affects the figure only. The reported ECE uses every bin, weighted by
    population exactly as the definition requires, so nothing is being excluded
    from the numbers.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, weight = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else \
                 (confidence >= lo) & (confidence <= hi)
        n_in = int(in_bin.sum())
        if n_in < min_count:
            continue
        conf.append(confidence[in_bin].mean())
        acc.append(correct[in_bin].mean())
        weight.append(n_in)
    return np.array(conf), np.array(acc), np.array(weight)


def load_all_probs(bayes_npz, point_npz):
    """Merge the Bayesian and point-estimate probability files into one dict.

    Both stages saved first-seed probabilities keyed "<method>__<cell>", so the
    merge is a rename away. Labels are identical between the two files because
    the prepared dataset was frozen before either stage ran; that is asserted
    rather than assumed, since a silent mismatch would corrupt every figure.
    """
    out = {}
    b = np.load(bayes_npz)
    p = np.load(point_npz)

    for key in b.files:
        if key.startswith("labels__") or key.startswith("samples__"):
            continue
        out[key] = b[key]
    for key in p.files:
        if key.startswith("labels__"):
            continue
        out[key] = p[key]

    labels = {k[len("labels__"):]: b[k] for k in b.files if k.startswith("labels__")}
    for k in p.files:
        if not k.startswith("labels__"):
            continue
        cell = k[len("labels__"):]
        if cell in labels and not np.array_equal(labels[cell], p[k]):
            raise ValueError(
                f"labels for {cell} differ between the two result files. The "
                f"prepared dataset must be the same for the comparison to mean "
                f"anything.")

    samples = {k[len("samples__"):]: b[k] for k in b.files
               if k.startswith("samples__")}
    return out, labels, samples


def cells_present(probs, method):
    return sorted(k.split("__", 1)[1] for k in probs if k.startswith(method + "__"))


# =============================================================================
# FIGURE 1: RELIABILITY DIAGRAMS
# =============================================================================

def fig_reliability(probs, labels, outdir):
    """The central figure: are the stated confidences honest?

    Left panel is the condition the model trained on, right panel is the one it
    never saw. On the left everything sits near the diagonal and the methods are
    hard to tell apart. On the right the point estimates fall well below it,
    which is the visual statement of the whole project: high confidence, low
    accuracy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)

    for ax, cell, title in [
            (axes[0], EASY_CELL, "Trained conditions\n(no shift)"),
            (axes[1], HARD_CELL, "Unseen receiver, day and sea state")]:

        # Everything below the diagonal claims more than it delivers. Shading
        # the region says so without a text label competing with the curves.
        ax.fill_between([0, 1], [0, 1], [0, 0], color="#d62728", alpha=0.05,
                        zorder=0, lw=0)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
                label="perfect calibration", zorder=1)

        for method in METHOD_ORDER:
            key = f"{method}__{cell}"
            if key not in probs or cell not in labels:
                continue
            conf, acc, weight = reliability_curve(probs[key], labels[cell])
            if len(conf) == 0:
                continue
            ax.plot(conf, acc, "-", lw=1.4, color=METHOD_COLOR[method],
                    label=METHOD_LABEL[method], zorder=2)
            # Marker area tracks how many predictions support each point.
            ax.scatter(conf, acc, s=8 + 40 * weight / weight.max(),
                       color=METHOD_COLOR[method], zorder=3, edgecolor="white",
                       linewidth=0.4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_title(title, fontsize=9)

    axes[0].set_ylabel("accuracy")
    axes[1].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    axes[0].text(0.96, 0.04, "shaded: overconfident", ha="right", fontsize=7,
                 style="italic", alpha=0.7)

    path = os.path.join(outdir, "fig_reliability_bayes.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# =============================================================================
# FIGURE 2: CALIBRATION ERROR ACROSS THE SHIFT AXES
# =============================================================================

def fig_ece_methods(bayes, point, outdir):
    """Expected calibration error for every method in every cell.

    Grouped by split so the progression along each shift axis is readable, with
    seed standard deviations as error bars. Bars that overlap should not be
    described as different in the text.
    """
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(11, 3.2), sharey=True)

    width = 0.16
    x = np.arange(len(TIERS))

    for ax, split in zip(axes, SPLITS):
        for i, method in enumerate(METHOD_ORDER):
            src = point if method in ("mle", "map") else bayes
            table = src[method]

            means = [table.get(f"{split}__{t}", {}).get("ece", np.nan)
                     for t in TIERS]
            errs = [table.get(f"{split}__{t}", {}).get("ece_std", 0.0)
                    for t in TIERS]

            ax.bar(x + (i - 2) * width, means, width, yerr=errs, capsize=2,
                   color=METHOD_COLOR[method], label=METHOD_LABEL[method],
                   error_kw={"lw": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(TIERS, rotation=20, ha="right")
        ax.set_title(SPLIT_LABEL[split], fontsize=9)

    axes[0].set_ylabel("expected calibration error")
    axes[0].legend(fontsize=7, ncol=1, framealpha=0.9)

    path = os.path.join(outdir, "fig_ece_methods.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# =============================================================================
# FIGURE 3: WHERE THE UNCERTAINTY COMES FROM
# =============================================================================

def fig_uncertainty_split(bayes, outdir):
    """Aleatoric and epistemic uncertainty, stacked, familiar against shifted.

    The stacked height is total predictive entropy, capped at log K = 2.303 for
    ten classes. The split matters more than the height. Epistemic uncertainty
    is disagreement between posterior samples, so it reflects the model not
    knowing; aleatoric uncertainty is agreement that the signal itself is
    ambiguous. Under severe channel corruption the second dominates, because
    every sampled model reaches the same conclusion that the fingerprint has
    been destroyed.

    A point estimate cannot appear on this figure at all. One set of weights
    means no disagreement to measure, so its epistemic term is exactly zero by
    construction, which is the structural limitation the whole stage addresses.
    """
    methods = [m for m in METHOD_ORDER if m in bayes]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    x = np.arange(len(methods))
    width = 0.36

    for offset, cell, hatch, tag in [(-width / 2, EASY_CELL, "", "familiar"),
                                     (+width / 2, HARD_CELL, "///", "shifted")]:
        alea = [bayes[m][cell]["entropy_aleatoric"] for m in methods]
        epis = [bayes[m][cell]["entropy_epistemic"] for m in methods]

        ax.bar(x + offset, alea, width, color="#c6dbef", hatch=hatch,
               edgecolor="white", label=f"aleatoric, {tag}")
        ax.bar(x + offset, epis, width, bottom=alea, color="#d62728",
               hatch=hatch, edgecolor="white", label=f"epistemic, {tag}")

    ax.axhline(np.log(10), color="k", ls=":", lw=1)
    ax.text(len(methods) - 0.5, np.log(10) + 0.04,
            "log K = 2.303, complete ignorance", ha="right", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods])
    ax.set_ylabel("predictive entropy (nats)")
    # Headroom so the legend clears the bars and the log K annotation.
    ax.set_ylim(0, np.log(10) * 1.30)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)

    path = os.path.join(outdir, "fig_uncertainty_split.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# =============================================================================
# FIGURE 4: WHAT CONCRETE DROPOUT LEARNED
# =============================================================================

def fig_dropout_rates(summary, outdir, summary_d10=None, sweep=None):
    """What Concrete Dropout fitted, and how much of it was the regulariser.

    LEFT PANEL. Rates at the two regulariser strengths, one line per seed. Every
    seed is drawn rather than a mean, because the reproducibility is the result:
    five independent runs landing on the same per-layer pattern means the
    pattern is not an initialisation artefact.

    RIGHT PANEL. Mean fitted rate against regulariser scale, with the spread
    across layers as a shaded band. This is the panel that carries the argument.
    The entropy term is the only part of the objective resisting collapse to
    p = 0, and its coefficient is scale * input_dim / N. At the default scale the
    likelihood overwhelms it and every rate sits near zero. Raise it far enough
    and the reverse happens: the rates converge on p = 0.5, which is simply
    where Bernoulli entropy is largest, so the regulariser is reporting its own
    optimum rather than anything about the data. Between those two failures the
    band widens, meaning the layers are being distinguished from one another.

    The fitted rate is therefore only informative in a middle range that has to
    be found by sweeping, which is the hyper-parameter Concrete Dropout was
    introduced to remove.
    """
    layers = ["conv1", "conv2", "conv3", "fc1"]
    x = np.arange(len(layers))

    def rates_of(s):
        return np.array([h["dropout_rates"] for h in
                         s["histories"].get("concrete_vi", [])
                         if "dropout_rates" in h])

    r1 = rates_of(summary)
    if r1.size == 0:
        print("  no learned dropout rates recorded, skipping")
        return
    r10 = rates_of(summary_d10) if summary_d10 else np.empty((0, len(layers)))

    ncols = 2 if sweep else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9.6 if sweep else 5.6, 3.3))
    ax = axes[0] if sweep else axes

    for s, row in enumerate(r1):
        ax.plot(x, row, "o-", ms=4, lw=1.2, alpha=0.8, color="#2ca02c",
                label="$d=1$" if s == 0 else None)
    for s, row in enumerate(r10):
        ax.plot(x, row, "s-", ms=4, lw=1.2, alpha=0.8, color="#8c564b",
                label="$d=10$" if s == 0 else None)

    fixed = summary["args"].get("dropout", 0.3)
    ax.axhline(fixed, color="k", ls="--", lw=1)
    ax.text(len(layers) - 1, fixed + 0.012,
            f"fixed rate used for MC dropout, $p={fixed}$",
            ha="right", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel("fitted dropout rate  $p$")
    ax.set_ylim(0, 0.62)
    ax.set_title("Fitted rates, five seeds each", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")

    if sweep:
        ax2 = axes[1]
        scales = sorted(sweep)
        mean = [np.mean(sweep[s]) for s in scales]
        lo = [np.min(sweep[s]) for s in scales]
        hi = [np.max(sweep[s]) for s in scales]

        ax2.fill_between(scales, lo, hi, alpha=0.2, color="#8c564b",
                         label="range across layers")
        ax2.plot(scales, mean, "o-", color="#8c564b", ms=5, lw=1.5,
                 label="mean across layers")
        ax2.axhline(0.5, color="k", ls=":", lw=1)
        ax2.text(scales[-1], 0.515, "$p=0.5$: entropy maximum",
                 ha="right", fontsize=7.5)

        ax2.set_xscale("log")
        ax2.set_xticks(scales)
        ax2.set_xticklabels([str(s) for s in scales])
        ax2.set_xlabel("entropy regulariser scale  $d \\times N$")
        ax2.set_ylabel("fitted dropout rate  $p$")
        ax2.set_ylim(0, 0.62)
        ax2.set_title("Rates track the regulariser, not the data", fontsize=9)
        ax2.legend(fontsize=7.5, loc="lower right")

    path = os.path.join(outdir, "fig_dropout_rates.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def table_dropout_sweep(sweep_runs, outdir):
    """Accuracy, calibration and fitted rates against regulariser strength.

    One seed per row, which is enough because the effect across scales is far
    larger than the seed variation measured at any one of them.
    """
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Effect of the entropy regulariser scale on Concrete Dropout "
        r"(seed 0). The fitted rates collapse towards $0$ when the term is weak "
        r"and saturate at the Bernoulli entropy maximum of $0.5$ when it is "
        r"strong; the spread across layers, which measures how far the data "
        r"still distinguishes them, is largest in between. Calibration improves "
        r"monotonically while accuracy falls.}",
        r"\label{tab:dropout-sweep}",
        r"\small",
        r"\begin{tabular}{rrrrrl}",
        r"\toprule",
        r"Scale & Mean $p$ & Spread & Mean accuracy & Mean ECE & Fitted rates \\",
        r"\midrule",
    ]
    for scale in sorted(sweep_runs):
        r = sweep_runs[scale]
        rates = r["rates"]
        spread = max(rates) - min(rates)
        fitted = ", ".join(f"{v:.2f}" for v in rates)
        lines.append(f"{scale} & {np.mean(rates):.3f} & {spread:.3f} & "
                     f"{r['accuracy']:.3f} & {r['ece']:.3f} & {fitted} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(outdir, "table_dropout_sweep.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# =============================================================================
# FIGURE 5: HOW WIDE THE GAUSSIAN POSTERIOR GOT
# =============================================================================

def fig_posterior_sigma(summary, outdir):
    """Posterior standard deviation per layer, against the prior.

    Reading this figure: a layer whose posterior stays far below the prior has
    been pinned down by the data. A layer that drifts up towards the prior has
    not, and the posterior is falling back on the prior because the likelihood
    has little to say about those weights.

    This is also the collapse check. Had every layer risen to meet the prior,
    the KL term would have overwhelmed the likelihood and the model would have
    learned nothing, which is a known failure of variational inference with a
    wide prior and is why it is plotted rather than assumed absent.
    """
    histories = summary["histories"].get("gaussian_vi", [])
    sigmas = [h["posterior_sigma"] for h in histories if "posterior_sigma" in h]
    if not sigmas:
        print("  no posterior sigmas recorded, skipping")
        return

    layers = list(sigmas[0].keys())
    x = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for s, d in enumerate(sigmas):
        ax.plot(x, [d[k] for k in layers], "o-", ms=4, lw=1.2, alpha=0.75,
                label=f"seed {s}")

    n_train = summary.get("n_train", 35720)
    l2 = summary["args"].get("l2_coeff", 1e-4)
    prior_sigma = float(np.sqrt(1.0 / (2.0 * n_train * l2)))

    ax.axhline(prior_sigma, color="k", ls="--", lw=1)
    ax.text(len(layers) - 1, prior_sigma + 0.012,
            f"prior sigma = {prior_sigma:.3f}", ha="right", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel("posterior standard deviation")
    ax.set_ylim(0, prior_sigma * 1.25)
    ax.legend(fontsize=7, ncol=2)

    path = os.path.join(outdir, "fig_posterior_sigma.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# =============================================================================
# TABLES
# =============================================================================

def table_bayes_results(bayes, point, outdir):
    """Main results table: accuracy and calibration error, all five methods."""
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Accuracy and expected calibration error across every test "
        r"split and sea-state tier, averaged over five seeds. The dynamic tier "
        r"and the listed conditions were withheld from training. Seed-to-seed "
        r"variation is omitted here for width and is shown as error bars in "
        r"Figure~\ref{fig:ece-methods}.}",
        r"\label{tab:bayes-results}",
        r"\footnotesize",
        # Column count follows METHOD_ORDER so adding a method cannot silently
        # break the alignment.
        r"\begin{tabular}{ll" + "r" * len(METHOD_ORDER) + "}",
        r"\toprule",
        "Held out & Sea state & "
        + " & ".join(METHOD_SHORT[m] for m in METHOD_ORDER) + r" \\",
        r"\midrule",
    ]

    for metric, heading in [("accuracy", "Accuracy"), ("ece", "Calibration error")]:
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{heading}}}}} \\")
        for split in SPLITS:
            for tier in TIERS:
                cell = f"{split}__{tier}"
                row = [SPLIT_LABEL[split].replace("+", r"\&"), tier]
                for method in METHOD_ORDER:
                    src = point if method in ("mle", "map") else bayes
                    r = src.get(method, {}).get(cell)
                    row.append("--" if r is None else
                               f"{r[metric]:.3f}")
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    path = os.path.join(outdir, "table_bayes_results.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def table_poi(bayes, point, outdir):
    """Probability of identification against the SNR band of each sea-state tier.

    The split is held at test_id throughout, so the receiver and the capture day
    are familiar and the only thing varying down the table is channel severity.
    That isolates the effect of signal-to-noise ratio from the effect of
    unfamiliar hardware, which the main results table deliberately conflates.

    Reported by tier rather than as a continuous curve because the per-burst SNR
    drawn during augmentation is applied and then discarded rather than stored;
    see the note in prepare_dataset.py. Each tier does span a known range, so
    this is the same measurement at coarser resolution.
    """
    bands = {"controlled": r"$+15$ to $+25$",
             "degraded":   r"$+5$ to $+15$",
             "dynamic":    r"$-5$ to $0$"}

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Probability of identification by sea-state tier, with the "
        r"signal-to-noise range each tier spans. The receiver and capture day "
        r"are familiar throughout, so severity is the only quantity varying "
        r"down the table. Chance is $0.100$ for ten emitters. The Dynamic tier "
        r"was withheld from training.}",
        r"\label{tab:poi}",
        r"\footnotesize",
        r"\begin{tabular}{ll" + "r" * len(METHOD_ORDER) + "}",
        r"\toprule",
        "Sea state & SNR (dB) & "
        + " & ".join(METHOD_SHORT[m] for m in METHOD_ORDER) + r" \\",
        r"\midrule",
    ]

    for tier in TIERS:
        cell = f"test_id__{tier}"
        row = [tier, bands[tier]]
        for method in METHOD_ORDER:
            src = point if method in ("mle", "map") else bayes
            r = src.get(method, {}).get(cell)
            row.append("--" if r is None else f"{r['accuracy']:.3f}")
        marker = r"$^{\dagger}$" if tier == "dynamic" else ""
        row[0] = row[0] + marker
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt]\footnotesize $^{\dagger}$withheld from training.",
        r"\end{table}",
    ]

    path = os.path.join(outdir, "table_poi.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def table_uncertainty(bayes, outdir):
    """Uncertainty decomposition on the easiest and hardest cells."""
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Predictive entropy split into aleatoric and epistemic parts, "
        r"on trained conditions and on the fully held-out cell. The point "
        r"estimates are absent because a single set of weights has no "
        r"epistemic term to measure.}",
        r"\label{tab:uncertainty}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & Conditions & Total & Aleatoric & Epistemic & "
        r"Wrong / right \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        if method not in bayes:
            continue
        for cell, tag in [(EASY_CELL, "trained"), (HARD_CELL, "held out")]:
            r = bayes[method][cell]
            w, c = r.get("epistemic_wrong"), r.get("epistemic_correct")
            wr = "--" if not (w and c) else f"{w / c:.2f}"
            lines.append(
                f"{METHOD_LABEL[method]} & {tag} & "
                f"{r['entropy_total']:.3f} & {r['entropy_aleatoric']:.3f} & "
                f"{r['entropy_epistemic']:.3f} & {wr} " + r"\\")
        lines.append(r"\addlinespace")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    path = os.path.join(outdir, "table_uncertainty.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.results) as f:
        summary = json.load(f)
    with open(args.point_results) as f:
        point_summary = json.load(f)

    bayes = summary["aggregated"]
    point = point_summary["aggregated"]

    probs, labels, _samples = load_all_probs(args.probs, args.point_probs)

    # The corrected Concrete Dropout run is carried alongside the default one
    # rather than replacing it, because the contrast between the two is the
    # result. It enters under its own method key so every figure and table picks
    # it up without special-casing.
    summary_d10 = None
    if args.concrete_d10 and os.path.exists(args.concrete_d10):
        with open(args.concrete_d10) as f:
            summary_d10 = json.load(f)
        bayes["concrete_vi_d10"] = summary_d10["aggregated"]["concrete_vi"]
        d10_probs = args.concrete_d10.replace("results_", "probabilities_") \
                                     .replace(".json", ".npz")
        if os.path.exists(d10_probs):
            extra = np.load(d10_probs)
            for key in extra.files:
                if key.startswith("concrete_vi__"):
                    probs["concrete_vi_d10__" + key.split("__", 1)[1]] = extra[key]
        print(f"merged {args.concrete_d10} as concrete_vi_d10")
    else:
        METHOD_ORDER.remove("concrete_vi_d10")

    # One-seed sweep over regulariser strength, for the mechanism figure. The
    # default scale comes from the main run's first seed so the sweep starts at
    # the setting the method ships with.
    sweep_rates, sweep_runs = {}, {}
    base_cells = summary["per_seed"]["concrete_vi"][0]
    base_rates = summary["histories"]["concrete_vi"][0]["dropout_rates"]
    sweep_rates[1] = base_rates
    sweep_runs[1] = {
        "rates": base_rates,
        "accuracy": float(np.mean([base_cells[c]["accuracy"] for c in base_cells])),
        "ece": float(np.mean([base_cells[c]["ece"] for c in base_cells])),
    }

    for spec in args.sweep:
        scale, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  sweep file missing, skipping: {path}")
            continue
        with open(path) as f:
            s = json.load(f)
        rates = s["histories"]["concrete_vi"][0]["dropout_rates"]
        cells = s["per_seed"]["concrete_vi"][0]
        sweep_rates[int(scale)] = rates
        sweep_runs[int(scale)] = {
            "rates": rates,
            "accuracy": float(np.mean([cells[c]["accuracy"] for c in cells])),
            "ece": float(np.mean([cells[c]["ece"] for c in cells])),
        }

    print("building figures")
    fig_reliability(probs, labels, args.outdir)
    fig_ece_methods(bayes, point, args.outdir)
    fig_uncertainty_split(bayes, args.outdir)
    fig_dropout_rates(summary, args.outdir, summary_d10, sweep_rates or None)
    fig_posterior_sigma(summary, args.outdir)

    print("building tables")
    table_bayes_results(bayes, point, args.outdir)
    table_poi(bayes, point, args.outdir)
    table_uncertainty(bayes, args.outdir)
    if sweep_runs:
        table_dropout_sweep(sweep_runs, args.outdir)

    # ---- the numbers quoted in the text, printed so they can be checked ------
    print("\nnumbers for the write-up")
    print(f"{'method':<14}{'mean ECE':>10}{'ECE hard':>10}{'conf hard':>11}"
          f"{'acc hard':>10}")
    print("-" * 56)
    for method in METHOD_ORDER:
        src = point if method in ("mle", "map") else bayes
        table = src.get(method)
        if not table:
            continue
        mean_ece = np.mean([table[c]["ece"] for c in table])
        hard = table[HARD_CELL]
        print(f"{METHOD_LABEL[method]:<14}{mean_ece:>10.4f}"
              f"{hard['ece']:>10.4f}{hard['mean_confidence']:>11.4f}"
              f"{hard['accuracy']:>10.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Figures and tables for the approximate Bayesian results.")
    p.add_argument("--results",
                   default=os.path.join("results", "results_bayes5.json"))
    p.add_argument("--probs",
                   default=os.path.join("results", "probabilities_bayes5.npz"))
    p.add_argument("--point-results",
                   default=os.path.join("results", "results_spec5.json"))
    p.add_argument("--point-probs",
                   default=os.path.join("results", "probabilities_spec5.npz"))
    p.add_argument("--outdir", default="paper")
    p.add_argument("--concrete-d10",
                   default=os.path.join("results", "results_cd10_s5.json"),
                   help="five-seed Concrete Dropout run at the corrected "
                        "regulariser scale; carried alongside the default run")
    p.add_argument("--sweep", nargs="*",
                   default=["10=results/results_cd_s10.json",
                            "50=results/results_cd_s50.json",
                            "100=results/results_cd_s100.json"],
                   help="scale=path pairs for the regulariser sweep")
    main(p.parse_args())
