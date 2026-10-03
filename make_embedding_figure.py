"""
make_embedding_figure.py
===============================================================================
Embedding-space visualisation of the learned emitter representation.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS ANSWERS
-------------------------------------------------------------------------------
The accuracy tables say the classifier falls to near chance on conditions it
never trained on. They do not say why, and there are two very different reasons
it could happen:

    1. The fingerprint is still present in the signal, but the network's
       representation of it has shifted, so the decision boundaries no longer
       sit in the right place. More or better training data would help.

    2. The fingerprint is no longer in the signal at all, because the channel
       destroyed it. Nothing would help.

These have different names in this project. The first is epistemic, the second
aleatoric, and the whole argument about calibration depends on telling them
apart. This figure separates them by eye.

Each burst is pushed through the trained network and the 128-unit activation of
the first fully connected layer is kept. That vector is the network's internal
description of the burst, immediately before classification. Projecting it to
two dimensions shows whether bursts from the same emitter land near each other.

Three panels, chosen to isolate one variable at a time:

    trained conditions      familiar receiver, calm sea    clusters expected
    held-out receiver       new receiver, calm sea         the representation
                                                           shifts, but the
                                                           signal is intact
    fully held out          new receiver, roughest sea     signal destroyed

-------------------------------------------------------------------------------
THE NUMBER THAT GOES WITH THE PICTURE
-------------------------------------------------------------------------------
A scatter plot is an argument by appearance, and t-SNE in particular can suggest
structure that is not there: it preserves local neighbourhoods but distorts
distances, and its layout depends on the perplexity setting. So a separability
statistic is computed alongside it, in the ORIGINAL 128 dimensions rather than
in the projection:

    neighbour purity = fraction of bursts whose nearest neighbour in embedding
                       space carries the same emitter label

For ten emitters, 1.0 means perfectly separated and roughly 0.1 means the
embedding carries no emitter information at all. This number is not affected by
the projection, so it says whether the picture can be trusted.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
    pip install scikit-learn        # for t-SNE; falls back to PCA without it

    python make_embedding_figure.py --data data/prepared_spec.npz \
        --checkpoint model_mc_dropout_bayes5.pt --outdir paper
===============================================================================
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_dataset import load_prepared
from models import build_model

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# One panel per question. Order matters: each step changes one thing.
PANELS = [
    ("test_id__controlled", "Trained conditions",
     "familiar receiver, calm sea"),
    ("test_rx__controlled", "Held-out receiver",
     "new receiver, still calm sea"),
    ("test_both__dynamic", "Fully held out",
     "new receiver, roughest sea state"),
]


# =============================================================================
# EMBEDDINGS
# =============================================================================

@torch.no_grad()
def embed(model, X, device, batch_size=256):
    """Return the 128-unit fc1 activation for every burst in X.

    A forward hook is used rather than re-implementing the forward pass, so the
    features extracted are guaranteed to be the ones the classifier actually
    sees. Re-implementing it would risk the two drifting apart silently.
    """
    feats = []
    handle = model.fc1.register_forward_hook(
        lambda _m, _inp, out: feats.append(out.detach().cpu().numpy()))

    model.eval()          # deterministic: dropout masks off for this analysis
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        model(xb)

    handle.remove()
    return np.concatenate(feats)


def neighbour_purity(Z, y):
    """Fraction of points whose nearest neighbour shares their label.

        purity = (1/n) * sum_i  1[ y_{nn(i)} == y_i ]

    Computed in the full embedding space, not the 2-D projection, so it is a
    property of the representation rather than of the visualisation. Chance is
    1/K for balanced classes, which is 0.1 here.

    Distances via the expansion ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2, with the
    diagonal set to infinity so a point is never its own neighbour.
    """
    sq = (Z ** 2).sum(axis=1)
    d2 = sq[:, None] - 2.0 * (Z @ Z.T) + sq[None, :]
    np.fill_diagonal(d2, np.inf)
    return float((y[d2.argmin(axis=1)] == y).mean())


def project(Z, seed=0):
    """Project to two dimensions, preferring t-SNE and falling back to PCA.

    t-SNE is the standard choice for this picture because it preserves local
    neighbourhood structure, which is exactly what "do same-emitter bursts sit
    together" asks about. PCA is kept as a fallback so the script still runs
    without scikit-learn; it finds the directions of greatest variance, which
    need not be the directions that separate emitters, so clusters usually look
    weaker than they are. Which one ran is reported in the caption.
    """
    try:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=seed).fit_transform(Z)
        return proj, "t-SNE"
    except ImportError:
        Zc = Z - Z.mean(axis=0)
        _u, _s, vt = np.linalg.svd(Zc, full_matrices=False)
        return Zc @ vt[:2].T, "PCA"


def subsample(X, y, n, seed=0):
    """Take n points, stratified by emitter so no class is lost.

    t-SNE is quadratic in the number of points, and the largest split here holds
    8,500 bursts. Stratifying rather than sampling at random keeps the class
    balance identical across panels, so apparent differences in cluster size
    between panels are real and not an artefact of who got sampled.
    """
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    per_class = max(1, n // len(np.unique(y)))
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y == c),
                   size=min(per_class, int((y == c).sum())), replace=False)
        for c in np.unique(y)])
    rng.shuffle(keep)
    return X[keep], y[keep]


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")

    data, config = load_prepared(args.data)
    n_classes = config["n_classes"]

    sample_key = PANELS[0][0]
    in_channels = data[sample_key]["X"].shape[1]

    model = build_model("spectrogram", n_classes, dropout_p=args.dropout,
                        in_channels=in_channels).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"loaded {args.checkpoint}  ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"device {device}\n")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.5, 3.9))
    cmap = plt.get_cmap("tab10")
    method = None
    purities = {}

    for ax, (key, title, subtitle) in zip(axes, PANELS):
        X, y = subsample(data[key]["X"], data[key]["y"], args.n_points)
        Z = embed(model, X, device)

        purity = neighbour_purity(Z, y)
        purities[key] = purity
        proj, method = project(Z, seed=args.seed)

        for c in range(n_classes):
            m = y == c
            ax.scatter(proj[m, 0], proj[m, 1], s=4, alpha=0.65,
                       color=cmap(c % 10), linewidths=0,
                       label=f"emitter {c}" if ax is axes[0] else None)

        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.06, f"neighbour purity {purity:.2f}",
                transform=ax.transAxes, ha="center", fontsize=8.5)
        print(f"{key:<24} n={len(y):>5,}  neighbour purity {purity:.3f}")

    axes[0].legend(fontsize=6, markerscale=2, loc="upper left",
                   ncol=2, framealpha=0.85)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "fig_embedding.pdf")
    fig.savefig(path)
    plt.close(fig)

    print(f"\nprojection: {method}")
    print(f"chance purity for {n_classes} emitters: {1.0/n_classes:.2f}")
    print(f"wrote {path}")

    # Printed so the numbers quoted in the paper can be checked against a run
    # rather than copied from a previous one.
    print("\nfor the write-up:")
    for key, p in purities.items():
        print(f"  {key:<24} {p:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Embedding-space visualisation of the emitter representation.")
    p.add_argument("--data", default=os.path.join("data", "prepared_spec.npz"))
    p.add_argument("--checkpoint", default="model_mc_dropout_bayes5.pt")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--n-points", type=int, default=1500,
                   help="points per panel; t-SNE cost grows quadratically")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="paper")
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
