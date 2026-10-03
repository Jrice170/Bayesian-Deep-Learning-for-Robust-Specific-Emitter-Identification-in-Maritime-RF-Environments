"""
Bayesian model averaging and uncertainty decomposition.

Joseph M. Rice, LTJG, USN
CS 4323, Naval Postgraduate School

Takes the per-sample predictions produced by an approximate posterior and turns
them into a single predictive distribution plus a split of the uncertainty into
aleatoric and epistemic parts.

All functions take probs of shape (S, N, K):
    S  posterior samples
    N  test examples
    K  emitter classes
"""

import numpy as np

EPS = 1e-12


def bayesian_model_average(probs: np.ndarray) -> np.ndarray:
    """Average the categorical predictions over posterior samples.

        p(y | x, D) ~= (1/S) sum_s p(y | x, theta_s),    theta_s ~ q(theta)

    Returns (N, K). This is the prediction actually reported; the argmax gives
    the class and the max gives the confidence.
    """
    return probs.mean(axis=0)


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy of a categorical distribution.

        H[p] = -sum_k p_k log p_k

    Maximal (log K) when the distribution is uniform, zero when it is certain.
    """
    return -np.sum(p * np.log(p + EPS), axis=axis)


def decompose_uncertainty(probs: np.ndarray) -> dict:
    """Split predictive uncertainty into aleatoric and epistemic components.

        total      = H[ (1/S) sum_s p_s ]            entropy of the average
        aleatoric  = (1/S) sum_s H[ p_s ]            average of the entropies
        epistemic  = total - aleatoric               mutual information I(y; theta)

    The two terms answer different questions.

    ALEATORIC is the uncertainty that remains even if the weights were known
    exactly: noise in the signal itself. Every sampled model is individually
    unsure, and they are unsure in the same way. More data would not help.

    EPISTEMIC is disagreement BETWEEN sampled models. Each one may be confident,
    but they are confident about different answers, which means the training
    data did not determine the prediction for this input. More data would help.

    Epistemic uncertainty is the quantity of interest for this project: it is
    what should rise on a receiver, capture day or sea state the model never
    trained on. Because it equals total minus aleatoric, it is exactly the
    mutual information between the prediction and the weights, and it is zero
    when every sampled model gives the identical distribution.

    Returns a dict of (N,) arrays.
    """
    mean_probs = bayesian_model_average(probs)

    total = entropy(mean_probs)                 # (N,)
    aleatoric = entropy(probs).mean(axis=0)     # (N,)
    epistemic = total - aleatoric               # (N,)

    # Mutual information is non-negative in exact arithmetic; small negative
    # values can appear from floating point when the samples agree almost
    # perfectly, so clip at zero rather than reporting a negative uncertainty.
    epistemic = np.maximum(epistemic, 0.0)

    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}


def predictive_variance(probs: np.ndarray) -> dict:
    """Variance-based decomposition, as an alternative to the entropy one.

        Var[y] = E_theta[ Var(y | theta) ]  +  Var_theta[ E(y | theta) ]
                 \\____ aleatoric ____/        \\____ epistemic ____/

    Computed on the one-hot representation of the categorical output and summed
    over classes. Reported alongside the entropy decomposition because Module 3
    presents both, and because the two can disagree in informative ways.
    """
    # Var(y|theta) for a categorical is p(1-p) per class.
    within = (probs * (1.0 - probs)).sum(axis=2).mean(axis=0)   # (N,)
    between = probs.var(axis=0).sum(axis=1)                      # (N,)
    return {"aleatoric": within, "epistemic": between,
            "total": within + between}


def sample_diversity(probs: np.ndarray) -> dict:
    """Diagnostic: how much do the posterior samples actually differ?

    If a method's samples are nearly identical, its epistemic uncertainty will
    be near zero regardless of the input, and the method is not contributing
    anything over a point estimate. Worth checking before interpreting results,
    particularly for Monte Carlo dropout at a low dropout rate.

    Returns the mean standard deviation of the predicted probability of the
    winning class, and the fraction of examples where the sampled models
    disagree about which class wins.
    """
    mean_probs = bayesian_model_average(probs)
    winner = mean_probs.argmax(axis=1)                       # (N,)

    winner_probs = probs[:, np.arange(probs.shape[1]), winner]   # (S, N)
    spread = winner_probs.std(axis=0).mean()

    per_sample_argmax = probs.argmax(axis=2)                 # (S, N)
    disagree = (per_sample_argmax != winner[None, :]).any(axis=0).mean()

    return {"prob_std": float(spread), "argmax_disagreement": float(disagree)}


def expected_calibration_error(probs_mean: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15):
    """Expected calibration error of the Bayesian-model-averaged prediction.

        ECE = sum_b (n_b / N) | acc(b) - conf(b) |

    Identical to the Deliverable II implementation so the numbers are directly
    comparable; takes the averaged probabilities rather than the raw samples.
    """
    confidence = probs_mean.max(axis=1)
    prediction = probs_mean.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else \
                 (confidence >= lo) & (confidence <= hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            bin_stats.append({"lo": float(lo), "hi": float(hi), "n": 0,
                              "accuracy": None, "confidence": None})
            continue
        acc = float(correct[in_bin].mean())
        conf = float(confidence[in_bin].mean())
        ece += (n_in / len(labels)) * abs(acc - conf)
        bin_stats.append({"lo": float(lo), "hi": float(hi), "n": n_in,
                          "accuracy": acc, "confidence": conf})

    return float(ece), bin_stats


def summarise(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Everything reported for one evaluation cell."""
    mean_probs = bayesian_model_average(probs)
    preds = mean_probs.argmax(axis=1)

    ent = decompose_uncertainty(probs)
    var = predictive_variance(probs)
    div = sample_diversity(probs)
    ece, bins = expected_calibration_error(mean_probs, labels)

    correct = preds == labels
    return {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(mean_probs.max(axis=1).mean()),
        "ece": ece,
        "reliability_bins": bins,
        "entropy_total": float(ent["total"].mean()),
        "entropy_aleatoric": float(ent["aleatoric"].mean()),
        "entropy_epistemic": float(ent["epistemic"].mean()),
        # Epistemic uncertainty split by whether the prediction was right.
        # A useful method should be more uncertain on the ones it got wrong.
        "epistemic_correct": float(ent["epistemic"][correct].mean()) if correct.any() else None,
        "epistemic_wrong": float(ent["epistemic"][~correct].mean()) if (~correct).any() else None,
        "variance_aleatoric": float(var["aleatoric"].mean()),
        "variance_epistemic": float(var["epistemic"].mean()),
        "sample_prob_std": div["prob_std"],
        "sample_argmax_disagreement": div["argmax_disagreement"],
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 66)
    print("UNCERTAINTY DECOMPOSITION - SELF TEST")
    print("=" * 66)

    rng = np.random.default_rng(0)
    S, N, K = 30, 2000, 10

    def onehotish(idx, conf, n, k):
        """Build (n,k) probs with `conf` on class idx and the rest spread."""
        p = np.full((n, k), (1 - conf) / (k - 1))
        p[np.arange(n), idx] = conf
        return p

    # --- 1. All samples identical and confident -----------------------------
    # No disagreement -> epistemic must be ~0. Each model is confident ->
    # aleatoric must also be low.
    print("\n[1] Identical, confident samples")
    idx = rng.integers(0, K, N)
    p = onehotish(idx, 0.95, N, K)
    probs = np.repeat(p[None], S, axis=0)
    d = decompose_uncertainty(probs)
    print(f"    total {d['total'].mean():.4f}  aleatoric {d['aleatoric'].mean():.4f}"
          f"  epistemic {d['epistemic'].mean():.6f}")
    print(f"    epistemic ~ 0: {'PASS' if d['epistemic'].mean() < 1e-6 else 'FAIL'}")

    # --- 2. All samples identical but uncertain -----------------------------
    # Still no disagreement -> epistemic ~0, but aleatoric is now high.
    # This is the case the decomposition exists to distinguish from case 3.
    print("\n[2] Identical, uncertain samples (uniform)")
    probs = np.repeat(np.full((N, K), 1.0 / K)[None], S, axis=0)
    d = decompose_uncertainty(probs)
    print(f"    total {d['total'].mean():.4f}  aleatoric {d['aleatoric'].mean():.4f}"
          f"  epistemic {d['epistemic'].mean():.6f}")
    print(f"    total ~ log(K) = {np.log(K):.4f}: "
          f"{'PASS' if abs(d['total'].mean() - np.log(K)) < 1e-3 else 'FAIL'}")
    print(f"    epistemic ~ 0:  {'PASS' if d['epistemic'].mean() < 1e-6 else 'FAIL'}")

    # --- 3. Samples confident but disagreeing -------------------------------
    # Each model is certain, but they point at different classes. Aleatoric
    # should be low and epistemic high. This is the OOD signature.
    print("\n[3] Confident samples that disagree with each other")
    probs = np.stack([onehotish(rng.integers(0, K, N), 0.95, N, K)
                      for _ in range(S)])
    d = decompose_uncertainty(probs)
    print(f"    total {d['total'].mean():.4f}  aleatoric {d['aleatoric'].mean():.4f}"
          f"  epistemic {d['epistemic'].mean():.4f}")
    ok = d["epistemic"].mean() > d["aleatoric"].mean()
    print(f"    epistemic > aleatoric: {ok}   {'PASS' if ok else 'FAIL'}")
    print("    -> this is what an out-of-distribution input should look like")

    # --- 4. Decomposition adds up -------------------------------------------
    print("\n[4] total == aleatoric + epistemic")
    err = np.abs(d["total"] - (d["aleatoric"] + d["epistemic"])).max()
    print(f"    max discrepancy {err:.2e}   {'PASS' if err < 1e-9 else 'FAIL'}")

    # --- 5. Bounds -----------------------------------------------------------
    print("\n[5] Entropy bounded by log(K)")
    ok = d["total"].max() <= np.log(K) + 1e-9
    print(f"    max total {d['total'].max():.4f} <= log({K}) = {np.log(K):.4f}"
          f"   {'PASS' if ok else 'FAIL'}")

    # --- 6. Diversity diagnostic --------------------------------------------
    print("\n[6] Sample diversity diagnostic")
    identical = np.repeat(onehotish(idx, 0.95, N, K)[None], S, axis=0)
    varied = np.stack([onehotish(rng.integers(0, K, N), 0.95, N, K) for _ in range(S)])
    a, b = sample_diversity(identical), sample_diversity(varied)
    print(f"    identical samples: prob_std {a['prob_std']:.4f}  "
          f"disagreement {a['argmax_disagreement']:.3f}")
    print(f"    varied samples   : prob_std {b['prob_std']:.4f}  "
          f"disagreement {b['argmax_disagreement']:.3f}")
    ok = a["prob_std"] < 1e-9 and b["prob_std"] > 0.1
    print(f"    diagnostic separates the two cases: {'PASS' if ok else 'FAIL'}")

    # --- 7. Variance decomposition ------------------------------------------
    print("\n[7] Variance decomposition agrees in direction with entropy")
    v_id = predictive_variance(identical)
    v_var = predictive_variance(varied)
    ok = v_id["epistemic"].mean() < v_var["epistemic"].mean()
    print(f"    epistemic variance: identical {v_id['epistemic'].mean():.5f}  "
          f"varied {v_var['epistemic'].mean():.5f}   {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 66)
