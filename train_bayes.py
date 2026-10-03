"""
train_bayes.py
===============================================================================
Approximate Bayesian inference for the SEI capstone project.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
-------------------------------------------------------------------------------
The point-estimation stage produced a classifier that was 94.6% accurate on
familiar conditions and near chance on an unseen sea state while still reporting
57% confidence. A single set of weights has no way to say "I have not seen this
before", so the obvious next step is to replace the point estimate with a
distribution over weights.

Three methods are run here, all evaluated on exactly the same twelve cells
(four test splits x three sea-state tiers) that the point estimates were:

    gaussian_vi    mean-field Gaussian variational inference. Each weight gets
                   a mean and a variance, both learned.

    concrete_vi    variational inference with a dropout posterior whose rate is
                   learned rather than fixed by hand.

    mc_dropout     variational inference with a dropout posterior at a FIXED
                   rate. This is the same objective as the MAP baseline, so the
                   model is trained identically to the point-estimate run and
                   the only change is that the dropout masks are left switched
                   on at test time.

-------------------------------------------------------------------------------
WHY THESE THREE
-------------------------------------------------------------------------------
All three are variational inference. The framework is held fixed and only the
approximating family q(theta) changes, so a difference in calibration is
attributable to the shape of the posterior rather than to a different inference
procedure. mc_dropout and concrete_vi differ in one thing only, whether the
dropout rate is a hyper-parameter or a learned quantity, which isolates the
question of whether hand-choosing that rate was costing anything.

-------------------------------------------------------------------------------
THE OBJECTIVE
-------------------------------------------------------------------------------
All three minimise the negative evidence lower bound,

    L(phi) = - E_q(theta)[ log p(D | theta) ]  +  KL[ q(theta) || p(theta) ]

The first term is estimated by drawing a weight sample and evaluating the usual
cross-entropy. Because cross-entropy is averaged over the minibatch rather than
summed over the dataset, the KL must be divided by the training set size to sit
on the same scale:

    loss = mean_batch_nll  +  beta * KL / N

That division is not cosmetic. Get it wrong by a factor of N and the prior
either vanishes or crushes the likelihood.

-------------------------------------------------------------------------------
KL WARM-UP
-------------------------------------------------------------------------------
beta is ramped linearly from 0 to 1 over the first few epochs. At initialisation
the posterior standard deviations are small and the KL against a wider prior is
large, so a cold start lets the KL term dominate and drives the network to the
prior before it has learned anything from the data. Ramping in gives the
likelihood a chance to establish the means first. This is a standard remedy and
is reported in the results rather than hidden, since a warm-up changes the
optimisation path even though it leaves the objective unchanged.

-------------------------------------------------------------------------------
MATCHING THE PRIOR TO THE POINT-ESTIMATE RUN
-------------------------------------------------------------------------------
The MAP baseline minimised  mean_nll + lambda * sum(theta^2)  with
lambda = 1e-4. Written as a negative log posterior over N examples that is a
zero-mean Gaussian prior with

    sigma_p^2 = 1 / (2 * N * lambda)

so the same prior is used for the variational runs. The methods being compared
therefore differ only in the posterior, not in how much regularisation they
happen to receive. See prior_variance_matching_map below.

-------------------------------------------------------------------------------
WHAT IS MEASURED
-------------------------------------------------------------------------------
For every cell, the S sampled predictions are averaged into one predictive
distribution and then summarised by uncertainty.py: accuracy, mean confidence,
expected calibration error, and the split of predictive entropy into aleatoric
and epistemic parts. The epistemic term is the one that matters. It should be
small on familiar conditions and large on the held-out sea state, and if it is
not, the method has not bought anything over a point estimate.

A diversity diagnostic is recorded alongside it. If the sampled models barely
differ, epistemic uncertainty will be near zero everywhere for a trivial reason,
and that needs to be visible rather than mistaken for a finding.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
    # everything, five seeds, as reported
    python train_bayes.py --data data/prepared_spec.npz \
        --seeds 0,1,2,3,4 --tag bayes5

    # quick check that the whole path runs before committing to it
    python train_bayes.py --data data/prepared_spec.npz \
        --epochs 3 --mc-samples 5 --tag smoke

    # one method at a time
    python train_bayes.py --methods gaussian_vi --tag gvi
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
from models import build_model, set_seed
from bayes_models import (SEI_GaussianVI, SEI_ConcreteVI, sample_predictions,
                          count_parameters)
import uncertainty as unc


METHODS = ["mc_dropout", "gaussian_vi", "concrete_vi"]

# Cells whose full sampled predictions are kept, for the figures that show how
# far individual posterior samples disagree. Keeping all S samples for all
# twelve cells would run to hundreds of megabytes for no additional insight.
SAMPLE_DETAIL_CELLS = ["test_id__controlled", "test_both__dynamic"]


# =============================================================================
# PRIOR
# =============================================================================

def prior_variance_matching_map(n_train: int, l2_coeff: float) -> float:
    """The Gaussian prior implied by the MAP run's L2 penalty.

        MAP loss (per example)  = mean_nll + lambda * sum(theta^2)
        scaled to the dataset   = sum_i nll_i + N * lambda * sum(theta^2)
        negative log prior      = sum(theta^2) / (2 * sigma_p^2)

        =>  sigma_p^2 = 1 / (2 * N * lambda)

    Using this keeps the variational runs and the point-estimate run on the same
    prior, so the comparison isolates the posterior.
    """
    return 1.0 / (2.0 * n_train * l2_coeff)


# =============================================================================
# DATA
# =============================================================================

def select_channels(X, channels):
    return X if channels is None else X[:, channels, ...]


def make_loader(X, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                    torch.from_numpy(y).long()),
                      batch_size=batch_size, shuffle=shuffle)


def combine_training_tiers(data, split, tiers, channels):
    """Stack the permitted tiers into one set. DYNAMIC is never included."""
    Xs, ys = [], []
    for tier in tiers:
        key = f"{split}__{tier}"
        if key in data:
            Xs.append(select_channels(data[key]["X"], channels))
            ys.append(data[key]["y"])
    return np.concatenate(Xs), np.concatenate(ys)


# =============================================================================
# BUILDING EACH METHOD
# =============================================================================

def build(method, n_classes, in_channels, n_train, args):
    """Return (model, loss_fn) for one method.

    loss_fn(model, logits, targets, beta) returns (total_loss, nll, penalty),
    where beta is the KL warm-up factor. Keeping the three objectives behind one
    signature means the training loop below is shared, so the methods cannot
    accidentally differ in anything except their loss.
    """
    if method == "gaussian_vi":
        prior_var = prior_variance_matching_map(n_train, args.l2_coeff)
        model = SEI_GaussianVI(n_classes, in_channels=in_channels,
                               flipout=not args.no_flipout,
                               prior_variance=prior_var,
                               posterior_rho_init=args.rho_init)

        def loss_fn(model, logits, targets, beta):
            nll = F.cross_entropy(logits, targets)
            kl = model.kl_divergence() / n_train
            return nll + beta * kl, nll, kl

    elif method == "concrete_vi":
        model = SEI_ConcreteVI(n_classes, n_train=n_train,
                               in_channels=in_channels,
                               dropout_reg_scale=args.dropout_reg_scale)

        def loss_fn(model, logits, targets, beta):
            nll = F.cross_entropy(logits, targets)
            # The Concrete Dropout regularisers already carry the 1/N scaling,
            # so no further division here.
            reg = model.regularisation()
            return nll + beta * reg, nll, reg

    elif method == "mc_dropout":
        model = build_model("spectrogram", n_classes,
                            dropout_p=args.dropout, in_channels=in_channels)

        def loss_fn(model, logits, targets, beta):
            nll = F.cross_entropy(logits, targets)
            # Identical to the MAP objective. For a fixed-rate dropout posterior
            # the KL reduces to the same L2 term, which is the reason the MAP
            # checkpoint can be reinterpreted as a variational one.
            reg = args.l2_coeff * model.L2reg()
            return nll + reg, nll, reg

    else:
        raise ValueError(f"unknown method {method}")

    return model, loss_fn


def prior_note(method, n_train, args):
    """One line describing the prior actually in force, printed and recorded."""
    if method == "gaussian_vi":
        v = prior_variance_matching_map(n_train, args.l2_coeff)
        return f"N(0, {v:.4f}) matching MAP lambda={args.l2_coeff:g}"
    if method == "concrete_vi":
        return (f"Concrete Dropout regularisers w=1/(100N), "
                f"d={args.dropout_reg_scale:g}/N, N={n_train:,}  "
                f"(entropy coefficient = {args.dropout_reg_scale:g} x "
                f"input_dim / N)")
    return f"fixed dropout p={args.dropout}, L2 lambda={args.l2_coeff:g}"


# =============================================================================
# TRAINING
# =============================================================================

def train_one(method, train_loader, val_loader, n_classes, in_channels,
              n_train, args, device, seed):
    """Train one method under one seed. Returns (model, history)."""
    set_seed(seed)
    model, loss_fn = build(method, n_classes, in_channels, n_train, args)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n{'=' * 74}")
    print(f"TRAINING: {method}   seed {seed}")
    print(f"{'=' * 74}")
    print(f"parameters : {count_parameters(model):,}")
    print(f"prior      : {prior_note(method, n_train, args)}")
    print(f"epochs     : {args.epochs}   batch {args.batch_size}   lr {args.lr}"
          f"   kl warm-up {args.kl_warmup} epochs")
    print()

    history = {"train_loss": [], "train_nll": [], "penalty": [],
               "val_loss": [], "train_acc": [], "val_acc": [], "beta": []}
    best_val, best_state = float("inf"), None
    start = time.time()

    for epoch in range(args.epochs):
        # Linear warm-up on the KL term, then held at 1. Only gaussian_vi and
        # concrete_vi use beta; mc_dropout ignores it, since its penalty is the
        # same L2 the point-estimate run applied from the first step.
        beta = 1.0 if args.kl_warmup <= 0 else min(1.0, (epoch + 1) / args.kl_warmup)

        model.train()
        run_loss = run_nll = run_pen = run_correct = run_n = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss, nll, pen = loss_fn(model, logits, yb, beta)
            loss.backward()
            optimizer.step()

            n = len(yb)
            run_loss += loss.item() * n
            run_nll += nll.item() * n
            run_pen += pen.detach().item() * n
            run_correct += (logits.argmax(1) == yb).sum().item()
            run_n += n

        train_loss = run_loss / run_n
        train_nll = run_nll / run_n
        train_pen = run_pen / run_n
        train_acc = run_correct / run_n

        # Validation uses a single weight sample for the variational models and
        # the deterministic pass for mc_dropout. This is a cheap progress signal
        # for checkpoint selection, not the reported result; the reported numbers
        # all come from the averaged S-sample predictive distribution.
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

        for k, v in [("train_loss", train_loss), ("train_nll", train_nll),
                     ("penalty", train_pen), ("val_loss", val_loss),
                     ("train_acc", train_acc), ("val_acc", val_acc),
                     ("beta", beta)]:
            history[k].append(v)

        # Selection is on the likelihood term, not the full objective. The
        # penalty differs in scale between methods, so selecting on the total
        # would not compare like with like across them.
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            line = (f"  epoch {epoch:>3}/{args.epochs}  beta {beta:.2f}  "
                    f"nll {train_nll:.4f}  pen {train_pen:.4f}  "
                    f"acc {train_acc:.4f}  |  val {val_loss:.4f} "
                    f"acc {val_acc:.4f}{'  *' if val_loss == best_val else ''}")
            # Posterior width is the diagnostic that reveals collapse in either
            # direction, and it does not show up in the loss.
            if hasattr(model, "posterior_sigma_summary"):
                s = model.posterior_sigma_summary()
                line += f"   sigma {np.mean(list(s.values())):.4f}"
            if hasattr(model, "dropout_rates"):
                line += f"   p {[round(p, 3) for p in model.dropout_rates()]}"
            print(line)

    print(f"\n  finished in {(time.time() - start) / 60:.1f} min   "
          f"best val loss {best_val:.4f}")

    model.load_state_dict(best_state)
    model = model.to(device)

    # Final state of whatever the method learned about its own uncertainty.
    if hasattr(model, "dropout_rates"):
        rates = model.dropout_rates()
        print(f"  learned dropout rates: {[round(p, 4) for p in rates]}")
        moved = max(abs(p - 0.1) for p in rates)
        if moved < 0.01:
            print("  NOTE: rates barely moved from their initial 0.1. The "
                  "regularisers are too weak to be fitting them, and the run "
                  "should be described as fixed-rate rather than learned.")
        history["dropout_rates"] = rates
    if hasattr(model, "posterior_sigma_summary"):
        sig = model.posterior_sigma_summary()
        print(f"  posterior sigma by layer: "
              f"{ {k: round(v, 4) for k, v in sig.items()} }")
        history["posterior_sigma"] = sig

    return model, history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, method, data, channels, device, args):
    """Evaluate one trained model across every test split and tier.

    Returns (per-cell summaries, mean predictive probabilities, retained sample
    detail). Each cell is predicted S times and the samples are averaged into
    one predictive distribution before any metric is computed, which is the
    Bayesian model average rather than a single network's output.
    """
    results, mean_probs, detail = {}, {}, {}
    mc = (method == "mc_dropout")

    for key in sorted(k for k in data if k.startswith("test_")):
        X = select_channels(data[key]["X"], channels)
        y = data[key]["y"]
        loader = make_loader(X, y, args.batch_size, shuffle=False)

        probs = sample_predictions(model, loader, device,
                                   n_samples=args.mc_samples, mc_dropout=mc)

        results[key] = unc.summarise(probs, y)
        mean_probs[key] = unc.bayesian_model_average(probs).astype(np.float32)
        if key in SAMPLE_DETAIL_CELLS:
            detail[key] = probs.astype(np.float32)

    return results, mean_probs, detail


def print_table(results, config, method, n_samples):
    print(f"\n{method}   (S = {n_samples} posterior samples)")
    print(f"{'split':<11}{'tier':<12}{'n':>7}{'acc':>8}{'conf':>8}{'ECE':>8}"
          f"{'epist':>8}{'alea':>8}{'disagree':>10}")
    print("-" * 80)
    held = config["held_out_tier"]
    for split in ["test_id", "test_rx", "test_day", "test_both"]:
        for tier in config["test_tiers"]:
            key = f"{split}__{tier}"
            if key not in results:
                continue
            r = results[key]
            print(f"{split:<11}{tier:<12}{r['n']:>7,}{r['accuracy']:>8.4f}"
                  f"{r['mean_confidence']:>8.4f}{r['ece']:>8.4f}"
                  f"{r['entropy_epistemic']:>8.4f}{r['entropy_aleatoric']:>8.4f}"
                  f"{r['sample_argmax_disagreement']:>10.3f}"
                  f"{'  <- held out' if tier == held else ''}")
        print()


# =============================================================================
# AGGREGATION
# =============================================================================

AGG_METRICS = ["accuracy", "mean_confidence", "ece", "entropy_total",
               "entropy_aleatoric", "entropy_epistemic",
               "epistemic_correct", "epistemic_wrong",
               "sample_prob_std", "sample_argmax_disagreement"]


def aggregate_seeds(per_seed):
    """Mean and sample standard deviation of every metric, across seeds.

    Reported because a single run confounds the effect being measured with the
    luck of the initialisation. In a project about whether stated uncertainty
    can be trusted, quoting bare point estimates would undercut the argument.
    """
    agg = {}
    for key in per_seed[0]:
        entry = {"n": per_seed[0][key]["n"]}
        for metric in AGG_METRICS:
            vals = [r[key].get(metric) for r in per_seed]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            a = np.asarray(vals, dtype=float)
            entry[metric] = float(a.mean())
            entry[f"{metric}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            entry[f"{metric}_values"] = a.tolist()
        agg[key] = entry
    return agg


def compare_methods(agg, config, baseline="mc_dropout"):
    """Differences against one method, flagged when larger than seed variation.

    The criterion is deliberately blunt: a difference counts only if it exceeds
    twice the combined seed-to-seed standard deviation. With a handful of seeds
    a formal test would claim more precision than the evidence carries, so this
    is described as a two-sigma screen in the write-up and nothing more.
    """
    others = [m for m in agg if m != baseline]
    if baseline not in agg or not others:
        return

    for method in others:
        print(f"\n{'=' * 78}")
        print(f"{method.upper()} vs {baseline.upper()}")
        print(f"{'=' * 78}")
        print(f"{'cell':<24}{'d accuracy':>24}{'d ECE':>24}")
        print("-" * 78)

        def fmt(a, b, metric):
            d = b[metric] - a[metric]
            pooled = np.sqrt(a[f"{metric}_std"] ** 2 + b[f"{metric}_std"] ** 2)
            flag = "*" if pooled > 1e-9 and abs(d) > 2 * pooled else " "
            return f"{d:+.4f} (+/-{pooled:.4f}){flag}"

        for key in sorted(agg[baseline]):
            a, b = agg[baseline][key], agg[method][key]
            print(f"{key:<24}{fmt(a, b, 'accuracy'):>24}{fmt(a, b, 'ece'):>24}")

        print("\n  * exceeds twice the combined seed variation. Unmarked rows "
              "are within\n  run-to-run noise and are not claimed as effects.")


def print_epistemic_check(agg, config):
    """Does uncertainty actually rise on conditions never trained on?

    This is the claim the whole deliverable rests on, so it gets its own table.
    A method that fails here has produced a posterior that is not tracking what
    the model does and does not know, however good its accuracy looks.

    TOTAL and EPISTEMIC are both reported, because they can move in opposite
    directions and the difference is the interesting part. Under severe channel
    corruption every sampled model tends towards a uniform prediction. They then
    AGREE that they do not know, so mutual information - the epistemic term -
    falls even as the total entropy climbs towards log K. Reporting epistemic
    uncertainty alone would make that look like a method failure when it is
    really a statement about where the uncertainty is coming from: the signal
    has been destroyed, which is aleatoric, rather than the input being
    unfamiliar to a model that could otherwise classify it.
    """
    print(f"\n{'=' * 78}")
    print("UNCERTAINTY vs DISTRIBUTION SHIFT")
    print(f"{'=' * 78}")
    print("  familiar = test_id on a trained tier, shifted = test_both on the "
          "held-out tier")
    print(f"\n{'method':<14}{'total fam':>11}{'total shift':>13}{'ratio':>8}"
          f"{'epi fam':>10}{'epi shift':>11}{'ratio':>8}{'wrong/right':>13}")
    print("-" * 78)

    held = config["held_out_tier"]
    seen = config["train_tiers"][0]
    for method, a in agg.items():
        fam = a.get(f"test_id__{seen}")
        shift = a.get(f"test_both__{held}")
        if not fam or not shift:
            continue

        def ratio(x, y):
            return y / x if x > 1e-9 else float("inf")

        t_f, t_s = fam["entropy_total"], shift["entropy_total"]
        e_f, e_s = fam["entropy_epistemic"], shift["entropy_epistemic"]

        # A method should also be more uncertain on the examples it got wrong
        # than the ones it got right, within a single cell. That is a stricter
        # test than the familiar/shifted contrast, since it cannot be passed by
        # simply being more uncertain about everything.
        w = shift.get("epistemic_wrong")
        r = shift.get("epistemic_correct")
        wr = (w / r) if (w is not None and r is not None and r > 1e-9) else float("nan")

        print(f"{method:<14}{t_f:>11.4f}{t_s:>13.4f}{ratio(t_f, t_s):>8.2f}"
              f"{e_f:>10.4f}{e_s:>11.4f}{ratio(e_f, e_s):>8.2f}{wr:>13.2f}")

    print(f"\n  log K = {np.log(config['n_classes']):.4f} is the ceiling on "
          "total entropy: complete ignorance.")
    print("  ratio > 1 means more uncertain on conditions never trained on.")
    print("  wrong/right > 1 means more uncertain on the predictions it got "
          "wrong, which is\n  the stricter test since it cannot be passed by "
          "being vague about everything.")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")

    data, config = load_prepared(args.data)
    n_classes = config["n_classes"]

    channels = None if args.channels is None else \
        [int(c) for c in args.channels.split(",")]

    X_train, y_train = combine_training_tiers(data, "train",
                                              config["train_tiers"], channels)
    X_val, y_val = combine_training_tiers(data, "val",
                                          config["train_tiers"], channels)
    n_train = len(y_train)
    in_channels = X_train.shape[1]

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 74)
    print("APPROXIMATE BAYESIAN INFERENCE FOR SEI")
    print("=" * 74)
    print(f"data          : {args.data}")
    print(f"classes       : {n_classes}")
    print(f"input shape   : {X_train.shape[1:]}")
    print(f"train / val   : {n_train:,} / {len(y_val):,}")
    print(f"held-out rx   : {config['held_out_rx']}")
    print(f"held-out day  : {config['held_out_day']}")
    print(f"held-out tier : {config['held_out_tier']}")
    print(f"methods       : {methods}")
    print(f"seeds         : {seeds}")
    print(f"MC samples    : {args.mc_samples}")
    print(f"device        : {device}")

    train_loader = make_loader(X_train, y_train, args.batch_size, True)
    val_loader = make_loader(X_val, y_val, args.batch_size, False)

    per_seed = {m: [] for m in methods}
    histories = {m: [] for m in methods}
    saved_probs, saved_detail = {}, {}

    for method in methods:
        for run_index, seed in enumerate(seeds):
            model, history = train_one(method, train_loader, val_loader,
                                       n_classes, in_channels, n_train,
                                       args, device, seed)

            results, mprobs, detail = evaluate(model, method, data, channels,
                                               device, args)
            per_seed[method].append(results)
            histories[method].append(history)

            if run_index == 0:
                # First seed supplies the probabilities used for the figures and
                # a checkpoint. The remaining seeds exist to quantify spread.
                saved_probs[method] = mprobs
                saved_detail[method] = detail
                torch.save(model.state_dict(), os.path.join("results", f"model_{method}_{args.tag}.pt"))

            print_table(results, config, f"{method}  seed {seed}",
                        args.mc_samples)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    agg = {m: aggregate_seeds(per_seed[m]) for m in methods}

    if len(seeds) > 1:
        for method in methods:
            print(f"\n{'=' * 74}")
            print(f"{method.upper()}  (mean over {len(seeds)} seeds)")
            print(f"{'=' * 74}")
            print(f"{'cell':<24}{'accuracy':>20}{'ECE':>20}{'epistemic':>14}")
            print("-" * 78)
            for key in sorted(agg[method]):
                r = agg[method][key]
                acc = f"{r['accuracy']:.4f} +/- {r['accuracy_std']:.4f}"
                ece = f"{r['ece']:.4f} +/- {r['ece_std']:.4f}"
                print(f"{key:<24}{acc:>20}{ece:>20}"
                      f"{r['entropy_epistemic']:>14.4f}")

    print_epistemic_check(agg, config)
    compare_methods(agg, config)

    # ---- save --------------------------------------------------------------
    with open(os.path.join("results", f"results_{args.tag}.json"), "w") as f:
        json.dump({"config": config, "args": vars(args), "seeds": seeds,
                   "methods": methods, "n_train": n_train,
                   "prior_notes": {m: prior_note(m, n_train, args)
                                   for m in methods},
                   "aggregated": agg, "per_seed": per_seed,
                   "histories": histories}, f, indent=2)

    arrays = {f"{m}__{k}": p
              for m, d in saved_probs.items() for k, p in d.items()}
    arrays.update({f"samples__{m}__{k}": p
                   for m, d in saved_detail.items() for k, p in d.items()})
    arrays.update({f"labels__{k}": data[k]["y"]
                   for k in data if k.startswith("test_")})
    np.savez_compressed(os.path.join("results", f"probabilities_{args.tag}.npz"), **arrays)

    print(f"\nsaved results_{args.tag}.json")
    print(f"saved probabilities_{args.tag}.npz")
    print(f"saved model_<method>_{args.tag}.pt for each method (first seed)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Approximate Bayesian inference for the SEI project.")
    p.add_argument("--data", default=os.path.join("data", "prepared_spec.npz"))
    p.add_argument("--methods", default=",".join(METHODS),
                   help="comma-separated subset of " + ", ".join(METHODS))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2-coeff", type=float, default=1e-4,
                   help="MAP prior strength; also sets the VI prior variance "
                        "so the two runs share a prior")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="fixed rate for the mc_dropout posterior")
    p.add_argument("--dropout-reg-scale", type=float, default=1.0,
                   help="multiplier on the Bernoulli entropy term of the "
                        "Concrete Dropout KL. This is the only part of the "
                        "objective resisting p -> 0, and at the default of 1 "
                        "its strength depends on each layer's input size. "
                        "Raise it to test whether rate collapse is a property "
                        "of the data or of a weak regulariser.")
    p.add_argument("--rho-init", type=float, default=-5.0,
                   help="initial rho for the Gaussian posterior; "
                        "sigma = softplus(rho)")
    p.add_argument("--kl-warmup", type=int, default=10,
                   help="epochs over which the KL weight ramps 0 -> 1. "
                        "0 disables the warm-up.")
    p.add_argument("--no-flipout", action="store_true",
                   help="use plain reparameterisation instead of flipout")
    p.add_argument("--mc-samples", type=int, default=20,
                   help="posterior samples S per test example")
    p.add_argument("--seeds", default="0")
    p.add_argument("--channels", default=None)
    p.add_argument("--tag", default="bayes1")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--cpu", action="store_true")

    main(p.parse_args())
