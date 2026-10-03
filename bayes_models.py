"""
Approximate Bayesian variants of the SEI classifier.

Joseph M. Rice, LTJG, USN
CS 4323, Naval Postgraduate School

Two variational distributions over the same convolutional architecture used for
the MLE and MAP baselines:

    SEI_GaussianVI     q(theta) = mean-field Gaussian, one mean and one
                       variance per weight, learned during training.

    SEI_ConcreteVI     q(theta) = dropout, with the dropout rate itself
                       learned rather than fixed by hand.

Standard Monte Carlo dropout needs no class of its own. Its variational
distribution is the dropout already present in the deterministic model, so the
MAP checkpoint from the point-estimation stage is reused directly and the masks
are simply left active at test time (see mc_dropout_predict).

WHY THESE TWO
-------------
Both are variational inference, so the framework is held fixed and only the
family q(theta) changes. That makes the comparison a controlled one: any
difference in calibration is attributable to the shape of the approximating
distribution rather than to a different inference procedure.

THE OBJECTIVE
-------------
All variational methods minimise the same negative ELBO:

    L(phi) = -E_q[ log p(D | theta) ]  +  KL[ q(theta) || p(theta) ]

The first term is estimated by sampling weights and evaluating the usual
cross-entropy. The second is computed in closed form for the Gaussian family,
and via the Concrete Dropout regulariser for the dropout family. Following the
course labs, the likelihood term is averaged over the minibatch and the KL term
is divided by the same count so the two are on a common scale.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# bayesian-torch supplies the reparameterised and flipout layers used in Lab 4.
#   pip install bayesian-torch
from bayesian_torch.layers.variational_layers import (
    Conv2dReparameterization, LinearReparameterization)
from bayesian_torch.layers.flipout_layers import Conv2dFlipout, LinearFlipout
from bayesian_torch.models.dnn_to_bnn import get_kl_loss


# Defaults matching the deterministic classifier so capacity is comparable.
CHANNELS = (32, 48, 96)
POOLED = (4, 2)
IN_CHANNELS = 3


# =============================================================================
# READING THE VARIATIONAL PARAMETERS OUT OF A bayesian-torch LAYER
# =============================================================================
#
# bayesian-torch stores a Gaussian per weight as a mean and a rho, with
#
#     sigma = softplus(rho) = log(1 + exp(rho))
#
# so sigma stays positive without a constraint on the optimiser. The attribute
# names differ between the convolutional and linear layers (mu_kernel versus
# mu_weight, depending on version), so both spellings are checked rather than
# assumed. Getting this wrong would silently skip layers, which is exactly the
# kind of error that produces a plausible-looking but wrong KL term.

_PARAM_GROUPS = (
    ("mu_kernel", "rho_kernel", "prior_weight_mu", "prior_weight_sigma"),
    ("mu_weight", "rho_weight", "prior_weight_mu", "prior_weight_sigma"),
    ("mu_bias",   "rho_bias",   "prior_bias_mu",   "prior_bias_sigma"),
)


def variational_groups(module):
    """Yield (mu, rho, prior_mu, prior_sigma) for every weight group in a layer."""
    for mu_name, rho_name, prior_mu_name, prior_sigma_name in _PARAM_GROUPS:
        mu = getattr(module, mu_name, None)
        rho = getattr(module, rho_name, None)
        if mu is None or rho is None:
            continue
        yield (mu, rho,
               getattr(module, prior_mu_name, 0.0),
               getattr(module, prior_sigma_name, 1.0))


def gaussian_kl(mu_q, sigma_q, mu_p, sigma_p):
    """KL[ N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ], summed over all weights.

        KL = log(sigma_p / sigma_q)
             + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2)
             - 1/2

    Summed, not averaged. The ELBO is a sum over parameters; averaging would
    silently rescale the prior by the number of weights.
    """
    sigma_p = torch.as_tensor(sigma_p, dtype=mu_q.dtype, device=mu_q.device)
    mu_p = torch.as_tensor(mu_p, dtype=mu_q.dtype, device=mu_q.device)
    return (torch.log(sigma_p / sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
            - 0.5).sum()


def fan_in_of(mu):
    """Number of inputs feeding one output unit, from the parameter shape."""
    return int(np.prod(mu.shape[1:])) if mu.dim() > 1 else int(mu.shape[0])


# =============================================================================
# MEAN-FIELD GAUSSIAN VARIATIONAL INFERENCE
# =============================================================================

class SEI_GaussianVI(nn.Module):
    """Mean-field Gaussian variational inference over the SEI classifier.

    Every weight is replaced by a Gaussian with its own mean and variance:

        q(theta_j) = N(mu_j, sigma_j^2)

    and a weight is drawn with the reparameterisation trick

        theta_j = mu_j + sigma_j * eps_j,    eps_j ~ N(0, 1)

    so gradients flow to mu and sigma. The parameter count roughly doubles,
    since each weight now carries a variance as well as a mean.

    FLIPOUT VS PLAIN REPARAMETERISATION
    A single weight sample shared across a whole minibatch makes the gradient
    estimate noisy. Flipout decorrelates the sample across examples in the batch
    using random sign flips, which lowers gradient variance at modest extra
    cost. Module 4 covers both; the course lab uses flipout for its
    classification example, so it is the default here.

    Each forward pass draws a fresh weight sample, so calling the model twice on
    the same input gives two different predictions. That is the intended
    behaviour and is what makes Bayesian model averaging possible at test time.
    """

    def __init__(self, n_classes: int, in_channels: int = IN_CHANNELS,
                 channels=CHANNELS, pooled_size=POOLED,
                 flipout: bool = True,
                 prior_mean: float = 0.0, prior_variance: float = 0.14,
                 posterior_rho_init: float = -5.0):
        """
        prior_variance defaults to 0.14, which is not an arbitrary choice. The
        MAP baseline used loss = mean_nll + 1e-4 * sum(theta^2) on 35,720
        training examples. Writing that as a negative log posterior over the
        whole dataset,

            N * lambda * sum(theta^2) = sum(theta^2) / (2 * sigma_p^2)
            =>  sigma_p^2 = 1 / (2 * N * lambda) = 1 / (2 * 35720 * 1e-4) = 0.14

        so 0.14 IS the prior the point-estimate model was already assuming. Using
        it here means the Gaussian VI run and the MAP run share a prior and differ
        only in whether the posterior is collapsed to a point. Any difference in
        calibration is then attributable to the posterior, not to a quietly
        different amount of regularisation. Pass prior_variance explicitly (see
        prior_variance_matching_map in train_bayes.py) if N or lambda change.
        """
        super(SEI_GaussianVI, self).__init__()

        Conv = Conv2dFlipout if flipout else Conv2dReparameterization
        Linear = LinearFlipout if flipout else LinearReparameterization

        c1, c2, c3 = channels

        # bayesian-torch's argument is called prior_variance, but the value is
        # written straight into its prior_weight_sigma buffer with no square
        # root taken. Its default of 1.0 hides the discrepancy, since sqrt(1)=1.
        # The standard deviation is therefore what has to be passed, or the
        # prior in force is the square of the one intended. Verified by the
        # prior-sigma check in the self-test below, which is there precisely
        # because this is silent when it goes wrong.
        prior_sigma = float(np.sqrt(prior_variance))
        prior = dict(prior_mean=prior_mean, prior_variance=prior_sigma,
                     posterior_mu_init=0.0, posterior_rho_init=posterior_rho_init)

        self.conv1 = Conv(in_channels, c1, kernel_size=3, padding=1, **prior)
        self.conv2 = Conv(c1, c2, kernel_size=3, padding=1, **prior)
        self.conv3 = Conv(c2, c3, kernel_size=3, padding=1, **prior)

        self.pool = nn.AdaptiveAvgPool2d(pooled_size)

        flat = c3 * pooled_size[0] * pooled_size[1]
        self.fc1 = Linear(flat, 128, **prior)
        self.fc2 = Linear(128, n_classes, **prior)

        self.n_classes = n_classes
        self.flipout = flipout
        self.prior_variance = prior_variance      # the true variance
        self.prior_sigma = prior_sigma            # what the layers store

        self._rescale_initial_means()

    def _rescale_initial_means(self):
        """Re-initialise the variational means with fan-in aware scaling.

        bayesian-torch draws the initial means from N(posterior_mu_init, 0.1)
        for every layer, irrespective of its size. That is roughly right for the
        first convolution but three to five times too wide for the deeper
        layers, so the untrained network produces large logits and a
        cross-entropy well above ln(K): it starts out confidently wrong rather
        than neutral.

        The target is the scale PyTorch itself uses for Conv2d and Linear, which
        is kaiming_uniform_ with a = sqrt(5). That draws uniformly from
        +/- sqrt(1/fan_in), giving a standard deviation of

            std = sqrt(1 / (3 * fan_in))

        Matching it puts the variational model on the same footing at
        initialisation as the deterministic baseline, so any difference in the
        final results reflects the inference method rather than a worse start.
        """
        with torch.no_grad():
            for module in self.modules():
                for mu, _rho, _pm, _ps in variational_groups(module):
                    if mu is getattr(module, "mu_bias", None):
                        mu.zero_()
                        continue
                    fan_in = fan_in_of(mu)
                    std = float(np.sqrt(1.0 / (3.0 * max(fan_in, 1))))
                    mu.normal_(0.0, std)

    def posterior_sigma_summary(self):
        """Mean posterior standard deviation per layer, as a training diagnostic.

        If these run away towards the prior standard deviation the KL term is
        overwhelming the likelihood and the model is collapsing to the prior. If
        they stay pinned at their tiny initial value the posterior is effectively
        a point estimate and the method is not doing anything. Either failure is
        easy to miss in the loss curve alone, so it is printed every epoch.
        """
        out = {}
        for name, module in self.named_modules():
            sigmas = [F.softplus(rho).mean().item()
                      for _mu, rho, _pm, _ps in variational_groups(module)]
            if sigmas:
                out[name] = float(np.mean(sigmas))
        return out

    def forward(self, x):
        # return_kl=False keeps the forward signature the same as the
        # deterministic model; the KL is collected separately with
        # get_kl_loss(model), following the course lab.
        x = F.relu(self.conv1(x, return_kl=False))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x, return_kl=False))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x, return_kl=False))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x, return_kl=False))
        return self.fc2(x, return_kl=False)          # logits

    def kl_divergence(self):
        """KL[q(theta) || p(theta)], summed over every weight in the network.

        Closed form for two Gaussians, per Module 4 section 4.4. This is the
        prior-regularisation term of the negative ELBO.

        WHY NOT get_kl_loss. bayesian-torch ships a helper, get_kl_loss(model),
        and the course lab uses it. Its internal kl_div takes the MEAN over the
        weights in each layer rather than the sum, so what it returns is roughly
        a per-weight average: about 45 for this network, where the true KL is
        five orders of magnitude larger. That is fine for the lab, because the
        constant gets absorbed into the choice of KL weight, but it is not the
        KL in the ELBO. Since this project reports the prior explicitly and
        compares against a MAP run with a matched prior, the honest summed
        quantity is used instead. get_kl_loss is still imported and reported
        alongside it in the self-test so the difference is visible rather than
        buried.
        """
        total = None
        for module in self.modules():
            for mu, rho, prior_mu, prior_sigma in variational_groups(module):
                sigma = F.softplus(rho)
                term = gaussian_kl(mu, sigma, prior_mu, prior_sigma)
                total = term if total is None else total + term
        return total

    def kl_divergence_library(self):
        """bayesian-torch's own KL helper, kept for comparison only."""
        return get_kl_loss(self)


# =============================================================================
# CONCRETE DROPOUT
# =============================================================================

class ConcreteDropout(nn.Module):
    """Dropout whose rate is learned rather than fixed.

    Standard dropout samples a hard Bernoulli mask, which has no gradient with
    respect to p. Concrete Dropout replaces it with a continuous relaxation:

        m = sigmoid( (log p - log(1-p) + log u - log(1-u)) / t ),  u ~ U(0,1)

    As the temperature t goes to zero this approaches a Bernoulli draw, but for
    t > 0 it is differentiable, so p can be optimised alongside the weights.

    WHY BOTHER. Module 5 notes that in ordinary Monte Carlo dropout the
    parameters controlling uncertainty, p_drop and sigma_q, are hyper-parameters
    rather than quantities learned from data. In this project the dropout rates
    were chosen as regularisation settings for the point-estimate models, with
    no reference to uncertainty at all. Learning them removes that arbitrariness.

    It does not remove hyper-parameters entirely: the two regulariser scales
    below replace the one dropout rate. The gain is that p is now fitted to the
    data and free to differ between layers.

    The regularisation term has two parts, following Gal's Concrete Dropout:

        weight term  = weight_reg * sum(w^2) / (1 - p)
        entropy term = dropout_reg * input_dim * [ p log p + (1-p) log(1-p) ]

    The first is the usual weight penalty, inflated as p grows because surviving
    weights carry more of the signal. The second is the entropy of the Bernoulli,
    which pushes p up; the balance between them determines the fitted rate.
    """

    def __init__(self, weight_regulariser: float, dropout_regulariser: float,
                 init_min: float = 0.1, init_max: float = 0.1,
                 temperature: float = 0.1):
        super().__init__()
        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser
        self.temperature = temperature

        # p is stored as a logit so it stays in (0, 1) under unconstrained
        # gradient descent.
        lo = np.log(init_min) - np.log(1.0 - init_min)
        hi = np.log(init_max) - np.log(1.0 - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(lo, hi))

        self.regularisation = 0.0

    @property
    def p(self):
        return torch.sigmoid(self.p_logit)

    def forward(self, x, layer):
        """Apply the relaxed mask to x, run `layer`, and record the regulariser."""
        p = self.p
        eps = 1e-7

        u = torch.rand_like(x)
        drop_prob = torch.sigmoid(
            (torch.log(p + eps) - torch.log(1.0 - p + eps)
             + torch.log(u + eps) - torch.log(1.0 - u + eps)) / self.temperature)

        # Keep the expected activation unchanged, as in scaled dropout.
        x = x * (1.0 - drop_prob) / (1.0 - p + eps)

        out = layer(x)

        # Number of input units this dropout acts on. For a convolutional input
        # of shape (batch, C, H, W) that is C*H*W. This is the multiplier on the
        # entropy term in Gal's formulation.
        input_dim = int(np.prod(x.shape[1:]))

        # Sum of squared weights of whichever module actually holds them.
        weight_sq = 0.0
        for m in (layer.modules() if hasattr(layer, "modules") else [layer]):
            if hasattr(m, "weight") and m.weight is not None:
                weight_sq = weight_sq + torch.sum(torch.square(m.weight))

        weights_reg = self.weight_regulariser * weight_sq / (1.0 - p + eps)
        entropy = p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps)
        dropout_reg = self.dropout_regulariser * input_dim * entropy

        self.regularisation = weights_reg + dropout_reg
        return out


class SEI_ConcreteVI(nn.Module):
    """The SEI classifier with Concrete Dropout in place of fixed-rate dropout.

    Architecture is unchanged from the deterministic model; only the dropout is
    different. The learned rates are reported in the results, since they say
    something about how much of the network the data actually constrains.
    """

    def __init__(self, n_classes: int, n_train: int,
                 in_channels: int = IN_CHANNELS, channels=CHANNELS,
                 pooled_size=POOLED, dropout_reg_scale: float = 1.0):
        """
        dropout_reg_scale multiplies the entropy term of the regulariser.

        WHY THIS IS EXPOSED. The regulariser has two parts pulling in opposite
        directions. The weight term grows as p rises, so it pushes the rate
        down. The Bernoulli entropy term,

            H(p) = p log p + (1-p) log(1-p)

        is most negative at p = 0.5 and approaches zero as p approaches 0 or 1,
        so minimising it pushes the rate UP. It is the only thing in the
        objective resisting collapse to p = 0, and its coefficient is

            dropout_regulariser * input_dim  =  scale * input_dim / n_train

        At scale 1 that coefficient depends entirely on how many inputs the
        layer happens to have, which is an accident of architecture rather than
        a statement about how much uncertainty the layer needs. For this network
        it ranges from 0.022 before the classifier to 0.172 before the second
        convolution, an eightfold spread, and in practice only the layer at the
        top of that range retained a usable rate. Raising the scale strengthens
        the term uniformly and tests whether the collapse observed at scale 1 is
        a property of the data or merely of a weak regulariser.
        """
        super(SEI_ConcreteVI, self).__init__()

        c1, c2, c3 = channels

        # Regulariser scales from the Concrete Dropout paper and the course lab:
        # both shrink as the training set grows, so the data term dominates when
        # there is plenty of data.
        w = 1.0 / (100.0 * float(n_train))
        d = dropout_reg_scale / float(n_train)

        self.dropout_reg_scale = dropout_reg_scale

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(pooled_size)

        flat = c3 * pooled_size[0] * pooled_size[1]
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.cd1 = ConcreteDropout(w, d)
        self.cd2 = ConcreteDropout(w, d)
        self.cd3 = ConcreteDropout(w, d)
        self.cd4 = ConcreteDropout(w, d)

        self.relu = nn.ReLU()
        self.n_classes = n_classes

    def forward(self, x):
        x = self.cd1(x, nn.Sequential(self.conv1, self.relu))
        x = F.max_pool2d(x, 2)

        x = self.cd2(x, nn.Sequential(self.conv2, self.relu))
        x = F.max_pool2d(x, 2)

        x = self.cd3(x, nn.Sequential(self.conv3, self.relu))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.cd4(x, nn.Sequential(self.fc1, self.relu))
        return self.fc2(x)

    def regularisation(self):
        """Sum of the Concrete Dropout regularisers; the KL term of the ELBO."""
        return (self.cd1.regularisation + self.cd2.regularisation
                + self.cd3.regularisation + self.cd4.regularisation)

    def dropout_rates(self):
        """The learned rates, for reporting."""
        return [float(cd.p.detach().cpu()) for cd in
                (self.cd1, self.cd2, self.cd3, self.cd4)]


# =============================================================================
# PREDICTION BY BAYESIAN MODEL AVERAGING
# =============================================================================

@torch.no_grad()
def sample_predictions(model, loader, device, n_samples: int = 20,
                       mc_dropout: bool = False) -> np.ndarray:
    """Run the model n_samples times and return every sampled prediction.

    Returns (n_samples, n_examples, n_classes) of softmax probabilities, which
    is what uncertainty.py expects.

    mc_dropout=True forces train mode so dropout masks keep being sampled at
    test time. That single flag is the whole of Monte Carlo dropout: without it
    the masks are replaced by their expectation and every sample is identical.
    The variational models sample their weights regardless of mode, so they are
    left in eval mode.
    """
    model.train() if mc_dropout else model.eval()

    samples = []
    for _ in range(n_samples):
        batch_probs = []
        for xb, _ in loader:
            logits = model(xb.to(device))
            batch_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        samples.append(np.concatenate(batch_probs))

    model.eval()
    return np.stack(samples)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import math

    print("=" * 66)
    print("BAYESIAN MODELS - SELF TEST")
    print("=" * 66)

    K, BATCH = 10, 4
    x = torch.randn(BATCH, 3, 64, 13)

    print("\n[1] Forward pass shapes")
    for name, model in [("GaussianVI (flipout)", SEI_GaussianVI(K)),
                        ("GaussianVI (reparam)", SEI_GaussianVI(K, flipout=False)),
                        ("ConcreteVI", SEI_ConcreteVI(K, n_train=35720))]:
        out = model(x)
        ok = out.shape == (BATCH, K)
        print(f"    {name:<22} -> {tuple(out.shape)}   {'PASS' if ok else 'FAIL'}")

    print("\n[2] Parameter counts (deterministic baseline was 156,058)")
    for name, model in [("GaussianVI", SEI_GaussianVI(K)),
                        ("ConcreteVI", SEI_ConcreteVI(K, n_train=35720))]:
        n = count_parameters(model)
        print(f"    {name:<12} {n:>9,}  ({n/156058:.2f}x baseline)")
    print("    GaussianVI should be ~2x: each weight carries a mean and a variance")

    print("\n[3] Every forward pass samples new weights")
    m = SEI_GaussianVI(K)
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
    differs = not torch.allclose(a, b)
    print(f"    two calls in eval mode differ: {differs}   "
          f"{'PASS' if differs else 'FAIL - not sampling'}")

    print("\n[4] KL divergence")
    # Every variational parameter in the model must be accounted for. If the
    # attribute names were guessed wrong, some layers would be skipped and the
    # KL would look plausible while being wrong.
    n_var = sum(mu.numel() + rho.numel()
                for mod in m.modules()
                for mu, rho, _pm, _ps in variational_groups(mod))
    n_total = count_parameters(m)
    covered = n_var == n_total
    print(f"    variational params found {n_var:,} of {n_total:,}   "
          f"{'PASS' if covered else 'FAIL - some layers skipped'}")

    # The prior standard deviations must be the ones actually requested. If the
    # buffer names were guessed wrong the lookup falls back to 1.0 and the KL
    # would be computed against a prior nobody asked for.
    want = float(np.sqrt(m.prior_variance))
    got = [float(torch.as_tensor(ps).mean())
           for mod in m.modules() for _mu, _rho, _pm, ps in variational_groups(mod)]
    ok_prior = len(got) > 0 and all(abs(g - want) < 1e-5 for g in got)
    print(f"    prior sigma found {got[0]:.4f} (want {want:.4f}) across "
          f"{len(got)} groups   {'PASS' if ok_prior else 'FAIL - prior lookup'}")

    kl = m.kl_divergence()
    finite = torch.isfinite(kl) and kl.item() > 0
    print(f"    summed KL = {kl.item():,.1f}   "
          f"{'PASS' if finite else 'FAIL'}")

    # Hand check on one layer. With mu ~ 0 and sigma_q = softplus(-5) = 0.00672
    # against a prior sigma_p = sqrt(0.14) = 0.374, each weight contributes
    # about log(0.374/0.00672) - 0.5 = 3.52 nats.
    per_weight = kl.item() / (n_total / 2)
    print(f"    per weight  = {per_weight:.2f} nats   expected ~3.5   "
          f"{'PASS' if 2.5 < per_weight < 4.5 else 'CHECK'}")
    print(f"    library get_kl_loss = {m.kl_divergence_library().item():.2f}"
          f"   (a per-weight mean, not the sum - see kl_divergence docstring)")

    print("\n[5] Untrained loss near ln(K)")
    m = SEI_GaussianVI(K)
    m.eval()
    with torch.no_grad():
        logits = m(torch.randn(512, 3, 64, 13))
        loss = F.cross_entropy(logits, torch.randint(0, K, (512,))).item()
    print(f"    measured {loss:.4f}   expected {math.log(K):.4f}   "
          f"{'PASS' if abs(loss - math.log(K)) < 0.3 else 'CHECK'}")

    print("\n[6] Concrete Dropout learns its rate")
    cm = SEI_ConcreteVI(K, n_train=35720)
    print(f"    initial rates {[round(p,4) for p in cm.dropout_rates()]}")
    opt = torch.optim.Adam(cm.parameters(), lr=1e-2)
    for _ in range(20):
        out = cm(x)
        loss = F.cross_entropy(out, torch.randint(0, K, (BATCH,))) + cm.regularisation()
        opt.zero_grad(); loss.backward(); opt.step()
    rates = cm.dropout_rates()
    print(f"    after 20 steps {[round(p,4) for p in rates]}")
    moved = any(abs(p - 0.1) > 1e-4 for p in rates)
    sane = all(0.0 < p < 1.0 for p in rates)
    print(f"    rates moved from init: {moved}   in (0,1): {sane}   "
          f"{'PASS' if moved and sane else 'FAIL'}")

    print("\n[7] MC dropout sampling flag")
    from models import SEI_CNN2D
    det = SEI_CNN2D(K)
    det.eval()
    with torch.no_grad():
        same = torch.allclose(det(x), det(x))
    det.train()
    with torch.no_grad():
        diff = not torch.allclose(det(x), det(x))
    print(f"    eval mode  -> identical predictions: {same}")
    print(f"    train mode -> differing predictions: {diff}")
    print(f"    {'PASS' if same and diff else 'FAIL'}")

    print("\n" + "=" * 66)
