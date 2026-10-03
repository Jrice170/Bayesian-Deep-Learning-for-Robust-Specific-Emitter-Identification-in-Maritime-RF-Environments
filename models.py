"""
models.py
===============================================================================
Convolutional network definitions for the SEI capstone project.

CS 4323 - Bayesian Methods for Neural Networks
Joseph M. Rice, LTJG, USN - Naval Postgraduate School

-------------------------------------------------------------------------------
WHAT THIS MODULE DOES
-------------------------------------------------------------------------------
Defines the classifier that maps one received burst to one of K transmitters.

Two variants are provided, matching the two front ends in frontends.py:

    SEI_CNN1D    consumes raw I/Q          input (batch, 2, 256)
    SEI_CNN2D    consumes a spectrogram    input (batch, 3, freq, time)
                 (log magnitude, cos(phase), sin(phase))

They share the same overall shape - three convolutional blocks, adaptive
pooling, then two fully connected layers - so that a difference in results can
be attributed to the input representation rather than to a difference in model
capacity.

-------------------------------------------------------------------------------
WHY A NEURAL NETWORK AND NOT A LINEAR MODEL
-------------------------------------------------------------------------------
The hardware effects that identify a transmitter are nonlinear. Power-amplifier
AM/AM and AM/PM distortion bend the signal as a function of its own amplitude;
phase noise multiplies it; clock jitter warps its timing. No weighted sum of the
input samples can undo those, so a linear classifier cannot separate the
transmitters. This is the justification required by the deliverable, and it is
also why the posterior over weights will later be intractable, which in turn is
why Deliverable III needs APPROXIMATE Bayesian methods.

-------------------------------------------------------------------------------
HOW MLE AND MAP ARE IMPLEMENTED  (follows the Lab 2 pattern)
-------------------------------------------------------------------------------
Both models expose an `L2reg()` method that returns the sum of squared weights,
so training code can write the MAP objective explicitly:

    MLE:   loss = cross_entropy(logits, y)
    MAP:   loss = cross_entropy(logits, y) + l2_coeff * model.L2reg()

This mirrors Lab 2. It is also the CORRECT way to do MAP with the Adam
optimizer, and not merely a stylistic choice. PyTorch's `weight_decay` argument
adds the penalty gradient directly to the gradient used by the adaptive update
rule, which is NOT equivalent to placing an L2 penalty in the loss when the
optimizer rescales gradients per parameter. (That discrepancy is exactly why
AdamW exists as a separate optimizer.) Writing the penalty into the loss keeps
the implementation faithful to the MAP objective

    theta_MAP = argmin [ -log p(D | theta) - log p(theta) ]

where a zero-mean Gaussian prior on the weights contributes the L2 term.

-------------------------------------------------------------------------------
WHY DROPOUT IS HERE FROM THE START
-------------------------------------------------------------------------------
Dropout serves two different purposes in this project, and both need it built
into the architecture now:

    Deliverable II   ordinary regularisation during training.
    Deliverable III  the variational distribution for Monte Carlo dropout.
                     At test time we leave dropout ON and run the network
                     several times; the Bernoulli masks ARE the samples from
                     the approximate posterior over weights.

Retro-fitting dropout later would change the architecture between deliverables
and break the comparison, so it goes in at the beginning.

-------------------------------------------------------------------------------
WHY THE MODEL RETURNS LOGITS, NOT PROBABILITIES
-------------------------------------------------------------------------------
Lab 2 dealt with two-class problems and applied a sigmoid inside `forward`. This
project is multiclass, and the convention there differs: `forward` returns the
raw K logits, and the softmax is applied inside `torch.nn.CrossEntropyLoss`.

The reason is numerical. CrossEntropyLoss combines log-softmax and the negative
log-likelihood into one operation that is stable even when a logit is large;
computing softmax first and taking its log afterwards can overflow. When actual
probabilities are needed (for the calibration analysis) call `predict_proba()`,
which applies the softmax explicitly.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1-D CNN: raw I/Q front end
# =============================================================================

class SEI_CNN1D(nn.Module):
    """
    Classifier over raw I/Q bursts.

    Input   (batch, 2, 256)     channel 0 = in-phase, channel 1 = quadrature
    Output  (batch, n_classes)  raw logits

    SHAPE WALKTHROUGH (with the default settings)

        input                          (batch,   2, 256)
        conv1 + relu                   (batch,  32, 256)   kernel 7, padding 3
        max pool 2                     (batch,  32, 128)
        conv2 + relu                   (batch,  64, 128)   kernel 5, padding 2
        max pool 2                     (batch,  64,  64)
        conv3 + relu                   (batch, 128,  64)   kernel 3, padding 1
        adaptive avg pool to 8         (batch, 128,   8)
        flatten                        (batch, 1024)
        fc1 + relu + dropout           (batch, 128)
        fc2                            (batch, n_classes)

    The adaptive pooling layer is deliberate: it fixes the flattened size at
    128 x 8 regardless of the input length, so the same class still works if the
    burst length ever changes.

    Kernel sizes shrink with depth (7, 5, 3). Early layers look at wider spans of
    the waveform to pick up things like rise-time and envelope shape; later
    layers refine finer detail.
    """

    def __init__(self, n_classes: int, dropout_p: float = 0.3,
                 channels=(32, 64, 128), pooled_length: int = 8):
        super(SEI_CNN1D, self).__init__()

        c1, c2, c3 = channels

        self.conv1 = nn.Conv1d(2, c1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(pooled_length)

        self.fc1 = nn.Linear(c3 * pooled_length, 128)
        self.fc2 = nn.Linear(128, n_classes)

        # Two dropout layers: one between conv blocks, one before the classifier.
        # Using a lower rate in the convolutional stack is standard - conv layers
        # have far fewer parameters than the fully connected layers, so they need
        # less regularisation and suffer more from aggressive dropout.
        self.drop_conv = nn.Dropout(dropout_p * 0.5)
        self.drop_fc = nn.Dropout(dropout_p)

        self.n_classes = n_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = self.drop_conv(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.drop_conv(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)                 # logits, not probabilities

    def L2reg(self):
        """
        Sum of squared parameters, for the MAP objective.

        Follows the Lab 2 pattern. Corresponds to a zero-mean Gaussian prior on
        every weight: -log p(theta) is proportional to sum(theta^2), so adding
        l2_coeff * L2reg() to the negative log-likelihood gives the MAP loss.
        """
        l2reg_sum = 0.0
        for param in self.parameters():
            l2reg_sum = l2reg_sum + torch.sum(torch.square(param))
        return l2reg_sum

    def predict_proba(self, x):
        """Class probabilities. Use this for calibration analysis, not training."""
        return F.softmax(self.forward(x), dim=1)


# =============================================================================
# 2-D CNN: spectrogram front end
# =============================================================================

class SEI_CNN2D(nn.Module):
    """
    Classifier over magnitude/phase spectrograms.

    Input   (batch, 3, n_freq, n_time)
            channel 0 = log magnitude, 1 = cos(phase), 2 = sin(phase)
    Output  (batch, n_classes)           raw logits

    The phase is supplied as a cosine/sine pair rather than as a raw angle,
    because the angle returned by an FFT wraps discontinuously at +/- pi and a
    convolutional filter would read those wraps as enormous local gradients.
    See the frontends.py docstring for the full argument.

    SHAPE WALKTHROUGH (for the default 64 x 13 spectrogram)

        input                          (batch,   3, 64, 13)
        conv1 + relu                   (batch,  32, 64, 13)   kernel 3, padding 1
        max pool 2x2                   (batch,  32, 32,  6)
        conv2 + relu                   (batch,  64, 32,  6)
        max pool 2x2                   (batch,  64, 16,  3)
        conv3 + relu                   (batch, 128, 16,  3)
        adaptive avg pool to 4x2       (batch, 128,  4,  2)
        flatten                        (batch, 1024)
        fc1 + relu + dropout           (batch, 128)
        fc2                            (batch, n_classes)

    NOTE ON CHANNEL WIDTHS. The 2-D model uses narrower channels (32, 48, 96)
    than the 1-D model (32, 64, 128). This is deliberate. A 2-D kernel holds
    k x k = 9 weights per input/output channel pair where a 1-D kernel of the
    same nominal size holds only k = 3, so identical channel widths would give
    the 2-D model roughly a third more parameters. Since the whole point of
    running both is to compare INPUT REPRESENTATIONS, the models must not differ
    meaningfully in capacity, or a win for the spectrogram could simply be a win
    for having more parameters. With these widths the two come within about 8%
    of each other; both counts are reported in the paper's architecture table.

    The adaptive pool matters more here than in the 1-D case, because the
    spectrogram's dimensions change whenever the STFT window is retuned - and
    window length is a hyperparameter we intend to tune. Adaptive pooling means
    the classifier head does not have to be resized every time.
    """

    def __init__(self, n_classes: int, dropout_p: float = 0.3,
                 channels=(32, 48, 96), pooled_size=(4, 2),
                 in_channels: int = 3):
        super(SEI_CNN2D, self).__init__()

        c1, c2, c3 = channels

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(pooled_size)

        flat = c3 * pooled_size[0] * pooled_size[1]
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.drop_conv = nn.Dropout(dropout_p * 0.5)
        self.drop_fc = nn.Dropout(dropout_p)

        self.n_classes = n_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.drop_conv(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.drop_conv(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)

    def L2reg(self):
        """Sum of squared parameters, for the MAP objective. See SEI_CNN1D."""
        l2reg_sum = 0.0
        for param in self.parameters():
            l2reg_sum = l2reg_sum + torch.sum(torch.square(param))
        return l2reg_sum

    def predict_proba(self, x):
        """Class probabilities. Use this for calibration analysis, not training."""
        return F.softmax(self.forward(x), dim=1)


# =============================================================================
# HELPERS
# =============================================================================

def build_model(front_end: str, n_classes: int, **kwargs):
    """
    Construct whichever model matches the front end, so training scripts do not
    have to branch on representation.

        model = build_model("raw_iq", n_classes=10)
        model = build_model("spectrogram", n_classes=10)
    """
    if front_end == "raw_iq":
        return SEI_CNN1D(n_classes, **kwargs)
    if front_end == "spectrogram":
        return SEI_CNN2D(n_classes, **kwargs)
    raise ValueError(f"unknown front end {front_end!r}")


def count_parameters(model) -> int:
    """Number of trainable parameters. Reported in the paper's architecture table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 0):
    """
    Fix every random number generator that affects training.

    Weight initialisation is the only randomness present before data is seen, so
    pinning it is what makes the MLE and MAP runs a fair comparison: both start
    from the identical set of random weights and differ only in the loss.
    """
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import math

    print("=" * 70)
    print("MODELS - SELF TEST")
    print("=" * 70)

    K = 10
    BATCH = 4

    # --- 1. Shapes ------------------------------------------------------------
    print("\n[1] Forward pass shapes")
    set_seed(0)

    m1 = SEI_CNN1D(n_classes=K)
    x1 = torch.randn(BATCH, 2, 256)
    out1 = m1(x1)
    ok1 = out1.shape == (BATCH, K)
    print(f"    CNN1D  {tuple(x1.shape)} -> {tuple(out1.shape)}   "
          f"{'PASS' if ok1 else 'FAIL'}")

    m2 = SEI_CNN2D(n_classes=K)
    x2 = torch.randn(BATCH, 3, 64, 13)
    out2 = m2(x2)
    ok2 = out2.shape == (BATCH, K)
    print(f"    CNN2D  {tuple(x2.shape)} -> {tuple(out2.shape)}   "
          f"{'PASS' if ok2 else 'FAIL'}")

    # --- 2. Parameter counts --------------------------------------------------
    # The two models must have comparable capacity, otherwise a difference in
    # results could be caused by model size rather than by input representation.
    # 2-D kernels hold k*k weights against a 1-D kernel's k, so the 2-D model
    # uses narrower channels to compensate.
    print("\n[2] Parameter counts (reported in the architecture table)")
    n1, n2 = count_parameters(m1), count_parameters(m2)
    print(f"    CNN1D  {n1:,}   channels (32, 64, 128)")
    print(f"    CNN2D  {n2:,}   channels (32, 48,  96), 3 input channels")
    ratio = max(n1, n2) / min(n1, n2)
    print(f"    ratio  {ratio:.3f}x  "
          f"{'PASS - comparable capacity' if ratio < 1.15 else 'CHECK - unfair comparison'}")

    # --- 3. Untrained loss should equal ln(K) ---------------------------------
    # An untrained network has random weights, so it should assign roughly equal
    # probability to every class. Cross-entropy is then -log(1/K) = ln(K).
    # If this number is far off, something is wrong before training even starts.
    print("\n[3] Untrained loss sanity check")
    set_seed(0)
    m = SEI_CNN1D(n_classes=K)
    m.eval()
    with torch.no_grad():
        logits = m(torch.randn(512, 2, 256))
        y = torch.randint(0, K, (512,))
        loss = F.cross_entropy(logits, y).item()
    expected = math.log(K)
    print(f"    measured {loss:.4f}   expected ln({K}) = {expected:.4f}")
    print(f"    difference {abs(loss - expected):.4f}   "
          f"{'PASS' if abs(loss - expected) < 0.15 else 'CHECK'}")

    # --- 4. L2reg behaves like a penalty --------------------------------------
    print("\n[4] L2reg()")
    v = m1.L2reg()
    print(f"    returns a scalar tensor : {v.dim() == 0}")
    print(f"    positive                : {v.item() > 0}")
    print(f"    value                   : {v.item():.2f}")
    # Doubling every weight should quadruple the sum of squares.
    with torch.no_grad():
        for p in m1.parameters():
            p.mul_(2.0)
    v2 = m1.L2reg()
    factor = v2.item() / v.item()
    print(f"    doubling all weights scales L2reg by {factor:.2f} (want 4.00)   "
          f"{'PASS' if abs(factor - 4.0) < 0.01 else 'FAIL'}")

    # --- 5. Dropout is active in train mode, inactive in eval mode ------------
    # This matters twice: it must be OFF for honest evaluation in Deliverable II,
    # and it must be turned back ON deliberately for MC dropout in Deliverable III.
    print("\n[5] Dropout behaviour")
    set_seed(0)
    m = SEI_CNN1D(n_classes=K, dropout_p=0.5)
    x = torch.randn(16, 2, 256)

    m.train()
    a, b = m(x), m(x)
    varies_in_train = not torch.allclose(a, b)

    m.eval()
    c, d = m(x), m(x)
    fixed_in_eval = torch.allclose(c, d)

    print(f"    train mode: repeated calls differ  {varies_in_train}   "
          f"{'PASS' if varies_in_train else 'FAIL'}")
    print(f"    eval  mode: repeated calls match   {fixed_in_eval}   "
          f"{'PASS' if fixed_in_eval else 'FAIL'}")
    print(f"    (Deliverable III will re-enable train mode at test time on purpose)")

    # --- 6. Gradients flow ----------------------------------------------------
    print("\n[6] Backward pass reaches every parameter")
    set_seed(0)
    m = SEI_CNN1D(n_classes=K)
    logits = m(torch.randn(8, 2, 256))
    loss = F.cross_entropy(logits, torch.randint(0, K, (8,)))
    loss = loss + 1e-5 * m.L2reg()
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    print(f"    parameters without a gradient: {len(missing)}   "
          f"{'PASS' if not missing else 'FAIL ' + str(missing)}")

    # --- 7. Adaptive pooling tolerates a different spectrogram size -----------
    print("\n[7] CNN2D accepts a retuned STFT window")
    for (f, t) in [(64, 13), (128, 5), (32, 29)]:
        try:
            out = SEI_CNN2D(n_classes=K)(torch.randn(2, 3, f, t))
            print(f"    spectrogram {f:>3} x {t:<3} -> {tuple(out.shape)}   PASS")
        except Exception as e:
            print(f"    spectrogram {f:>3} x {t:<3} -> FAILED: {e}")

    print("\n" + "=" * 70)
    print("Self test complete.")
    print("=" * 70)
