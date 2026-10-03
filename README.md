# Specific Emitter Identification under Maritime Channel Distortion

Joseph M. Rice, LTJG, USN
CS 4323, Naval Postgraduate School

Every radio has manufacturing imperfections that show up in the signal it
transmits. They are hard to fake, which makes them useful for identifying a
specific transmitter. The problem is that the signal also picks up distortion
from the environment it travelled through and from whatever receiver recorded
it, and a model can end up learning that instead.

This project asks whether a classifier that identifies transmitters knows when
it has stopped being able to. It does not, and the work measures by how much.

## What I found

Trained on familiar conditions, the classifier reaches 94.6% accuracy with a
calibration error of 0.007. Tested on a sea state it never saw, accuracy falls
to 17.1%, barely above the 10% chance floor for ten emitters, while it still
reports 60% confidence. The failure is silent. The model does not become
visibly confused, it stays confident and becomes wrong.

Replacing the single set of weights with an approximate posterior does not
recover the accuracy. At those signal-to-noise ratios the channel has destroyed
the fingerprint, and no inference procedure recovers information that is not
there. What it recovers is the honesty of the confidence.

| Method | Mean ECE | Mean accuracy | ECE, hardest cell | Confidence, hardest cell |
|---|---|---|---|---|
| MLE | 0.261 | 0.523 | 0.432 | 0.603 |
| MAP | 0.226 | 0.519 | 0.323 | 0.510 |
| MC dropout | 0.149 | 0.523 | 0.221 | 0.405 |
| Gaussian VI | 0.174 | 0.510 | 0.236 | 0.428 |
| Concrete VI, d=1 | 0.248 | 0.504 | 0.421 | 0.586 |
| Concrete VI, d=10 | 0.163 | 0.440 | 0.227 | 0.349 |

Accuracy barely moves. Confidence does, and that is the point. A track reported
at 60% confidence and wrong 83% of the time gets acted on. The same track at
41% gets escalated to an operator.

Two other results came out of it.

An embedding analysis separates two failure modes that accuracy alone hides. On
a held-out receiver the emitters are still cleanly separated in the network's
internal representation, 0.993 neighbour purity, while accuracy has fallen to
0.550. The fingerprint survives and the decision boundaries do not. On the
roughest sea state the representation itself is gone. The uncertainty
decomposition, computed from completely different machinery, assigns those two
cases the same way round.

Concrete Dropout, which learns its own dropout rate, collapses that rate to
near zero at its default setting and calibrates no better than plain maximum
likelihood under shift. That turned out not to be a statement about the data.
Only one term in the objective resists the collapse, the Bernoulli entropy, and
its strength scales with the number of inputs to each layer. Raising it by a
factor of ten stops the collapse and brings calibration level with fixed-rate
dropout. Raising it further pins every rate at 0.5, where the entropy is
largest, so the fitted value then describes the regulariser rather than the
data. The rate is only meaningful in a band you have to find by sweeping, which
is the hyper-parameter the method was meant to remove.

## Data

[WiSig](https://cores.ee.ucla.edu/wisig/), `ManyTx` subset. 511,515 IEEE 802.11
preamble captures from 150 physical transmitters, recorded by 18 physical USRP
receivers over four capture days. Download it separately, it is 2.5 GB and not
in this repository.

The transmitter fingerprints and receiver imperfections are real hardware. The
maritime channel is not. No public dataset has maritime captures with known
emitter labels, so I simulate the propagation and apply it to the real
recordings.

## How it fits together

```
WiSig captures -> maritime channel -> power normalisation -> STFT spectrogram
               -> CNN -> MLE / MAP / three variational methods
```

| File | What it does |
|---|---|
| `gauntlet.py` | Simulated maritime channel: two-ray fade, sea-surface scattering, Doppler, ducting, noise |
| `wisig_loader.py` | Picks a cohort of emitters and builds the held-out splits |
| `frontends.py` | Turns bursts into raw I/Q or spectrogram tensors |
| `models.py` | The deterministic CNN classifiers |
| `prepare_dataset.py` | Runs the pipeline once and freezes the output |
| `verify_data.py` | Checks the prepared data before I trust it |
| `bayes_models.py` | Gaussian and Concrete Dropout posteriors over the same network |
| `uncertainty.py` | Bayesian model averaging, aleatoric and epistemic decomposition |
| `train_bayes.py` | Trains and evaluates the three variational methods |
| `make_figures_bayes.py` | Figures and LaTeX tables |
| `train.py` | Trains MLE and MAP, evaluates accuracy and calibration |
| `make_figures.py` | Figures and tables for the point-estimate results |
| `make_embedding_figure.py` | t-SNE of the penultimate layer plus neighbour purity |

Each module tests itself when run directly:

```bash
python gauntlet.py
python wisig_loader.py      # uses mock data, no download needed
python frontends.py
python models.py
python bayes_models.py
python uncertainty.py
```

## Running it

```bash
pip install numpy torch matplotlib scikit-learn bayesian-torch
```

The CUDA build of torch matters. On a 50-series card you need cu128, since the
cu121 builds have no kernels for that architecture and fail at the first GPU
operation. I lost a couple of hours to a CPU-only install that ran everything
about six times slower without saying so.

```bash
# Put ManyTx.pkl from the WiSig download in data/, then build a cohort.
# About 5 GB of RAM, run once.
python -c "from wisig_loader import SplitSpec, build_subset; \
  build_subset('data/ManyTx.pkl', 'data/cohort10.npz', k=10, \
  spec=SplitSpec(held_out_rx='8-8', held_out_day='2021_03_23', seed=0))"

# Apply the channel and transform to spectrograms
python prepare_dataset.py --subset data/cohort10.npz --out data/prepared_spec.npz

# Check it before training on it
python verify_data.py --data data/prepared_spec.npz

# Point estimates, five seeds
python train.py --data data/prepared_spec.npz --seeds 0,1,2,3,4 --tag spec5

# Approximate Bayesian methods, five seeds. About 2 hours on an RTX 5060.
python train_bayes.py --seeds 0,1,2,3,4 --tag bayes5

# Concrete Dropout with the corrected entropy regulariser
python train_bayes.py --methods concrete_vi --seeds 0,1,2,3,4 \
  --dropout-reg-scale 10 --tag cd10_s5

# Figures and tables
python make_figures.py --results results/results_spec5.json \
                       --probs results/probabilities_spec5.npz --outdir paper
python make_figures_bayes.py --outdir paper
python make_embedding_figure.py --outdir paper
```

## Layout

```
*.py            the pipeline, top level so the scripts import each other plainly
paper/          the final paper, LaTeX source, figures and tables
docs/           the data review
data/           not in the repository; put the WiSig download here
results/        not in the repository; written by the training scripts
```

`data/` and `results/` are excluded. Together they come to roughly 850 MB of
downloads, prepared tensors, prediction dumps and checkpoints, and all of it
regenerates from the commands above.
