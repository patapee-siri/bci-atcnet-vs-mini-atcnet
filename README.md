# ATCNet vs. Mini-ATCNet: An Empirical Accuracy-Efficiency Trade-off Study for Motor Imagery EEG Classification

## Abstract

Deployable brain-computer interface (BCI) systems for real-time motor imagery decoding face a
practical constraint that is rarely quantified alongside accuracy: the computational budget
available near the user is often far smaller than the GPU workstation used during model
development. This work presents a controlled empirical comparison between ATCNet (Altaheri,
Muhammad, and Alsulaiman, 2023), a published attention-based architecture for EEG motor imagery
classification, and Mini-ATCNet, a lightweight variant obtained by systematically reducing
ATCNet's own hyperparameters rather than introducing a new architecture family. Both
configurations are evaluated under an identical, leakage-free, subject-dependent five-fold
cross-validation protocol on the BCI Competition IV-2a dataset (nine subjects, four-class motor
imagery). Across forty-five subject-fold runs per configuration, ATCNet achieves a mean accuracy of
66.8 percent using 115,172 trainable parameters, while Mini-ATCNet achieves 54.9 percent using only
8,024 parameters (7.0 percent of the full model), training 44 percent faster per fold and running
inference 44 percent faster per trial. Normalized by parameter count, Mini-ATCNet is approximately
twelve times more efficient in accuracy delivered per million parameters. These results are
presented as a quantified trade-off rather than a claim that either configuration is unconditionally
superior, and the per-subject and per-class breakdowns further show that the accuracy gap between
the two configurations is not uniform, but concentrated in subjects and classes that are already
harder to classify under the full model.

## 1. Introduction

Motor imagery electroencephalography (EEG) classification is a foundational task in
non-invasive BCI research, with applications ranging from assistive control for users with motor
impairments to neurorehabilitation. A growing subset of these applications, particularly assistive
control loops that must respond to a user's intent in real time, cannot assume access to a full GPU
workstation at inference time. Model size and inference latency therefore become first-class design
constraints alongside classification accuracy, yet much of the published literature on EEG deep
learning architectures reports accuracy in isolation, leaving the accuracy that a practitioner would
have to sacrifice to meet a given latency or memory budget unspecified.

This project investigates that trade-off directly using ATCNet as a base architecture: a real,
published, and independently verifiable model rather than a hypothetical baseline. Rather than
proposing a new architecture, this work asks a narrower and more directly answerable question: if
every major hyperparameter of an existing, strong architecture is deliberately reduced for a
lightweight deployment target, what fraction of its accuracy is retained, and what is gained in
return along the axes that actually matter for deployment (training time and inference latency)?

The contributions of this repository are threefold. First, ATCNet is reimplemented directly from
its official source code and evaluated under a rigorously leakage-free protocol, distinct from the
sliding-window evaluation used in some prior EEG classification work on this dataset, where
overlapping windows from the same trial can appear in both the training and test partitions.
Second, a lightweight variant, Mini-ATCNet, is constructed with every hyperparameter reduction made
explicit and justified, rather than left as an unspecified "smaller" model. Third, accuracy,
training time, and inference time are measured jointly under one controlled protocol, allowing the
trade-off between them to be reported as a single, quantified statement rather than three
disconnected numbers.

## 2. Related Work

**ATCNet.** Altaheri, Muhammad, and Alsulaiman (2023) introduced ATCNet for EEG motor imagery
classification, combining a convolutional feature extraction block in the style of EEGNet
(temporal convolution followed by depthwise spatial convolution and a separable convolution), a
sliding-window multi-head self-attention block, and a temporal convolutional network (TCN) composed
of dilated causal residual convolutions, with predictions fused by averaging the per-window
classification logits. The original work reports 81.10 percent accuracy on BCI Competition IV-2a
with 113,732 parameters under its own training and evaluation protocol. ATCNet serves as the sole
base architecture in this study; its official implementation was consulted directly during
reimplementation.

**EEGNet and related compact convolutional architectures.** ATCNet's convolutional block inherits
its depthwise and separable convolution design from EEGNet-style architectures, which established
that compact, structured convolutional networks can be competitive for EEG decoding without the
parameter counts typical of vision-scale convolutional networks. This lineage motivates the premise
of this study: that a further reduction in an already compact architecture's width and depth is a
reasonable place to look for a deployment-oriented lightweight variant.

**Multi-task and metric-learning approaches to motor imagery classification.** A separate line of
work on this dataset, including MIN2Net and MixNet from the same research group, addresses the
related but distinct problem of subject-independent generalization through multi-task learning and
deep metric learning objectives layered on top of a discriminative backbone. That direction is
complementary to, but outside the scope of, this repository, which focuses specifically on the
accuracy-efficiency trade-off of a purely discriminative architecture under a subject-dependent
protocol. Subject-independent evaluation of the two configurations studied here is identified as
future work in Section 8.

**Model compression.** Structured hyperparameter reduction, as used here, is a coarse but
transparent alternative to more sophisticated compression techniques such as knowledge distillation
or pruning. It was chosen for this study specifically because every reduction can be stated and
justified individually (Section 4.3), which keeps the resulting comparison interpretable rather than
dependent on a separate compression algorithm's own behavior.

## 3. Methodology

### 3.1 Dataset

All experiments use the BCI Competition IV-2a dataset (Tangermann et al., 2012), comprising EEG
recordings from nine subjects performing four classes of motor imagery: left hand, right hand,
foot, and tongue. Each subject contributed 288 trials in the labeled training session, recorded from
22 EEG channels at 250 Hz. Only the labeled session ("T") is used, since the evaluation session
("E") in this dataset does not carry ground-truth labels usable for the subject-dependent protocol
adopted here.

### 3.2 Preprocessing and Epoching

Each trial is extracted as a single, non-overlapping epoch spanning the four-second motor imagery
cue window (0 to 4 seconds relative to cue onset), band-pass filtered between 4 and 40 Hz. This
trial-level epoching is a deliberate methodological choice: sliding-window epoching, in which
multiple overlapping windows are drawn from a single trial and later split across training and test
sets, allows near-duplicate samples from the same trial to appear on both sides of the split,
inflating the reported accuracy. Every trial in this study contributes exactly one sample to exactly
one partition. No common spatial pattern (CSP) filtering or other hand-crafted spatial feature
extraction is applied; both architectures under comparison learn spatial filtering directly from
the raw channel data through their own depthwise convolutional layers.

### 3.3 Model Architecture

Both configurations share the same three-block architecture, reimplemented from ATCNet's official
`models.py` and `attention_models.py`:

1. **Convolutional block.** A temporal convolution, followed by a depthwise spatial convolution
   across all EEG channels, followed by a separable convolution, each with batch normalization,
   an exponential linear unit (ELU) activation, average pooling, and dropout.
2. **Sliding-window attention block.** The convolutional block's output is divided into several
   overlapping temporal windows. Each window is independently processed by a multi-head
   self-attention block (pre-normalization, multi-head attention, dropout, residual connection).
3. **Temporal convolutional block.** Each attended window is passed through a stack of dilated
   causal one-dimensional convolutions with residual connections, and the final per-class logits
   from every window are fused by simple averaging.

`max_norm` weight constraints and L2 regularization are applied throughout, matching the official
implementation.

### 3.4 The Lightweight Variant: Mini-ATCNet

Mini-ATCNet is not an architecture from the published literature; it is a self-designed lightweight
variant produced by reducing seven of ATCNet's own hyperparameters, with the block structure
described in Section 3.3 left entirely unchanged. Every reduction is listed explicitly in Table 1.

**Table 1.** Hyperparameter differences between ATCNet_full and Mini-ATCNet.

| Hyperparameter | ATCNet_full | Mini-ATCNet | Rationale |
|---|---|---|---|
| `eegn_F1` (temporal filters) | 16 | 8 | Halves the feature width of every downstream layer, since `F2 = F1 x D` propagates through the rest of the network |
| `eegn_D` (depthwise multiplier) | 2 | 1 | Avoids doubling the channel count after the depthwise spatial convolution |
| `n_windows` (attention windows) | 5 | 3 | Each window owns an independent temporal convolutional block and dense classification head, making this the single largest parameter-count lever in the architecture |
| `tcn_filters` | 32 | 16 | Matches the halved feature width from the smaller convolutional block |
| `tcn_depth` | 2 | 1 | A single dilated residual block rather than two |
| Attention heads (`num_heads`) | 2 | 1 | A single attention head over an already reduced feature dimension |
| Attention key dimension (`key_dim`) | 8 | 4 | Matches the halved feature width |

No hyperparameter outside this table differs between the two configurations. The resulting parameter
counts, measured directly from the instantiated models, are 115,172 for ATCNet_full and 8,024 for
Mini-ATCNet (7.0 percent of the full model).

### 3.5 Training Protocol

Each configuration is evaluated independently for each of the nine subjects using subject-dependent,
stratified five-fold cross-validation, yielding forty-five subject-fold runs per configuration.
Within each fold, the training partition is further split (85/15) into an inner training set and a
validation set used for early stopping and learning-rate scheduling.

Two additions to the training procedure were introduced after an initial run showed every fold
consuming its full epoch budget without early stopping ever triggering, indicating the optimization
was not yet using its available training time effectively:

- **Data augmentation.** Jitter, amplitude scaling, and magnitude warping are applied to the inner
  training partition only, after the train and validation split and before feature standardization,
  so that augmented samples can never leak into validation or test data.
- **Learning-rate warm-up.** The learning rate is linearly increased from zero to its target value
  over the first ten epochs before a standard reduce-on-plateau schedule takes over.

Both additions were confirmed, by direct comparison against the pre-augmentation run, to improve
mean accuracy for both configurations without any change to either architecture. Early stopping
monitors validation loss rather than validation accuracy, since a model that collapses to predicting
a single class for every trial produces a flat validation accuracy curve that would otherwise be
misread by early stopping as convergence.

### 3.6 Evaluation Metrics

For each of the forty-five subject-fold runs per configuration, the following are recorded: overall
accuracy, macro-averaged F1 score, Cohen's kappa, wall-clock training time, wall-clock inference
time per trial, the number of epochs actually run, and whether the fold's predictions collapsed to a
single dominant class (defined as more than 90 percent of predictions falling into one class).
Results are aggregated as the mean and standard deviation across all forty-five runs per
configuration, and separately broken down per subject and per class.

## 4. Results

**Table 2.** Aggregate results across forty-five subject-fold runs per configuration.

| Metric | ATCNet_full | Mini-ATCNet |
|---|---|---|
| Trainable parameters | 115,172 | 8,024 (7.0% of full) |
| Mean accuracy | 66.8% (SD 16.6%) | 54.9% (SD 17.5%) |
| Mean F1 (macro) | 0.660 (SD 0.171) | 0.521 (SD 0.182) |
| Mean Cohen's kappa | 0.558 (SD 0.221) | 0.399 (SD 0.233) |
| Mean training time per fold | 159.8s (SD 2.0s) | 89.3s (SD 1.6s) |
| Mean inference time per trial | 110.2ms (SD 4.9ms) | 62.0ms (SD 3.8ms) |
| Accuracy points per million parameters | 5.8 | 68.4 |
| Mode-collapsed folds | 0 of 45 | 0 of 45 |

![Accuracy, training speed, and inference speed comparison](results/atcnet_vs_mini_comparison.png)

*Figure 1. Mean accuracy, training time per fold, and inference time per trial, with error bars
showing one standard deviation across forty-five subject-fold runs per configuration.*

### 4.1 Per-Subject Analysis

![Per-subject accuracy](results/atcnet_vs_mini_per_subject.png)

*Figure 2. Mean accuracy per subject (averaged across five folds), compared against chance level
(25 percent for four-class classification).*

The accuracy gap between the two configurations is not uniform across subjects. For subjects with
comparatively high accuracy under both configurations (S03, S07, S08), Mini-ATCNet remains within
four to eight percentage points of ATCNet_full. For subjects with comparatively low accuracy under
both configurations (S04, S05, S06), the gap widens considerably. This pattern is consistent with
the reduced capacity of Mini-ATCNet mattering most precisely where the underlying classification
problem is already harder for a given subject's EEG signal, a phenomenon documented elsewhere in the
BCI literature under the informal term "BCI illiteracy," rather than the smaller model imposing a
fixed accuracy penalty uniformly across all subjects.

### 4.2 Per-Class Analysis

![Per-class F1 score](results/atcnet_vs_mini_per_class_f1.png)

*Figure 3. Per-class F1 score, pooled across all subjects and folds.*

The same unevenness observed across subjects is also present across classes. The "Foot" class shows
the largest gap between configurations (F1 of 0.64 for ATCNet_full versus 0.50 for Mini-ATCNet),
while "Right" shows the smallest (0.68 versus 0.57). Foot is also the weakest class for ATCNet_full
itself, suggesting that Mini-ATCNet's reduced capacity again disproportionately affects classes that
were already the most difficult for the full model, rather than degrading all classes equally.

### 4.3 Accuracy-Efficiency Trade-off

![Parameter count vs accuracy trade-off](results/atcnet_vs_mini_efficiency.png)

*Figure 4. Mean accuracy against trainable parameter count (logarithmic scale).*

Expressed as a single trade-off statement: Mini-ATCNet gives up 11.9 percentage points of mean
accuracy relative to ATCNet_full in exchange for a 93 percent reduction in parameter count, a 44
percent reduction in training time per fold, and a 44 percent reduction in inference time per trial.
Normalized by parameter count, Mini-ATCNet delivers approximately twelve times more accuracy per
million parameters than ATCNet_full (68.4 versus 5.8 accuracy points per million parameters), which
is the relevant comparison when the binding deployment constraint is model size or memory footprint
rather than absolute accuracy.

## 5. Discussion

The magnitude of the accuracy gap observed here (11.9 percentage points) should be interpreted
alongside the substantial reduction in model size that produced it (93 percent fewer parameters).
Neither configuration is presented as categorically preferable; the appropriate choice depends on
which constraint, accuracy or deployment footprint, dominates a given application. For an assistive
BCI system where an incorrect classification has a direct and immediate cost to the user, the
accuracy loss of Mini-ATCNet may not be an acceptable trade for its latency advantage. For a
system where the binding constraint is memory or power budget on embedded hardware, the same trade
may be entirely justified.

It is also worth noting that the 66.8 percent mean accuracy obtained for ATCNet_full in this study
is not directly comparable to the 81.10 percent reported in the original ATCNet publication, since
the two figures are produced under different evaluation protocols. The original work evaluates on a
held-out session using the dataset's own train/test session split, whereas this study evaluates
using subject-dependent five-fold cross-validation within a single session. The absolute accuracy
figures reported here should therefore be read as an internally consistent comparison between the
two configurations under one protocol, not as a reproduction of ATCNet's original reported result.

## 6. Limitations

- **Subject-dependent evaluation only.** All results in this study reflect performance on held-out
  trials from a subject the model has already been trained on. Whether the accuracy gap between the
  two configurations widens, narrows, or reverses under subject-independent evaluation, where the
  model is tested on a subject entirely absent from training, is not addressed here.
- **A single, shared hyperparameter configuration across all subjects.** Learning rate, warm-up
  length, and regularization strength were held fixed across all nine subjects for both
  architectures, rather than tuned per subject. This is a deliberate choice for a controlled
  comparison, but it likely understates the accuracy each configuration could achieve with
  per-subject tuning.
- **Identical regularization strength for both configurations.** The same L2 and max-norm
  constraint values were applied to both ATCNet_full and Mini-ATCNet. Given the large difference in
  parameter count between the two, this regularization strength may be well suited to the larger
  model while being unnecessarily restrictive for the smaller one.
- **No architecture-level comparison against the original ATCNet protocol.** As discussed in
  Section 5, the accuracy figures reported here are not directly comparable to the original ATCNet
  publication's reported accuracy, since the evaluation protocols differ.

## 7. Conclusion

This study set out to answer a specific, quantifiable question: when a published EEG motor imagery
architecture is deliberately reduced along every major hyperparameter for a lightweight deployment
target, how much accuracy is given up, and how much is gained in return along the metrics that
matter for real-time deployment? Under a controlled, leakage-free, subject-dependent evaluation
protocol on BCI Competition IV-2a, reducing ATCNet to 7.0 percent of its original parameter count
costs 11.9 percentage points of mean accuracy while reducing training time and inference latency by
44 percent each, and improving accuracy delivered per parameter by roughly a factor of twelve. These
figures are intended to support an informed choice between the two configurations based on the
constraints of a specific deployment scenario, rather than to argue that one configuration is
unconditionally superior to the other.

## 8. Future Work

- **Subject-independent (leave-one-subject-out) evaluation.** Training on eight subjects and testing
  on the ninth, repeated across all nine subjects, would test whether the accuracy gap between the
  two configurations observed here under subject-dependent evaluation persists, widens, or narrows
  when the model must generalize to entirely unseen subjects, a substantially harder and more
  deployment-realistic generalization setting.
- **Per-subject hyperparameter tuning.** Tuning learning rate, warm-up length, and regularization
  strength individually per subject, rather than sharing one configuration across all nine subjects,
  could close part of the observed gap, particularly for the lower-accuracy subjects identified in
  Section 4.1.
- **Ensembling across fold-level models.** Combining the predictions of the five fold-level models
  saved for each subject at inference time could reduce the high per-subject variance visible in
  Figure 2 without requiring any architectural change.
- **Reduced regularization for Mini-ATCNet.** Given the limitation noted in Section 6, independently
  tuning the L2 and max-norm constraint strength for Mini-ATCNet, rather than inheriting
  ATCNet_full's values, may recover some of the observed accuracy gap.

## Repository Structure

```
bci-atcnet-vs-mini-atcnet/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 02_atcnet_vs_mini_atcnet_augment_lrwarmup.ipynb   (end to end, GPU runnable, developed on Kaggle, with saved outputs)
├── src/
│   ├── config.py     (dataset paths, channel list, MODEL_CONFIGS, training defaults)
│   ├── data.py        (load_subject_epochs: trial level epoching, no sliding window leakage)
│   ├── augment.py      (jitter, scaling, magnitude warp augmentation for the inner training split)
│   ├── model.py         (ATCNet architecture blocks, build_atcnet, LRWarmup callback)
│   └── train.py          (subject dependent five fold CV training loop and CLI entrypoint)
└── results/
    ├── generate_plots.py                     (regenerates the figures above from the notebook's own recorded output)
    ├── atcnet_vs_mini_comparison.png
    ├── atcnet_vs_mini_per_subject.png
    ├── atcnet_vs_mini_efficiency.png
    ├── atcnet_vs_mini_per_class_f1.png
    └── legacy/                                (original plots from the Kaggle run, kept for reference)
```

## How to Run

The dataset itself is not included in this repository (see below). The notebook in `notebooks/` is
the reference, end to end, GPU runnable version developed and run on Kaggle. The `src/` directory
contains the same training logic split into importable modules, runnable directly:

```bash
pip install -r requirements.txt

python -m src.train --subject 1 --config Mini-ATCNet --save_dir_root ./saved_models
python -m src.train --subject 1 --config ATCNet_full --save_dir_root ./saved_models
```

Set `DATA_DIR` in `src/config.py` (or pass `data_dir=` to `load_subject_epochs`) to point at a
local copy of the dataset instead of the Kaggle mount path.

### Dataset

BCI Competition IV, dataset 2a, in GDF format: [official description and download](https://www.bbci.de/competition/iv/#dataset2a).
Not redistributed here due to size and licensing.

## Acknowledgments

The BCI Competition IV-2a dataset was made available by its original organizers and contributors
(Tangermann et al., 2012). The ATCNet architecture and its official implementation, which this
project reimplements and extends with a lightweight variant, are due to Altaheri, Muhammad, and
Alsulaiman (2023). All model training in this project was performed on Kaggle's freely provided
Tesla T4 GPU infrastructure.

## References

- Altaheri, H., Muhammad, G., and Alsulaiman, M. (2023). Physics-informed attention temporal
  convolutional network for EEG-based motor imagery classification. *IEEE Transactions on
  Industrial Informatics*, 19(2), 2249 to 2258. https://doi.org/10.1109/TII.2022.3197419.
  Official implementation: https://github.com/Altaheri/EEG-ATCNet
- Tangermann, M., Muller, K. R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., Leeb, R.,
  Mehring, C., Miller, K. J., Mueller-Putz, G., Nolte, G., Pfurtscheller, G., Preissl, H.,
  Schalk, G., Schlogl, A., Vidaurre, C., Waldert, S., and Blankertz, B. (2012). Review of the BCI
  Competition IV. *Frontiers in Neuroscience*, 6, 55. Dataset description:
  https://www.bbci.de/competition/iv/desc_2a.pdf
