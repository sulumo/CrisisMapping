# CrisisMMD Multimodal Classification

A PyTorch project for multitask crisis-event classification using text and images from the CrisisMMD dataset. This implementation is adapted from [Munia03/Multimodal_Crisis_Event](https://github.com/Munia03/Multimodal_Crisis_Event).

The project extends the upstream approach in two main ways:

1. **Unified task annotations:** the task-specific annotation files are combined into a single TSV for each split, with one label column per task.
2. **Multi-layer visual representation:** alongside a baseline using the final pooled CLIP visual representation, a second model combines early, middle, and final visual representations using task-specific gates.

This README describes the current implementation and the available experimental results. The results are single-run results; limitations affecting interpretation are stated below.

## Tasks

The model predicts three tasks from an image and associated text:

| Task | Objective | Output classes |
|---|---|---:|
| Task 1 | Informativeness | 2 |
| Task 2 | Humanitarian / crisis-information category | 6 |
| Task 3 | Damage severity | 3 |

Task 1 uses `not_informative` = 0 and `informative` = 1. Task 3 uses `little_or_no_damage` = 0, `mild_damage` = 1, and `severe_damage` = 2.

Task 2 uses the following mapping in `crisismmd_dataset.py`:

| Index | Label |
|---:|---|
| 0 | `infrastructure_and_utility_damage` |
| 1 | `not_humanitarian` |
| 2 | `other_relevant_information` |
| 3 | `rescue_volunteering_or_donation_effort` |
| 4 | `vehicle_damage` |
| 5 | `affected_individuals`, `injured_or_dead_people`, and `missing_or_found_people` (merged class) |

The implementation retains six Task 2 output logits. The multi-layer training script notes that class index 1 (`not_humanitarian`) has no samples in the current data setup and assigns it a class weight of zero. Although `not_humanitarian` and `other_relevant_information` can appear semantically close in some cases, the code treats them as distinct labels; they are not merged. The three people-related labels are the ones mapped to the same index, 5.

Missing labels are mapped to `-1`. Each task's loss and metrics are computed using samples with valid labels for that task, so the number of evaluated samples can differ by task.

## Dataset preparation and split assignment

The dataset reader expects these annotation files relative to the project root:

```text
merged/
├── merged_train.tsv
├── merged_dev.tsv
└── merged_test.tsv
```

It expects CrisisMMD images under `./datasets/CrisisMMD_v2.0/`, with the paths in the TSV's `image` column treated as relative paths beneath that root.

The merged TSV is expected to contain these columns:

```text
event_name, tweet_id, image_id, tweet_text, image,
label_task1, label_task2, label_task3, event_category
```

The reader also supports `text` as an alternative to `tweet_text`, and `event` as an alternative to `event_category`. If `llava_text` is present and `--text_from llava` is selected, non-empty LLaVA-generated text is appended to the tweet text. Otherwise, the tweet text is used. Text is tokenized with the CLIP tokenizer with a maximum length of 77 tokens. The dataset code also initializes ELECTRA and LLaMA tokenizers for the current implementation, so access to the corresponding tokenizer assets may be required even though the classifiers use CLIP-tokenized text.

### How the merge script assigns splits

The task-specific annotation files were merged into a unified dataset using `image_id` as the sample key. The merging script builds split maps from the original task datasets and assigns one global split to each merged row using this priority:

1. Informativeness split, if the image is present in the Informativeness source.
2. Damage Severity split, if it is not present in Informativeness but is present in Damage Severity.
3. Humanitarian split, if it is present in neither of the first two sources.

This priority-based procedure ensures that each merged row receives one global split, but **does not guarantee that the original split assignment for every task is preserved** when the same image has conflicting task-specific assignments. The matching key is `image_id`; confirm that this ID is unique for the intended sample across all source files. If the intended identity is the combination of `tweet_id` and `image_id`, the merge procedure should use that composite key consistently.

The original paper's task-level split counts are:

| Task | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Informativeness | 9,599 | 1,573 | 1,534 | 12,706 |
| Humanitarian | 2,874 | 477 | 451 | 3,802 |
| Damage Severity | 2,468 | 529 | 529 | 3,526 |

The provided model logs report test counts of 1,534 for Informativeness, 449 for Humanitarian, and 529 for Damage Severity. Thus, Informativeness and Damage Severity match the original paper's test counts, while Humanitarian is lower by two samples (449 vs. 451; approximately 0.44%). The cause of this discrepancy has not been established from the available logs. These are **task-labeled sample counts**, not unique merged-row counts; one merged row may carry labels for multiple tasks.

The merging script is the best way for readers to inspect and reproduce the transformation. The generated TSV files should only be redistributed with the repository if the applicable CrisisMMD terms permit redistribution. If permission is unclear, provide the script and instructions for generating the TSVs from a separately obtained copy of the original annotations instead of committing the generated data.

## Model options

Both models use the pretrained `openai/clip-vit-base-patch32` checkpoint and three task-specific classification heads. The CLIP image and text transformer encoders are frozen; projection, fusion, gating (for the multi-layer option), and classifier layers are trainable.

### Option 1 — Baseline: final pooled representation

Implementation: `baseline/models_clip.py`

The baseline uses the final pooled representation (`pooler_output`) from the CLIP vision and text encoders. The features are mapped to 768-dimensional representations, projected to a shared 128-dimensional space, and passed through a guided cross-modal feature interaction/fusion pipeline. The fused representation is used by three classifier heads, producing 2, 6, and 3 logits for Tasks 1, 2, and 3, respectively.

This is the simpler reference model: it uses a single final pooled visual representation rather than explicitly combining intermediate vision-transformer features.

### Option 2 — Multi-layer: task-specific visual gating

Implementation: `multilayer/CLIP_CrisisKAN_TaskGate.py`

The multi-layer model requests hidden states from the frozen CLIP vision encoder and extracts:

- Early visual representation: `hidden_states[3]`, using the CLS token.
- Middle visual representation: `hidden_states[7]`, using the CLS token.
- Final visual representation: `pooler_output`.

Each representation is projected to the visual feature dimension. The three representations are concatenated and supplied to three separate gating networks, one for each task. Each gate produces weights over the early, middle, and final representations; the weighted combination is then used in the cross-modal fusion and task-specific classification path.

The motivation is to allow the three tasks to use different mixtures of visual features from different transformer depths. This is an architectural hypothesis, not a guarantee of improved performance across all tasks.

## Training

The main training scripts are:

- Baseline: `baseline/main.py`
- Multi-layer: `multilayer/main.py`

The scripts load train, development, and test datasets, initialize the model, optimize trainable parameters with AdamW, and use a warmup-linear learning-rate schedule. The default arguments in `args.py` include:

| Argument | Default |
|---|---:|
| `--learning_rate` | `1e-4` |
| `--batch_size` | `64` |
| `--num_epochs` | `24` |
| `--weight_decay` | `0.01` |
| `--warmup_ratio` | `0.1` |
| `--seed` | `42` |
| `--num_workers` | `8` |
| `--load_size` | `228` |
| `--crop_size` | `224` |
| `--text_from` | `llava` |
| `--save_dir` | `./output` |

The combined objective uses the following default task weights in both trainers:

\[
\mathcal{L} = 0.16\mathcal{L}_{\mathrm{info}} + 0.46\mathcal{L}_{\mathrm{humanitarian}} + 0.38\mathcal{L}_{\mathrm{damage}}.
\]

Missing labels are ignored independently for each task. The multi-layer script additionally passes explicit per-class loss weights for each task; the baseline script uses unweighted cross-entropy. This difference is important when interpreting the reported comparison.

Run the scripts from the project root so that the relative data paths resolve as expected:

```bash
python baseline/main.py --device cuda --num_epochs 24 --batch_size 64 --model_name baseline
python multilayer/main.py --device cuda --num_epochs 24 --batch_size 64 --model_name multilayer
```

The scripts import shared modules such as `args.py`, `optimization.py`, and `crisismmd_dataset.py`. The dataset implementation also imports `preprocess.py` and `base_dataset.py`; ensure those required modules are present in the repository. Install compatible versions of PyTorch, Transformers, torchvision, timm, scikit-learn, Pillow, termcolor, NumPy, and any other imported dependencies in your environment. Exact dependency versions should be recorded in a requirements file if they are known.

Training logs and checkpoints are written under the selected save directory and run name. The scripts load `best.pt` for the final validation and test prediction stage. The `--merged_dataset_path` argument is declared in `args.py`, but the current dataset reader uses the fixed `./merged/merged_{phase}.tsv` paths; setting that argument alone does not currently redirect dataset loading.

## Results

The following results are from the available single runs. Overall accuracy is calculated across all available labeled task instances, not across unique images or tweets.

### Summary comparison

| Metric | Baseline | Multi-layer | Multi-layer − baseline |
|---|---:|---:|---:|
| Weighted validation loss | 0.400173 | 0.408855 | +0.008682 |
| Overall accuracy | 0.8750 | 0.8746 | −0.0004 |
| Task 1 accuracy | 0.9237 | 0.9179 | −0.0058 |
| Task 1 macro-F1 | 0.9134 | 0.9080 | −0.0054 |
| Task 2 accuracy | 0.9421 | 0.9332 | −0.0089 |
| Task 2 macro-F1 | 0.7182 | 0.7073 | −0.0109 |
| Task 3 accuracy | 0.6767 | 0.6994 | +0.0227 |
| Task 3 macro-F1 | 0.5594 | 0.5852 | +0.0258 |

### Per-task results

| Model | Task | Test samples | Accuracy | Micro-F1 | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Informativeness | 1,534 | 0.9237 | 0.9237 | 0.9134 | 0.9237 |
| Baseline | Humanitarian | 449 | 0.9421 | 0.9421 | 0.7182 | 0.9414 |
| Baseline | Damage severity | 529 | 0.6767 | 0.6767 | 0.5594 | 0.6660 |
| Multi-layer | Informativeness | 1,534 | 0.9179 | 0.9179 | 0.9080 | 0.9183 |
| Multi-layer | Humanitarian | 449 | 0.9332 | 0.9332 | 0.7073 | 0.9315 |
| Multi-layer | Damage severity | 529 | 0.6994 | 0.6994 | 0.5852 | 0.6829 |

### Confusion matrices

The following matrices are copied from the available evaluation logs. Rows represent true labels and columns represent predicted labels according to the evaluation code. The exact Task 2 class ordering should be verified before interpreting individual cells: the classifier has six output logits, but the supplied matrices are 5×5 because only observed labels appear to have been included.

**Baseline — Task 1**

```text
[[444,  60],
 [ 57, 973]]
```

**Baseline — Task 2**

```text
[[80,  4,   0, 0, 1],
 [11, 208,  2, 0, 0],
 [ 3,  1, 128, 0, 0],
 [ 1,  0,   0, 0, 0],
 [ 0,  0,   3, 0, 7]]
```

**Baseline — Task 3**

```text
[[37, 17,  26],
 [14, 39,  59],
 [18, 37, 282]]
```

**Multi-layer — Task 1**

```text
[[453,  51],
 [ 75, 955]]
```

**Multi-layer — Task 2**

```text
[[77,  5,  3, 0, 0],
 [12, 206, 3, 0, 0],
 [ 0,  2, 130, 0, 0],
 [ 1,  0,  0, 0, 0],
 [ 0,  1,  3, 0, 6]]
```

**Multi-layer — Task 3**

```text
[[39, 13,  28],
 [ 9, 38,  65],
 [13, 31, 293]]
```

## Interpretation and limitations

- **Task-specific trade-off:** in these runs, the multi-layer model improves Damage Severity accuracy by 2.27 percentage points and macro-F1 by 2.58 points. The baseline performs better on Informativeness and Humanitarian classification.
- **Overall performance is nearly unchanged:** overall accuracy is 0.8750 for the baseline and 0.8746 for the multi-layer model.
- **The architecture is not isolated experimentally:** the multi-layer run uses explicit per-class loss weights, while the baseline run uses unweighted cross-entropy. The score differences cannot therefore be attributed solely to multi-layer visual gating.
- **Single-run results:** results were not averaged across multiple random seeds. They should be treated as the results of the available runs, not as estimates of average performance or statistical evidence of a consistent improvement.
- **Stochastic evaluation transforms:** the dataset implementation applies random horizontal flips and random crops to development samples. Test samples use random crops. Evaluation may therefore vary across repeated runs for the same checkpoint. This is a limitation of the current implementation; the available results do not establish how much it affects the reported metrics.
- **Class imbalance:** Task 2 accuracy is high while macro-F1 is notably lower, showing that accuracy does not reflect performance equally across classes. Task 3 also exhibits confusion between mild and severe damage.
- **Task 2 matrix interpretation:** the reported confusion matrices are 5×5 even though the classifier outputs six logits. The exact label ordering should be checked against the evaluation implementation before drawing class-specific conclusions.
- **Split assignment:** the merging script's priority-based global split assignment can override a task's original split when the same image has conflicting assignments across task-specific files. The original test counts match for Informativeness and Damage Severity; the Humanitarian test count is two samples lower. The cause of this difference has not been established.
- **Environment reproducibility:** the training code seeds Python, NumPy, and PyTorch and configures cuDNN determinism. Exact reproducibility can still depend on software versions, hardware, data processing, and checkpoint selection.

These limitations do not invalidate the reported results, but they constrain the conclusions that can be drawn. The current evidence supports describing multi-layer gating as a task-specific alternative that performed better on Damage Severity in the available run, rather than claiming a universal improvement over the baseline.
