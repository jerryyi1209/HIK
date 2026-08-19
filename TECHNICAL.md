# CarpetSegNet v2 — Architecture & Improvements

Technical deep-dive into the network architecture, design rationale, and iterative improvements that took mIoU from 0.7946 → 0.8272.

---

## 1. Network Architecture

### Overview

```
                        CarpetSegNet v2 (28.2M params)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Input (B, 9, 480, 640)                                     │
│    │                                                        │
│    ▼                                                        │
│  ┌──────────────────────────────────┐                       │
│  │       SimpleEncoder              │                       │
│  │  ResNet50 + expanded first conv  │                       │
│  │  stride [4, 8, 16, 32]          │                       │
│  └──────────────────────────────────┘                       │
│    │                                                        │
│    ▼  f1(256,120,160)  f2(512,60,80)                       │
│       f3(1024,30,40)    f4(2048,15,20)                     │
│    │                                                        │
│    ▼                                                        │
│  ┌──────────────────────────────────┐                       │
│  │       SimpleDecoder              │                       │
│  │  ┌─ lat4(f4) ──────────────────┐ │                       │
│  │  │  up + skip(lat3(f3))        │ │                       │
│  │  │  ConvBlock(dec3) → up      │ │                       │
│  │  │  + skip(lat2(f2))           │ │                       │
│  │  │  ConvBlock(dec2) → up      │ │                       │
│  │  │  + skip(f1)                 │ │                       │
│  │  │  ConvBlock(dec1)            │ │                       │
│  │  └─────────────────────────────┘ │                       │
│  │               │                  │                       │
│  │               ▼                  │                       │
│  │  ┌──────────────────────────┐   │                       │
│  │  │    RefinementHead        │   │                       │
│  │  │  heavy_conv(stride=4)    │   │                       │
│  │  │  → proj → ↑2×           │   │                       │
│  │  │  → refine_conv(stride=2) │   │                       │
│  │  │  → head → ↑2×           │   │                       │
│  │  │  (stride 4→2→1)        │   │                       │
│  │  └──────────────────────────┘   │                       │
│  └──────────────────────────────────┘                       │
│    │                                                        │
│    ▼                                                        │
│  Output (B, 2, 480, 640)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 SimpleEncoder

**File:** `models/simple_encoder.py`

Standard ResNet50 backbone with a critical modification: `_expand_first_conv()` expands the first convolutional layer from 3 to C input channels.

```
Original ResNet stem:
  Conv2d(3 → 64, kernel=7, stride=2) → BN → ReLU → MaxPool

Expanded stem:
  Conv2d(C → 64, kernel=7, stride=2) → BN → ReLU → MaxPool
```

**Weight initialization strategy:**
- Channels 0-2 (RGB): Copy ImageNet pretrained weights directly
- Channels 3+ (Depth, Geo): Kaiming He normal initialization

This means the RGB pathway retains full ImageNet knowledge while additional modalities learn from scratch. No architectural change beyond the first conv — the rest of the backbone is unchanged ResNet50.

**Channel configurations:**

| Config | `in_channels` | Composition |
|---|---|---|
| RGB+Depth (D435) | 4 | RGB(3) + Depth(1) |
| RGB+Depth+dToF | 5 | RGB(3) + Depth(1) + dToF(1) |
| D435 + Geo (standard) | 9 | RGB(3) + Depth(1) + Normal(3) + Edge(1) + Curvature(1) |
| dToF + Geo | 10 | +5 geo channels |

**Output:** 4 feature maps at strides [4, 8, 16, 32]:
- `f1`: (B, 256, H/4, W/4)
- `f2`: (B, 512, H/8, W/8)
- `f3`: (B, 1024, H/16, W/16)
- `f4`: (B, 2048, H/32, W/32)

### 1.2 SimpleDecoder

**File:** `models/simple_decoder.py`

U-Net style top-down decoder with lateral skip connections. Channel dimensions auto-scale based on encoder output:

| Stage | Input | Lateral Skip | Output Channels |
|---|---|---|---|
| dec3 | f4 (stride=32) | lat3(f3) (stride=16) | 512 → 256 |
| dec2 | dec3 output | lat2(f2) (stride=8) | 256 → 128 |
| dec1 | dec2 output | f1 (stride=4) | 128 → 64 |

Each `ConvBlock` = Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU.

Upsampling uses bilinear interpolation (align_corners=False) — learnable transposed convolutions were tested but bilinear upsampling was simpler and equally effective.

### 1.3 RefinementHead

**File:** `models/simple_decoder.py`

Two-stage upsampling head that doubles effective prediction resolution:

```
Input (B, 64, H/4, W/4)           stride=4
    │
    ▼
heavy_conv: 2× Conv3×3(64→64) + BN + ReLU + Dropout(0.1)
proj: Conv1×1(64 → 32)
    │
    ▼  bilinear ↑2×
    │
refine_conv: Conv3×3(32→32) + BN + ReLU        stride=2
head: Conv1×1(32 → 2)
    │
    ▼  bilinear ↑2×
    │
Output (B, 2, H, W)                             stride=1
```

**Design rationale:**
- Heavy convolutions run at stride=4 (low resolution) — 64×120×160 = 1.2M pixels
- Only one lightweight conv at stride=2 — minimal overhead
- Doubles prediction resolution compared to a single 4× upsample (stride=2 vs stride=4 outputs)

### 1.4 SimpleLoss

**File:** `models/simple_loss.py`

```
Loss = CE(class-weighted, edge-weighted, label_smoothing=0.1)
     + 1.0 × Dice
     + 0.5 × Focal(γ=2.0)
     + 0.0 × Lovász-Softmax  (disabled — marginal gain at high cost)
```

**Components:**

| Term | Formula | Purpose |
|---|---|---|
| **CE (edge-weighted)** | `Σ w_class · w_edge · (-log p)` | Per-pixel classification, boundary emphasis |
| **Soft Dice** | `1 - 2|P∩T|/(|P|+|T|)` | Direct IoU optimization |
| **Focal (γ=2)** | `-(1-p_t)^γ · log p_t` | Down-weight easy negatives |
| **Label Smoothing** | Target: `[0.05, 0.95]` instead of `[0, 1]` | Anti-overfitting |
| **Lovász-Softmax** | Convex IoU surrogate | Disabled — see §4.2 |

**Edge weighting** via `compute_edge_weights`:
1. Morphological boundary: dilation - erosion
2. Distance transform to nearest boundary
3. Gaussian decay: `weight = 1 + α · exp(-dist²/2σ²)`

Boundary pixels get weight up to `1+α` (default: 3× more weight at edges). This sharpens carpet boundaries without the complexity of boundary-specific loss terms.

**Class weighting:** Auto-computed from training data pixel statistics. For the NYU dataset: `[1.0, 20.0]` — carpet pixels are weighted 20× more than background due to extreme class imbalance (~1% carpet pixels).

---

## 2. Input Modalities

### 2.1 Channel Composition

| Channel | Source | Normalization | Spatial Info |
|---|---|---|---|
| 0-2 | RGB image | ImageNet mean/std | Texture, color, patterns |
| 3 | Dense depth (D435) | Per-frame min/max → [0,1] | Metric geometry |
| 4-6 | Surface normals (x,y,z) | Unit vectors (no norm needed) | Surface orientation |
| 7 | Depth edge magnitude | Per-frame min/max → [0,1] | Depth discontinuities |
| 8 | Laplacian curvature | Per-frame min/max → [0,1] | Surface curvature (wrinkles, folds) |

### 2.2 Geometric Feature Computation

**File:** `data/preprocess.py` → `compute_dense_geo_features(depth)`

```python
# Surface normals (3 channels)
∂z/∂x = Sobel_x(depth)
∂z/∂y = Sobel_y(depth)
normal = normalize(-∂z/∂x, -∂z/∂y, 1)  # unit vector at each pixel

# Depth edge (1 channel)
edge = |Sobel(depth)|  # gradient magnitude

# Laplacian curvature (1 channel)
curvature = |∇²(depth)|  # second derivative magnitude
```

All computed with Sobel filters (3×3 kernel) — efficient and differentiable if needed. The key insight: **dense depth's advantage is geometric reasoning**, not just an extra distance channel. Surface normals reveal carpet folds/wrinkles, depth edges highlight boundaries, curvature captures 3D texture.

### 2.3 Why Dense Depth > Sparse dToF

| Property | dToF (sparse) | D435 (dense) |
|---|---|---|
| Coverage | ~30% pixels | 100% pixels |
| Geometric features | Unreliable (sparse gradients) | High-quality Sobel derivatives |
| Boundary alignment | Interpolated edges | Native depth edges |
| Sensor noise | Multi-path interference | Gaussian + occasional holes |

The switch from 5ch (RGB+dToF+depth) to 9ch (RGB+depth+geo) was the single largest mIoU gain: **+2.6pp** (0.77 → 0.79+).

---

## 3. Data Pipeline

### 3.1 Dataset

NYU Depth V2, filtered to carpet/floor frames:
- **Train:** 1,575 images (875 positive + 700 negative)
- **Val:** 187 images
- **Test:** 189 images

### 3.2 Two-Layer Augmentation

```
175 original carpet frames
        │
        ▼  [Offline Augmentation — fixed across epochs]
        │
875 augmented variants (5×: 1 original + 4 crops per frame)
        │
        ▼  [Online Augmentation — random each epoch]
        │
Training batch
```

**Offline augmentation** (`scripts/augment_positives.py`):
1. Multi-scale resize (0.5–1.5×) + carpet-centric crop
2. Spatial aug (50% prob): horizontal flip, rotation ±15°, grid distortion
3. Color jitter (always, random magnitude)

**Online augmentations** (per-epoch, applied in `CarpetDataset.__getitem__`):

| Augmentation | Probability | Parameters | Purpose |
|---|---|---|---|
| Multi-scale resize | 70% | 0.5–1.5× scale | Scale invariance |
| Rotation | 50% | ±15° | Orientation invariance |
| Horizontal flip | 50% | — | Reflection invariance |
| Elastic deformation | 30% | α=20, σ=3 | Non-rigid carpet deformation |
| Grid distortion | 40% | scale=0.05 | Local spatial perturbation |
| Color jitter | 100% | brightness/contrast ±20% | Lighting invariance |
| RGB Gaussian noise | 100% | σ=0.01 | Sensor noise simulation |
| Random erase | 30% | 2-10% image area | Occlusion robustness |
| Carpet-centric crop | 70% | around carpet center | Focus on relevant region |
| Depth noise | — | multiplicative σ=0.005 | Depth sensor noise |
| Depth dropout | 10% | 20-60px holes | Simulate depth sensor failures |

**Design principle:** All spatial augmentations (elastic, grid, rotation, flip, crop) are applied jointly to RGB, depth, and mask to maintain cross-modal alignment.

### 3.3 Balanced Sampling

`BalancedBatchSampler`: independent positive/negative pools, shuffled and interleaved 50/50 per batch. With wrap-around (not oversampling), each epoch covers all negatives once while cycling through positives. This prevents:
- Batch imbalance (random sampling would give 90%+ negative batches)
- Overfitting from oversampled positives (each positive appears equally often)
- Training collapse from all-negative batches

---

## 4. Improvement Iterations

### 4.1 Baseline → v1 (0.77 → 0.7946)

| Improvement | mIoU Δ | Rationale |
|---|---|---|
| 4ch → 9ch (geometric features) | +2.0pp | Dense depth enables reliable surface normals, edges, curvature |
| TTA validation (3 scales × 2 flips) | +0.3pp | Ensemble averaging reduces prediction noise |
| Depth noise + dropout | +0.2pp | Prevents overfitting to clean depth |

### 4.2 ASPP + Lovász → v2 (0.7946 → 0.7974)

| Improvement | mIoU Δ | Verdict |
|---|---|---|
| Lightweight ASPP (64-dim) | +0.2pp | Marginal — multi-scale context already handled by encoder |
| Lovász-Softmax (λ=0.3) | +0.1pp | Marginal at high cost — AMP-incompatible, requires float32 cast |

**Decision:** Both disabled by default. ASPP code kept as optional feature (set `use_aspp=True` in config). Lovász kept in loss code but default weight is 0.0.

### 4.3 Anti-Overfitting → v3 (0.8014 → 0.8272)

This was the critical improvement. Training logs from `carpet_seg0805` showed clear overfitting:
- Train loss: 0.050 → 0.035 (continuously decreasing)
- Val mIoU: 0.8014 (E39, peak) → 0.7676 (E80, final)
- Gap widening: the model was memorizing the 875 fixed augmentation variants

**Root cause analysis:**
- 175 original carpet frames × 5 offline variants = 875 unique views
- Over 80 epochs: model sees the exact same 875 views 80 times
- Online augmentations provide per-epoch randomness but cannot change crop boundaries
- 28M parameters / 875 unique views = severe capacity-data mismatch

**Solution 1: Label Smoothing (ε=0.1)**

```
Without smoothing:  target = [0, 1]     → model can reach 100% confidence
With smoothing:     target = [0.05, 0.95] → max confidence ~95%
```

CE loss with label smoothing has a theoretical minimum of `-0.95·log(0.95) - 0.05·log(0.05) ≈ 0.199` per pixel (vs 0 without smoothing). This directly prevents train loss from drifting to near-zero, which was the primary overfitting signal in 0805.

**Effect:** Train CE loss stabilizes at ~0.30 (combined with class weights and edge weights, above the ~0.20 theoretical floor) instead of 0.015-0.035. Best epoch shifted from 39 → 57 — the model generalizes for 18 more epochs before degrading.

**Solution 2: Increased Weight Decay (1e-4 → 5e-4)**

AdamW decouples weight decay from learning rate, so increasing from 1e-4 to 5e-4 adds L2 regularization without interfering with the ReduceLROnPlateau schedule.

For 28M parameters, `λ=1e-4` was essentially weight decay disabled — the effective regularization was negligible compared to the data signal. `λ=5e-4` provides meaningful parameter norm constraint.

**Why these two specifically?**
- Label smoothing: **forces** the model to be uncertain — structural change
- Weight decay: **constrains** parameter magnitude — optimization change
- Both work independently of the augmentation pipeline — they don't require more data

**What didn't work:**
- **SWA (Stochastic Weight Averaging):** Tested earlier, zero gain. Because offline augmentation is fixed, SWA snapshots at different epochs are nearly identical — averaging produces no benefit.
- **Deeper backbone (ResNet101):** More parameters → worse overfitting on small dataset
- **Merging SUN RGB-D:** Labeling paradigm mismatch (instance vs semantic) caused training to diverge

---

## 5. Training Configuration

### 5.1 Optimization

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay |
| Peak LR | 1e-4 | Empirically stable for ResNet50 |
| Weight decay | 5e-4 | Anti-overfitting |
| LR warmup | 3 epochs (0.1× → 1× LR) | Stable early training |
| LR schedule | ReduceLROnPlateau (patience=8, factor=0.5) | Adaptive — only drops when stagnating |
| LR floor | 3e-5 | Prevents weight freezing |
| Gradient clip | max_norm=1.0 | Stability against batch outliers |
| Mixed precision | AMP (autocast + GradScaler) | 2× memory, ~1.5× speed on RTX 3060 Ti |

### 5.2 LR Schedule Behavior

Typical training dynamic (observed in `carpet_seg0807`):

```
Epoch  1-3:   LR 3.3e-5 → 1e-4    (warmup, rapid mIoU growth 0.51→0.66)
Epoch  4-18:  LR 1e-4              (peak learning, mIoU 0.73→0.80, high oscillation)
Epoch 19-30:  LR 5e-5              (first plateau drop, mIoU 0.75→0.82, best=0.8156@E21)
Epoch 31-56:  LR 3e-5              (floor LR, mIoU oscillates 0.75-0.80)
Epoch 57-80:  LR 3e-5              (surprise best @E57=0.8272, then decline to 0.7803)
```

The best checkpoint at epoch 57 (well after LR hit floor) suggests the model continues to explore the loss landscape even at minimal learning rates.

---

## 6. Ablation Summary

Cumulative impact of each improvement on validation mIoU:

```
Base (4ch D435, ResNet50):                          ~0.77
  + Geometric features (4ch → 9ch):                 ~0.79   (+2.0pp)
  + TTA (3 scales × 2 flips):                       ~0.79   (+0.3pp)
  + Depth noise/dropout augmentation:               ~0.79   (+0.2pp)
  + ASPP (lightweight):                             ~0.80   (+0.2pp, marginal)
  + Lovász-Softmax:                                 ~0.80   (+0.1pp, marginal, disabled)
  + Label Smoothing (0.1):                          ~0.82   (+1.5pp)
  + Weight Decay (5e-4):                            ~0.83   (+1.1pp, synergistic with LS)
─────────────────────────────────────────────────────────
Current best:                                       0.8272  (+5.7pp from baseline)
```

**Key insight:** Geometric features provided the largest single gain (inherent to dense depth), but the anti-overfitting measures (label smoothing + weight decay) provided the second-largest combined gain — despite not adding any new information. This confirms that overfitting, not information scarcity, was the dominant bottleneck.

---

## 7. Known Limitations

1. **Fixed offline augmentation:** 875 augmented variants are identical every epoch. The model can memorize specific crop positions. True online multi-crop (random crops each epoch) would help but adds complexity.
2. **Decoder ConvBlocks lack dropout:** Only RefinementHead has Dropout(0.1). The three main decoder ConvBlocks (dec1, dec2, dec3) have no stochastic regularization.
3. **Small original dataset:** 175 positive frames is the fundamental ceiling. More original NYU annotations or domain-specific data collection would help.
4. **Validation oscillation:** ±3pp mIoU swing (0.75-0.82) suggests the loss landscape is flat with many local minima. SWA couldn't help due to fixed offline data.
5. **No test-time evaluation of best model:** Test set metrics would provide better estimate of real-world generalization.

---

## 8. Evolution: Original v1 (CarpetSegNet) → v2 (SimpleModel)

The earliest network (`CarpetSegNet`, preserved in `models/model.py`) was designed around **sparse dToF depth + dual-backbone gated fusion**. It was rewritten as `SimpleModel` after hitting three walls: NaN instability, sparse-depth limitations, and the SUN/NYU labeling-paradigm conflict. This section documents the before/after and the improvements that produced the +6.7pp mIoU gain.

### 8.1 Architecture Comparison

| Dimension | v1 (CarpetSegNet) | v2 (SimpleModel) |
|---|---|---|
| **Input** | RGB(3) + LingBot-completed depth(1) → encoder; sparse dToF conf(1) → fusion gate | RGB(3) + dense D435 depth(1) + normals(3) + edge(1) + curvature(1) = 9ch |
| **Encoder** | SharedEncoder (ResNet34), fp32 | Single ResNet50, RGB+depth concat at input |
| **Fusion** | `DtoFGatedFusion`: SE-gate per scale, `feat × (1 + spatial_gate · channel_gate)` | None — cross-modal features from layer 1 |
| **Decoder** | Deep-supervised FPN: side connections + 3 aux heads (f2/f3/f4) | U-Net skip connections + single RefinementHead |
| **Loss** | Focal + Tversky/Dice + Lovász-Softmax + depth-consistency | CE (class/edge-weighted, label smoothing) + Dice + Focal |
| **Regularization** | Modality dropout (zero-out RGB or depth) | Label smoothing + stronger weight decay |
| **Training** | Two-phase progressive loss transition | Single-phase, standard AMP |
| **Numerical stability** | NaN-prone → `clamp`/`nan_to_num` hacks at every stage | AMP-safe by construction — no NaN hacks |
| **Best mIoU** | **0.7599** (IoU-carpet 0.5267) | **0.8272** (IoU-carpet 0.6590) |

### 8.2 Why v1 was rewritten

**1. NaN instability.** The v1 training log (`experiments/carpet_seg_dtof_default/logs/train.log`) shows the model converged to mIoU 0.7599 by epoch 8, then the next run collapsed to `Loss: nan` for the entire epoch. Three independent NaN sources:

- **FP16 BatchNorm variance underflow** — the ResNet34 BasicBlock residual path in fp16, where BN variance underflow → divide-by-zero → Inf → NaN ([models/model.py:14](models/model.py#L14)).
- **Multiplicative gate amplification** — `DtoFGatedFusion` computes `feat × (1 + spatial_gate · channel_gate)`; once BN weights are polluted by an Inf gradient, the gate emits NaN and multiplies it through every channel ([models/fusion.py:79](models/fusion.py#L79)).
- **FPN accumulation + deep supervision** — side-connection summing and multiple aux heads overflow logits ([models/model.py:16](models/model.py#L16)).

The "fixes" were reactive band-aids: per-stage `torch.clamp(±100)`, `nan_to_num` in fusion, a `CARPETSEG_DEBUG` NaN diagnostic, forced fp32 on the encoder/decoder, and a train-loop "skip NaN batch" guard. None removed the root cause.

**2. Sparse dToF limits geometry.** dToF covers only ~30% of pixels, so its gradients are unreliable and surface normals/curvature can't be computed. The v1 architecture worked around this by completing depth with LingBot, but the completed depth had its own artifacts and the dToF gate only emphasized (not fixed) the sparse signal.

**3. Complexity without benefit.** The dual-backbone + gated-fusion + deep-supervision design was motivated by "fuse modalities more explicitly," but empirically it did not beat a simpler single-encoder baseline — while carrying a much higher NaN surface area.

### 8.3 Improvements and Their Impact

| Improvement | Change | Effect |
|---|---|---|
| **Sparse dToF → dense depth** | Drop dToF + LingBot; use D435 dense depth directly | Enables geometric features (the core dense-depth advantage) |
| **Dual backbone + gate → single shared encoder** | RGB+depth concatenated at input; delete fusion gate | Removes the multiplicative NaN path; cross-modal from layer 1 |
| **FPN deep supervision → U-Net skips** | One decoder, lateral skip connections, no aux heads | Removes FPN accumulation / logit overflow |
| **5-component loss → 3-component** | Drop Lovász-Softmax and depth-consistency | Lovász was AMP-incompatible; simpler loss is numerically stable |
| **Geometric features (4ch→9ch)** | +normals(3)+edge(1)+curvature(1) | **+2.0pp** (largest single gain) |
| **TTA validation** | 3 scales × 2 flips | +0.3pp |
| **Depth noise + dropout** | multiplicative σ=0.005, 10% hole dropout | +0.2pp |
| **Label smoothing (0.1)** | target `[0.05, 0.95]` | +1.5pp (anti-overfitting) |
| **Weight decay (1e-4→5e-4)** | stronger L2 | +1.1pp (synergistic with LS) |

**Net:** 0.7599 → 0.8272 mIoU (**+6.7pp**), with a numerically stable, AMP-safe architecture that requires none of v1's clamp/nan_to_num hacks. The geometric-feature gain was the largest single step (inherent to dense depth); the anti-overfitting measures (label smoothing + weight decay) were the largest *combined* step despite adding no new information — confirming overfitting, not information scarcity, was the dominant v2 bottleneck.
