# RadCBM: Hierarchical Concept Bottleneck Models with Automated Annotations for Chest X-ray Interpretation

## Paper Summary Document

---

## 1. Problem Statement

### 1.1 The Interpretability Gap

Deep learning models for medical image classification achieve high accuracy but operate as black boxes. In clinical settings, this opacity is problematic:

- Clinicians cannot verify the reasoning behind predictions
- Errors are difficult to diagnose and correct
- Regulatory approval requires explainability
- Trust is essential for clinical adoption

### 1.2 Concept Bottleneck Models (CBMs)

CBMs address interpretability by forcing predictions through an intermediate concept layer:

```
Image → Concept Predictions → Label Prediction
```

Each concept is human-interpretable (e.g., "lung opacity," "cardiomegaly"). The final prediction is a function of these concepts, making the reasoning transparent.

**The bottleneck problem:** CBMs require manual concept annotations for training. For medical imaging, this means:

- Expert radiologists must label each concept for each image
- Annotation cost scales with dataset size × number of concepts
- Typical datasets have 100,000+ images and 50-200 concepts
- This is prohibitively expensive

### 1.3 The Opportunity

Radiology reports already contain concept-level information in natural language:

> "There is patchy opacity in the left lower lobe consistent with pneumonia. No pleural effusion. Heart size is normal."

This report implicitly labels:
- `lung:opacity` = present
- `lung:pneumonia` = present
- `pleura:effusion` = absent
- `heart:cardiomegaly` = absent

**Key insight:** We can extract concept annotations automatically from paired radiology reports, eliminating manual labeling.

---

## 2. Proposed Solution: RadCBM

### 2.1 Overview

RadCBM is a framework that:

1. Extracts concept annotations from radiology reports using RadGraph
2. Maps extracted entities to standardized medical terminology (RadLex/UMLS)
3. Organizes concepts hierarchically by anatomical region
4. Trains a two-level Concept Bottleneck Model
5. Provides interpretable predictions at region and finding levels

### 2.2 Key Contributions

1. **Automated concept annotation:** First work to use RadGraph for CBM concept extraction, eliminating manual labeling

2. **Hierarchical concept organization:** Two-level structure (anatomy → findings) that mirrors radiological reasoning

3. **Multiplicative gating:** Region abnormality gates finding predictions, enforcing clinical consistency

4. **Standardized vocabulary:** UMLS/RadLex mapping ensures consistent, clinically-grounded concepts

---

## 3. Method

### 3.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Radiology Reports                                              │
│        ↓                                                        │
│  RadGraph Extraction                                            │
│        ↓                                                        │
│  Entity Normalization (RadLex/UMLS)                             │
│        ↓                                                        │
│  Semantic Type Filtering                                        │
│        ↓                                                        │
│  Hierarchical Concept Vocabulary                                │
│        ↓                                                        │
│  Concept Target Vectors (ĉ)                                     │
│                                                                 │
│  Chest X-ray Images ──→ Hierarchical CBM ──→ Predictions        │
│                              ↑                                  │
│                         Supervision                             │
│                          (ĉ, y)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Concept Extraction from Reports

#### 3.2.1 RadGraph Extraction

RadGraph is an information extraction model that identifies:

- **Entities:** Observations (OBS) and Anatomy (ANAT)
- **Attributes:** Definitely Present (DP), Definitely Absent (DA), Uncertain (U)
- **Relations:** `located_at` linking observations to anatomy

Example extraction:

```
Report: "Patchy opacity in the left lower lobe. No effusion."

Entities:
  - e1: {text: "opacity", label: OBS-DP}
  - e2: {text: "left lower lobe", label: ANAT-DP}
  - e3: {text: "effusion", label: OBS-DA}

Relations:
  - {source: e1, target: e2, label: located_at}
```

#### 3.2.2 Entity Normalization

Raw text spans are mapped to canonical terms via RadLex (preferred) or UMLS:

```
"opacities" → RadLex lookup → RID3874 → "opacity"
"effusion"  → RadLex lookup → RID34539 → "pleural effusion"
```

#### 3.2.3 Semantic Type Filtering

Only clinically relevant types are retained:

**Included (Findings - Level 2):**
| RadLex Type | RID | Description |
|-------------|-----|-------------|
| Imaging observation | RID5 | opacity, consolidation, mass |
| Clinical finding | RID6 | edema, effusion |
| Pathophysiologic process | RID4 | pneumonia, fibrosis |

**Included (Anatomy - Level 1):**
| RadLex Type | RID | Description |
|-------------|-----|-------------|
| Anatomic entity | RID2 | lung, heart, pleura (filtered to chest) |

**Excluded:**
| RadLex Type | RID | Rationale |
|-------------|-----|-----------|
| Imaging modality | RID10 | Not findings |
| Procedure | RID11 | Not findings |
| Property/Attribute | RID12 | Too generic |
| Modifier | RID7 | Handled via compound concepts |
| Report component | RID13 | Meta-information |

### 3.3 Hierarchical Concept Structure

#### 3.3.1 Two-Level Hierarchy

```
Level 1: Anatomical Regions
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│  Lung   │  Heart  │ Pleura  │Mediast. │  Bone   │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘
     │         │         │         │         │
Level 2: Findings per Region
┌────┴────┐    │    ┌────┴────┐    │         │
│opacity  │    │    │effusion │    │         │
│nodule   │   ...   │thicken. │   ...      ...
│consol.  │         │pneumo.  │
└─────────┘         └─────────┘
```

#### 3.3.2 Concept Vocabulary Construction

From training corpus:

1. Extract all (anatomy, finding, status) tuples via RadGraph
2. Normalize entities via RadLex
3. Filter by semantic type
4. Group findings by anatomical region
5. Apply frequency threshold (τ ≥ 50 occurrences)
6. Remove redundant concepts (correlation > 0.95)

Result: K anatomical regions, each with N_r findings. Total N = Σ N_r concepts.

#### 3.3.3 Concept Target Construction

For each report, construct target vector ĉ ∈ {0, 0.5, 1}^N:

```
ĉ_i = 1.0   if concept i is OBS-DP (definitely present)
ĉ_i = 0.0   if concept i is OBS-DA (definitely absent)
ĉ_i = 0.5   if concept i is OBS-U (uncertain)
ĉ_i = 0.0   if concept i is not mentioned (assumed absent)
```

Region-level targets (for Level 1 supervision):

```
â_r = 1.0   if any finding in region r has ĉ_i > 0
â_r = 0.0   otherwise
```

### 3.4 Model Architecture

#### 3.4.1 Overview

```
Image x
    ↓
Backbone φ (DenseNet-121)
    ↓
Features h ∈ ℝ^D
    ↓
┌───────────────────────────────────────┐
│ Level 1: Region Abnormality           │
│                                       │
│   a = σ(W_a · h + b_a) ∈ [0,1]^K     │
│                                       │
│   a_r = "Is region r abnormal?"       │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Level 2: Finding Prediction           │
│                                       │
│   For each region r:                  │
│     f_r = σ(W_r · h + b_r) ∈ [0,1]^{N_r}│
│                                       │
│   f_r^{(i)} = "Is finding i present?" │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Multiplicative Gating                 │
│                                       │
│   c_r = a_r · f_r                     │
│                                       │
│   Region abnormality gates findings   │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Concept Vector                        │
│                                       │
│   c = [c_{r_1}; c_{r_2}; ...; c_{r_K}]│
│                                       │
│   c ∈ [0,1]^N                         │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Label Prediction (Linear)             │
│                                       │
│   ŷ = W_y · c + b_y                   │
│                                       │
│   p(y|x) = softmax(ŷ)                 │
└───────────────────────────────────────┘
```

#### 3.4.2 Mathematical Formulation

**Notation:**

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| x | Input image | H × W × 3 |
| h | Backbone features | D |
| a | Region abnormality scores | K |
| f_r | Finding scores for region r | N_r |
| c | Full concept vector | N |
| ŷ | Class logits | L |

**Forward pass:**

```
h = φ(x)                                    # Backbone

a = σ(W_a h + b_a)                          # Level 1: Region abnormality

For each region r ∈ {1, ..., K}:
    f_r = σ(W_r h + b_r)                    # Level 2: Finding prediction
    c_r = a_r · f_r                         # Gating

c = [c_1; c_2; ...; c_K]                    # Concatenate

ŷ = W_y c + b_y                             # Label prediction
```

**Key property:** The label predictor W_y is linear over concepts, making it fully interpretable. Entry W_y[l, i] indicates how concept i influences class l.

### 3.5 Training

#### 3.5.1 Loss Function

Joint optimization of three terms:

```
L = L_label + λ_1 · L_region + λ_2 · L_finding
```

**Label loss (cross-entropy):**

```
L_label = -Σ_l y_l log p(y=l | x)
```

**Region loss (binary cross-entropy):**

```
L_region = -1/K Σ_r [â_r log(a_r) + (1 - â_r) log(1 - a_r)]
```

**Finding loss (binary cross-entropy with soft targets):**

```
L_finding = -1/N Σ_i [ĉ_i log(f_i) + (1 - ĉ_i) log(1 - f_i)]
```

Note: Finding loss is computed on raw predictions f_r, not gated concepts c_r, to ensure gradient flow.

#### 3.5.2 Handling Uncertainty

For concepts with ĉ_i = 0.5 (uncertain):

Option A: Treat as soft target (pushes prediction toward 0.5)

Option B: Down-weight in loss:
```
L_finding = -1/N Σ_i ω_i [ĉ_i log(f_i) + (1 - ĉ_i) log(1 - f_i)]

where ω_i = 1.0 if ĉ_i ∈ {0, 1}, else 0.5
```

#### 3.5.3 Handling Class Imbalance

Most concepts are absent (0) in any given report. Apply positive weighting:

```
L_finding = -1/N Σ_i [β_i · ĉ_i log(f_i) + (1 - ĉ_i) log(1 - f_i)]

where β_i = count(ĉ_i = 0) / count(ĉ_i = 1)
```

#### 3.5.4 Training Procedure

```
Input: Images X, Concept targets Ĉ, Labels Y, Hyperparameters λ_1, λ_2
Output: Trained model parameters θ

For each epoch:
    For each batch (x, ĉ, y):
        1. h = φ(x)                         # Backbone forward
        2. a = σ(W_a h + b_a)               # Region prediction
        3. For each r: f_r = σ(W_r h + b_r) # Finding prediction
        4. c = [a_1·f_1; ...; a_K·f_K]      # Gating
        5. ŷ = W_y c + b_y                  # Label prediction
        
        6. Compute L_label(ŷ, y)
        7. Compute L_region(a, â)
        8. Compute L_finding(f, ĉ)
        
        9. L = L_label + λ_1·L_region + λ_2·L_finding
        10. Backpropagate and update θ
```

### 3.6 Inference and Interpretation

#### 3.6.1 Prediction

Given new image x:

```
h = φ(x)
a = σ(W_a h + b_a)
For each r: c_r = a_r · σ(W_r h + b_r)
c = [c_1; ...; c_K]
ŷ = argmax(W_y c + b_y)
```

#### 3.6.2 Explanation

For predicted class ŷ, the contribution of concept i is:

```
contrib_i = c_i · W_y[ŷ, i]
```

- c_i: concept activation (from image)
- W_y[ŷ, i]: learned weight (how concept influences class)
- Product: actual contribution to decision

**Hierarchical explanation:**

```
Region contribution: contrib_r = Σ_{i ∈ F_r} c_i · W_y[ŷ, i]

Finding contribution: contrib_i = c_i · W_y[ŷ, i]
```

#### 3.6.3 Interpretation Example

```
Prediction: Pneumonia (87.3%)
============================================================

LUNG (abnormality: 94%)                    [region contrib: +0.842]
----------------------------------------------------------------
  opacity         f=0.92 × a=0.94 = c=0.86    w=+0.61  → +0.523
  consolidation   f=0.78 × a=0.94 = c=0.73    w=+0.45  → +0.298
  atelectasis     f=0.31 × a=0.94 = c=0.29    w=+0.12  → +0.035

PLEURA (abnormality: 71%)                  [region contrib: +0.156]
----------------------------------------------------------------
  effusion        f=0.82 × a=0.71 = c=0.58    w=+0.33  → +0.191

HEART (abnormality: 15%)                   [region contrib: -0.043]
----------------------------------------------------------------
  cardiomegaly    f=0.34 × a=0.15 = c=0.05    w=-0.12  → -0.006
```

#### 3.6.4 Concept Intervention

To test counterfactuals ("What if this concept were different?"):

```
c' = c with c_i set to v
ŷ' = argmax(W_y c' + b_y)

If ŷ' ≠ ŷ, concept i is critical for the prediction.
```

---

## 4. Design Decisions and Justifications

### 4.1 Why Hierarchical Structure?

| Benefit | Explanation |
|---------|-------------|
| Clinical alignment | Radiologists reason by region, then findings |
| Structured explanations | Users see region-level summary before details |
| Parameter efficiency | Shared backbone, region-specific heads |
| Gating regularization | Prevents hallucinated findings in normal regions |

### 4.2 Why Multiplicative Gating?

The equation c_r = a_r · f_r enforces:

- If region is normal (a_r ≈ 0), all findings are suppressed
- Findings are only active when region is abnormal
- Matches clinical logic: you don't report lung findings if lungs are normal

### 4.3 Why Linear Label Predictor?

Non-linear predictors (MLP) would:
- Increase accuracy slightly
- Destroy interpretability
- Make W_y weights meaningless

Linear predictor ensures:
- Direct concept → class mapping
- Weights are human-interpretable
- Concept interventions have predictable effects

### 4.4 Why RadLex over SNOMED CT?

| Factor | RadLex | SNOMED CT |
|--------|--------|-----------|
| Domain | Radiology-specific | General clinical |
| Terminology | Matches report language | More formal |
| Imaging coverage | Excellent | Inconsistent |
| Vocabulary size | ~75K (focused) | ~350K (noisy) |

### 4.5 Why Not Learn the Hierarchy?

Fixed hierarchy (anatomy → findings) is:
- Clinically grounded
- Interpretable
- Stable across datasets

Learned hierarchy would:
- Require more data
- Potentially learn non-clinical groupings
- Be harder to explain

---

## 5. Experimental Design

### 5.1 Datasets

| Dataset | Images | Reports | Labels |
|---------|--------|---------|--------|
| MIMIC-CXR | 377K | 227K | 14 CheXpert labels |
| CheXpert | 224K | - | 14 labels |

### 5.2 Baselines

1. **Black-box CNN:** Same backbone, no concept bottleneck
2. **Flat CBM:** All concepts in single layer, no hierarchy
3. **CheXpert-labeled CBM:** Concepts from CheXpert labeler (14 concepts)
4. **Post-hoc CBM:** Concepts predicted but not used in training

### 5.3 Evaluation Metrics

**Classification:**
- AUC-ROC (per class and macro)
- F1 score

**Concept prediction:**
- Concept AUC (can model predict concepts from images?)
- Concept accuracy

**Interpretability:**
- Explanation faithfulness (do interventions change predictions correctly?)
- Radiologist evaluation (optional)

### 5.4 Ablations

1. Hierarchy vs. flat concepts
2. Multiplicative gating vs. concatenation
3. Soft labels (0.5) vs. hard labels (exclude uncertain)
4. Concept loss weight (λ_2)
5. Frequency threshold (τ)

---

## 6. Key Equations Summary

### Concept Target Construction

```
ĉ_i = 1.0 (present), 0.0 (absent), 0.5 (uncertain)
```

### Model Forward Pass

```
h = φ(x)                                # Backbone
a = σ(W_a h + b_a)                      # Region abnormality
f_r = σ(W_r h + b_r)                    # Finding prediction
c_r = a_r · f_r                         # Gating
c = [c_1; ...; c_K]                     # Full concept vector
ŷ = W_y c + b_y                         # Label prediction
```

### Training Loss

```
L = L_CE(ŷ, y) + λ_1 · L_BCE(a, â) + λ_2 · L_BCE(f, ĉ)
```

### Explanation

```
contrib_i = c_i · W_y[ŷ, i]             # Concept contribution
contrib_r = Σ_{i ∈ F_r} contrib_i       # Region contribution
```

---

## 7. Paper Writing Notes

### Title Options

1. "Hierarchical Concept Bottleneck Models with Automated Annotations for Chest X-ray Interpretation"
2. "Automating Concept Bottleneck Models for Interpretable Chest X-ray Classification"
3. "From Radiology Reports to Interpretable Models: Hierarchical CBMs via RadGraph"

### Avoid

- "Novel framework" (show, don't tell)
- "State-of-the-art" (let numbers speak)
- "Comprehensive experiments" (just describe what you did)
- "Knowledge graphs" (RadGraph is not a knowledge graph)
- Overuse of "leverage," "utilize," "furthermore"

### Emphasize

- Practical impact: eliminates manual annotation
- Clinical alignment: hierarchy matches radiologist reasoning
- Interpretability: two-level explanations
- Reproducibility: standardized vocabulary via RadLex

---

## 8. Appendix: Implementation Details

### 8.1 Concept Vocabulary (Example)

```yaml
hierarchy:
  lung:
    - opacity
    - nodule
    - consolidation
    - atelectasis
    - mass
    - infiltrate
    - fibrosis
    - pneumothorax
  
  heart:
    - cardiomegaly
    - enlarged
    - cardiac_silhouette_abnormal
  
  pleura:
    - effusion
    - thickening
    - pneumothorax
  
  mediastinum:
    - widened
    - mass
    - lymphadenopathy
  
  bone:
    - fracture
    - lesion
    - degenerative_changes
```

### 8.2 Hyperparameters

```yaml
training:
  backbone: densenet121
  backbone_pretrained: ImageNet
  region_hidden_dim: 128
  learning_rate: 1e-4
  batch_size: 32
  epochs: 50
  
  lambda_1: 0.3    # Region loss weight
  lambda_2: 0.5    # Finding loss weight
  
  uncertain_weight: 0.5    # Weight for uncertain concepts in loss
  frequency_threshold: 50  # Minimum concept occurrences

concept_extraction:
  radgraph_model: radgraph-xl
  umls_source: RadLex (primary), SNOMEDCT_US (fallback)
  similarity_threshold: 0.8
```

### 8.3 RadLex Type Filtering

```yaml
include:
  - RID5   # Imaging observation
  - RID6   # Clinical finding
  - RID4   # Pathophysiologic process
  - RID2   # Anatomic entity (chest only)

exclude:
  - RID7   # Modifier
  - RID8   # Object
  - RID10  # Imaging modality
  - RID11  # Procedure
  - RID12  # Property/Attribute
  - RID13  # Report component
  - RID14  # Treatment
  - RID15  # Medical device

conditional:
  - RID9   # Non-anatomic substance (only: air, fluid, calcium, blood)
```
