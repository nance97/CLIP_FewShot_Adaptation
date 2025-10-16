# 🧠 Few-Shot Adaptation of CLIP via Manifold Transport (C2T2D)
*A lightweight, teacher-free head that geodesically transports CLIP text anchors to align with image embeddings under domain shift.*

---

## 🎯 Overview
This project studies how to adapt **CLIP** to a visual domain with **very few labeled samples** while **preserving zero-shot generalization**.  
Instead of full fine-tuning or prompt learning, we introduce a **manifold-aware transport head** (C2T2D) that makes **small, valid moves on CLIP’s unit-sphere text manifold** to better align text anchors with image embeddings.

**Key idea:** keep CLIP frozen; learn a tiny head that predicts **which way** and **how far** to move each class’s text anchor for a given image, then score with standard CLIP logits.

---

## 🔧 Method (what the notebook implements)

### 1) Canonical text anchors
- Build one **L2-normalized** text embedding per class (the “anchor” \(z_c\)), using a CLIP text template.

### 2) Phrase-bank → shared semantic atlas → class tangent bases
CLIP text features lie on the **unit hypersphere**. To respect this geometry we work in **tangent spaces**:
- For each class \(c\), collect **multiple prompt variants** (phrase bank) and project their differences into the **tangent space** at \(z_c\).
- Learn a **shared semantic atlas** across classes; then derive **mixed, orthonormal class-specific bases** \(G_c \in \mathbb{R}^{d\times K}\) (QR re-orthonormalization ensures valid tangent directions).

> Intuition: \(G_c\) captures *how the concept can bend* locally (semantically valid directions) without leaving CLIP’s manifold.

### 3) Few-shot image prototypes (optional but implemented)
- From few-shot image embeddings \(\hat{u}_i\), compute **class prototypes** (cluster centers) and **project them into the tangent at \(z_c\)**.  
- These prototype-derived directions complement \(G_c\) with **visual variation** grounded in data.

### 4) C2T2D transport head (teacher-free)
Given an image embedding \(x\) and a class anchor \(z_c\):
- Build features \(h_{x,c} = [x,\, z_c,\, x^\top z_c,\, 1-(x^\top z_c)]\).
- **Direction MLP** predicts mixture weights \(\beta_c\) over the \(K\) tangent directions in \(G_c\) → gives a unit tangent \(u_c\).
- **Step MLP** predicts a non-negative **step size** \(s_c\).
- **Geodesic update** (on the sphere) transports the anchor:
  \[
  \tilde{z}_c(x) \;=\; \cos(s_c)\,z_c \;+\; \sin(s_c)\,u_c,
  \]
  keeping \(\tilde{z}_c\) on the unit sphere (geometry-preserving).
- Compute **CLIP-style logits** \( \ell_c(x)=\gamma\, x^\top \tilde{z}_c(x)\) and apply standard contrastive classification.

> Same motion, two views: image-toward-text or text-toward-image. The notebook uses **text-side transport**.

---

## 🧪 Training objectives (as coded)
A mix of **contrastive**, **stability**, and **geometry** terms:

1. **InfoNCE (Cross-Entropy)**  
   Temperature-scaled CE over logits \( \ell(x) \) encourages correct alignment **after transport**.

2. **Dropout Consistency**  
   Evaluate the head twice per batch under dropout → encourage transported anchors \(W, W'\) to **agree** (smooth atlas mixtures).

3. **Span Regularizer**  
   Penalize movement **outside the class span** (keep updates inside the tangent basis subspace).

4. **Step Regularizer**  
   Bound step magnitudes \(s_c\) to keep moves **small and local** (avoid over-deformation).

5. **Zero-Shot Tether (ZS-tether)**  
   Keep transported anchors near originals to **preserve CLIP’s zero-shot behavior**, mitigating catastrophic forgetting.

---

## 📊 Evaluation protocol
- **Datasets:** Flowers102.  
- **Splits:** explicit **base vs. novel** categories; **few-shot k** per base class.  
- **Metrics:**  
  - **Harmonic Mean** \(HM = \frac{2}{1/\text{base} + 1/\text{novel}}\) to capture the retain-and-transfer trade-off.

> The notebook includes zero-shot baselines, few-shot data extraction, and evaluation helpers for base/novel/H-mean.

---

## 📈 What to expect (qualitative summary)
- Transport yields **higher base-class accuracy** while keeping novel performance close to the zero-shot baseline (retain + adapt).  
- Works with CLIP **as-is** (frozen); only the small head is trained.

---


## 💡 Why this approach
- **Manifold-aware:** updates are **geodesic**; anchors stay on CLIP’s sphere (no arbitrary warping).  
- **Data-efficient:** only a **tiny transport head** is learned; CLIP stays frozen.  
- **Transfer-friendly:** phrase-bank atlas + prototypes combine **semantic** and **visual** variation; ZS-tether helps **avoid forgetting**.

---

## 👩‍💻 Author
**Nancy Kalaj** — Master’s in Artificial Intelligence Systems, University of Trento  
Focus: *Multimodal learning • Domain adaptation • Efficient alignment.*

