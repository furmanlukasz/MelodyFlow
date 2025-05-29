**Title:**
A Proposed Framework for Applying CHARM to EEG Data

**Prepared by:** \[Your Name]
**Affiliation:** \[Your Lab/Institution]
**Date:** \[Insert Date]

---

**Abstract:**
We propose a novel framework for applying the Complex Harmonics Decomposition (CHARM) method to EEG data, adapted from the work of Deco, Sanz Perl, and Kringelbach (2025). CHARM, originally designed to uncover low-dimensional, nonlocal structures in fMRI BOLD signals using Schrödinger-based kernel operators, is here extended for the analysis of high temporal resolution EEG data. This framework incorporates EEG-specific preprocessing, time-frequency decomposition, and nonlocal kernel construction to extract latent manifolds reflective of critical neural dynamics.

---

**1. Introduction**
The CHARM framework provides a dimensionality reduction approach based on a complex kernel derived from the Schrödinger equation:

$$
\left(i \frac{\partial}{\partial t} - \mathcal{L} \right) \hat{u} = 0,
$$

where $\mathcal{L}$ is the Laplace-Beltrami operator acting over a Riemannian manifold $\mathcal{M}$, and $\hat{u}(x,t)$ evolves according to nonlocal interference patterns. CHARM's kernel enables the projection of high-dimensional data $X \in \mathbb{R}^{M \times N}$ (with $M$ spatial channels and $N$ time samples) into a reduced non-Euclidean space that preserves long-range temporal-spatial interactions.

---

**2. Motivation for EEG Adaptation**
Unlike fMRI, EEG provides millisecond-level temporal resolution but is susceptible to noise, artefacts, and reference-dependence. To adapt CHARM effectively:

* EEG preprocessing must include filtering, referencing, artefact rejection.
* Envelope dynamics of oscillatory bands (e.g., $\alpha$, $\beta$) are more suitable analogues to BOLD fluctuations.
* Dimensionality reduction must preserve spatial dependencies while respecting the temporal structure of neural oscillations.

---

**3. Proposed CHARM-EEG Framework**

**3.1 Preprocessing:**

* Apply spatial referencing (e.g., common average, REST, Laplacian).

* Bandpass filter EEG into narrow bands $B = [b_1, b_2] \subset \mathbb{R}$.

* Extract analytic envelope via Hilbert transform:

  $$
  a(t) = |x(t) + i \mathcal{H}[x(t)]|,
  $$

  where $\mathcal{H}$ is the Hilbert transform.

* Downsample and segment into windows of length $T_w$ with stride $s$.

**3.2 Feature Representation:**
Each window $w_j \in \mathbb{R}^M$ represents the mean envelope activity over electrodes for window $j$.

Construct a data matrix:

$$
X = [w_1, w_2, \ldots, w_N] \in \mathbb{R}^{M \times N}.
$$

**3.3 Complex Kernel (Schrödinger):**
Define pairwise distances between columns of $X$:

$$
D_{ij} = \|w_i - w_j\|^2,
$$

then build the complex kernel:

$$
\hat{W}_{ij} = e^{i D_{ij} / \sigma},
$$

with scale parameter $\sigma > 0$.

---

**4. Transition Matrix and Diffusion Process**

We simulate diffusion on this graph of complex-valued similarities:

1. **Diffused Kernel:**

$$
Q(t) = |\hat{W}^t|^2
$$

where $\hat{W}^t$ is the matrix powered to $t$ steps and $|\cdot|^2$ denotes element-wise squared modulus. This matrix models nonlocal influence over $t$ steps.

2. **Row-normalization:**

$$
D_{ii} = \sum_j Q_{ij}, \quad P(t) = D^{-1} Q(t)
$$

This defines the **transition matrix** $P(t)$, a row-stochastic matrix encoding the probability of transitioning between brain states over $t$ diffusion steps.

3. **Spectral Decomposition:**

$$
P(t) = \Psi \Lambda \Psi^T
$$

Where $\Lambda$ contains eigenvalues $\lambda_1, \ldots, \lambda_k$, and $\Psi$ the corresponding eigenvectors.

4. **Manifold Embedding:**

$$
\hat{y}_i = [\lambda_1 \psi_1(i), \lambda_2 \psi_2(i), \ldots, \lambda_k \psi_k(i)]
$$

This low-dimensional coordinate $\hat{y}_i \in \mathbb{R}^k$ captures the critical, nonlocal geometry of EEG dynamics.

---

**5. Validation Metrics**

* **Edge-Centric Metastability (ECM):** Quantify dynamic fluctuations in manifold space vs. source.
* **Spectral Gap:** Determine optimal $k$ via eigenspectrum.
* **Temporal Smoothness:** Assess clustering and transitions across $\hat{y}_i$.

---

**6. Conclusion**
The CHARM-EEG framework adapts a powerful nonlocal manifold learning method to time-resolved EEG data. By focusing on band-limited envelope representations and leveraging complex diffusion kernels, this method offers a principled way to extract low-dimensional, critical, and interpretable spatiotemporal patterns.

---

**References:**
Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics reveal low-dimensional manifolds of critical brain dynamics. *Physical Review E, 111*(1), 014410.
